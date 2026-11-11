#!/usr/bin/env python3
"""
train_student.py  -  lightweight Siamese correlation student, trained on CPU from student_data.py.

How it works
  SAR patch  -> small CNN (stride 8) -> feature map
  RGB window -> small CNN (stride 8) -> feature map
  one cross-correlation (a single conv) slides the patch features over the window features
  -> heat-map -> soft-argmax = sub-cell position of the patch inside the window.
  One forward pass replaces the teacher's match + RANSAC for that window.

Labels come from student_data.PairSampler: the registration correction is applied in memory and
the unreliable west block is skipped automatically. No files are changed.

Usage (same folder as sar_rgb_match_check_v2.py, offset_diagnostic.py, student_data.py):
    python train_student.py --selftest                 # 10 s: checks shapes + 8-px cell geometry
    python train_student.py --minutes 30               # train for 30 min, keep best-on-val
Options: --ds 4 --patch 1024 --win 2048 --batch 8 --lr 2e-3 --threads N --out student.pt
"""
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 8          # feature cells are 8 (down-scaled) pixels apart


# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------
def cbr(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, c=48):
        super().__init__()
        self.net = nn.Sequential(cbr(1, 16, 2), cbr(16, 32, 2), cbr(32, 32, 1),
                                 cbr(32, 64, 2), cbr(64, 64, 1), nn.Conv2d(64, c, 1))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)            # unit-length feature per cell


class Student(nn.Module):
    def __init__(self, c=48):
        super().__init__()
        self.enc_s, self.enc_r = Encoder(c), Encoder(c)   # separate: SAR and optical differ
        self.log_scale = nn.Parameter(torch.tensor(3.0))  # logits = exp(.) * mean cosine

    def forward(self, patch, window):
        ft, fw = self.enc_s(patch), self.enc_r(window)    # (B,C,Ht,Wt), (B,C,Hw,Ww)
        B, C, Ht, Wt = ft.shape
        out = F.conv2d(fw.reshape(1, B * C, fw.shape[2], fw.shape[3]), ft, groups=B)   # (1,B,Ho,Wo)
        return self.log_scale.exp() * out[0] / (Ht * Wt)  # (B,Ho,Wo) logits over placements


# ------------------------------------------------------------------------------------------
# loss / localisation (positions in feature CELLS; x = column, y = row)
# ------------------------------------------------------------------------------------------
def soft_argmax(logits):
    B, Ho, Wo = logits.shape
    p = F.softmax(logits.reshape(B, -1), 1).reshape(B, Ho, Wo)
    xs, ys = torch.arange(Wo).float(), torch.arange(Ho).float()
    return torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], 1)


def loss_fn(logits, lab, sigma=0.75, l1_w=1.0):
    B, Ho, Wo = logits.shape
    xs, ys = torch.arange(Wo).float().view(1, 1, Wo), torch.arange(Ho).float().view(1, Ho, 1)
    d2 = (xs - lab[:, 0].view(-1, 1, 1)) ** 2 + (ys - lab[:, 1].view(-1, 1, 1)) ** 2
    t = torch.exp(-d2 / (2 * sigma ** 2))
    t = t / t.sum((1, 2), keepdim=True)
    logp = F.log_softmax(logits.reshape(B, -1), 1).reshape(B, Ho, Wo)
    ce = -(t * logp).sum((1, 2)).mean()
    l1 = F.smooth_l1_loss(soft_argmax(logits), lab)
    return ce + l1_w * l1, (float(ce), float(l1))


def argmax_xy(logits):
    B, Ho, Wo = logits.shape
    idx = logits.reshape(B, -1).argmax(1)
    return torch.stack([(idx % Wo).float(), (idx // Wo).float()], 1)


@torch.no_grad()
def peak_xy(logits, r=2):
    """argmax, then a local soft-argmax over a (2r+1)^2 neighbourhood -> sub-cell position."""
    B, Ho, Wo = logits.shape
    arg = argmax_xy(logits)
    out = []
    for b in range(B):
        px, py = int(arg[b, 0]), int(arg[b, 1])
        x0, x1, y0, y1 = max(0, px - r), min(Wo, px + r + 1), max(0, py - r), min(Ho, py + r + 1)
        sub = logits[b, y0:y1, x0:x1]
        p = F.softmax(sub.reshape(-1), 0).reshape(sub.shape)
        xs, ys = torch.arange(x0, x1).float(), torch.arange(y0, y1).float()
        out.append(torch.stack([(p.sum(0) * xs).sum(), (p.sum(1) * ys).sum()]))
    return torch.stack(out)


def to_tensors(samples):
    p = torch.from_numpy(np.stack([s["patch"] for s in samples]))[:, None] - 0.5
    w = torch.from_numpy(np.stack([s["window"] for s in samples]))[:, None] - 0.5
    lab = torch.from_numpy(np.stack([s["label"] for s in samples])) / STRIDE
    return p, w, lab


@torch.no_grad()
def evaluate(model, val, mpp, ds, success_px=10.0):
    model.eval()
    e_sub, e_arg = [], []
    for i in range(0, len(val), 16):
        p, w, lab = to_tensors(val[i:i + 16])
        lg = model(p, w)
        for pred, store in ((peak_xy(lg), e_sub), (argmax_xy(lg), e_arg)):
            store += torch.hypot(pred[:, 0] - lab[:, 0], pred[:, 1] - lab[:, 1]).tolist()
    to_m = STRIDE * ds * mpp
    e, a = np.array(e_sub) * to_m, np.array(e_arg) * to_m
    return dict(median=float(np.median(e)), mean=float(e.mean()), p90=float(np.percentile(e, 90)),
                max=float(e.max()), argmax_median=float(np.median(a)),
                success=float((e <= success_px * mpp).mean()))


@torch.no_grad()
def bench(model, P, W, n=20):
    model.eval()
    p, w = torch.randn(1, 1, P, P), torch.randn(1, 1, W, W)
    model(p, w)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); model(p, w); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


# ------------------------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    P, W = 64, 128
    m = Student()
    m.enc_r.load_state_dict(m.enc_s.state_dict())
    m.eval()
    rng = np.random.default_rng(0)
    win = torch.from_numpy(rng.random((6, 1, W, W), dtype=np.float32)) - 0.5
    offs = [(0, 0), (16, 8), (48, 48), (8, 40), (24, 0), (40, 16)]       # multiples of 8, <= W-P
    pat = torch.stack([win[i, :, oy:oy + P, ox:ox + P] for i, (ox, oy) in enumerate(offs)])
    with torch.no_grad():
        lg = m(pat, win)
    n = W // STRIDE - P // STRIDE + 1
    assert tuple(lg.shape) == (6, n, n), f"bad logits shape {tuple(lg.shape)}"
    lab = torch.tensor([[ox / STRIDE, oy / STRIDE] for ox, oy in offs])
    hit = int((argmax_xy(lg) == lab).all(1).sum())
    print(f"geometry: identical crops, shared weights -> correlation peak at the true cell in {hit}/6")
    assert hit == 6, "cell geometry mismatch (label convention vs encoder stride)"

    m = Student()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for i in range(60):
        loss, _ = loss_fn(m(pat * 1.3 + 0.05, win), lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if i < 5: first = (first or 0) + float(loss) / 5
        if i >= 55: last = (last or 0) + float(loss) / 5
    print(f"training: loss {first:.2f} -> {last:.2f} over 60 steps")
    assert last < first, "loss did not decrease"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--val-n", type=int, default=96)
    ap.add_argument("--init", help="start from these weights (e.g. student_last.pt)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="student.pt")
    ap.add_argument("--reg-json")
    ap.add_argument("--samples-csv")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.threads:
        torch.set_num_threads(a.threads)
    for v, nm in ((a.patch, "patch"), (a.win, "win")):
        if v % (a.ds * STRIDE):
            raise SystemExit(f"--{nm} must be a multiple of ds*{STRIDE} = {a.ds * STRIDE}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sar_rgb_match_check_v2 as m
    from offset_diagnostic import load_all
    from student_data import Registration, ReliabilityMask, PairSampler

    out_dir = Path(m.out_path).parent
    reg = Registration(a.reg_json or out_dir / "registration_offset_v2.json")
    mask = ReliabilityMask(a.samples_csv or out_dir / "registration_samples_v2.csv", reg)
    sar, rgb_gray, sar_geo, rgb_geo = load_all()
    mpp = m.dist_m(rgb_geo, (0, 0), (1000, 0)) / 1000.0
    H, W_ = rgb_gray.shape
    print(f"[mask] {int((~mask.good).sum())}/{len(mask.good)} calibration patches disagree with the correction; "
          f"{100 * mask.area_share(W_, H):.0f}% of the map is treated as reliable")
    sampler = PairSampler(sar, rgb_gray, lambda x, y: m.sar_pix_to_rgb_pix(sar_geo, rgb_geo, x, y),
                          reg, mask, ds=a.ds, patch_orig=a.patch, win_orig=a.win)
    val = sampler.fixed_val(a.val_n)
    print(f"[data] {a.val_n} fixed val samples; 1 feature cell = {STRIDE * a.ds * mpp:.1f} m; "
          f"label noise floor ~ registration scatter (about 2-3 m)")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Student()
    if a.init:
        model.load_state_dict(torch.load(a.init)["state"])
        print(f"[init] weights loaded from {a.init}")
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    last_path = Path(a.out).with_name(Path(a.out).stem + "_last.pt")
    t0, budget, step, best = time.time(), a.minutes * 60, 0, float("inf")
    cfg = dict(ds=a.ds, patch=a.patch, win=a.win, stride=STRIDE, c=48)
    stats = None
    while True:
        frac = (time.time() - t0) / budget
        done = frac >= 1.0
        if not done:
            lr = a.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            p, w, lab = to_tensors([sampler.sample(rng, "train") for _ in range(a.batch)])
            loss, (ce, l1) = loss_fn(model(p, w), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
        if done or step % a.eval_every == 0:
            stats = evaluate(model, val, mpp, a.ds)
            stats["score"] = stats["median"] + 0.25 * stats["p90"]     # median alone hides wrong-peak failures
            torch.save(dict(state=model.state_dict(), cfg=cfg, val=stats), last_path)
            tag = ""
            if stats["score"] < best:
                best = stats["score"]
                torch.save(dict(state=model.state_dict(), cfg=cfg, val=stats), a.out)
                tag = "  *saved"
            print(f"[{(time.time() - t0) / 60:5.1f} min] step {step:5d} ({step / max(time.time() - t0, 1):.2f}/s) "
                  f"loss {float(loss):.3f}  val median {stats['median']:.1f} m  p90 {stats['p90']:.1f} m  "
                  f"<=10px {100 * stats['success']:.0f}%  (argmax-only median {stats['argmax_median']:.1f} m){tag}")
        if done:
            break

    ck = torch.load(a.out)
    model.load_state_dict(ck["state"])
    ms = bench(model, a.patch // a.ds, a.win // a.ds)
    s = ck["val"]
    print(f"\n=== BEST STUDENT ({a.out}) ===")
    print(f"  held-out cells: median {s['median']:.1f} m  p90 {s['p90']:.1f} m  max {s['max']:.1f} m  "
          f"success(<=10px) {100 * s['success']:.0f}%")
    print(f"  speed: {ms:.0f} ms per window on this CPU   (teacher: about 6800 ms per match call in your logs)")


if __name__ == "__main__":
    main()
