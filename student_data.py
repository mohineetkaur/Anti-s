#!/usr/bin/env python3
"""
student_data.py  -  training pairs for the student, built on the fly.

* The co-registration correction is applied IN MEMORY when a sample is drawn. No GeoTIFF is
  rewritten, so changing the correction later only means re-running with a new JSON/CSV.
* Where the SAR/RGB shift is not trustworthy (your west block) patches are skipped
  automatically. "Trustworthy" is decided from registration_samples_v2.csv: a location counts as
  reliable only if every measured patch within `radius` px agrees with the fitted correction.
  Map more of the west later, re-run, and the mask grows by itself. Nothing is hard-coded.
* Train/val are split spatially (val = whole map cells, train windows never touch them).

Numpy only. Run  python student_data.py  to self-test the label geometry on synthetic maps.

Conventions: everything is in the pixel-centre convention of the main script. "ds" = down-scale
factor (block mean). A sample is (patch, window, label): patch = SAR crop, window = RGB crop,
label = (x, y) of the patch's top-left pixel inside the window, in DOWN-SCALED pixels.
"""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np


def block_mean(a, ds):
    """Exact integer-factor area average (what cv2.INTER_AREA does), keeps pixel blocks aligned."""
    h, w = (a.shape[0] // ds) * ds, (a.shape[1] // ds) * ds
    s = np.ascontiguousarray(a[:h, :w]).reshape(h // ds, ds, w // ds, ds).sum(axis=(1, 3), dtype=np.uint32)
    return (s / float(ds * ds) + 0.5).astype(np.uint8)


class Registration:
    """true RGB position of a SAR point = g + shift(g), g = position the geotransforms give."""

    def __init__(self, json_path):
        j = json.load(open(json_path))
        self.coef = np.asarray(j["coef"], float)          # rows: a ; B^T   (constant model: just a)
        self.g0 = np.asarray(j["g0"], float)
        self.affine = len(self.coef) == 3

    def shift(self, g):
        g = np.asarray(g, float)
        s = self.coef[0].copy()
        if self.affine:
            s = s + (g - self.g0) @ self.coef[1:]
        return s


class ReliabilityMask:
    def __init__(self, csv_path, reg, gate_px=25.0, bad_radius=1200.0, support_radius=1500.0, min_nb=2):
        """Reliable at p  <=>  no measured patch within bad_radius disagrees with the fitted
        correction by more than gate_px, AND >= min_nb agreeing patches lie within support_radius."""
        rows = list(csv.DictReader(open(csv_path)))
        self.g = np.array([[float(r["gx"]), float(r["gy"])] for r in rows])
        C = np.array([[float(r["Cx"]), float(r["Cy"])] for r in rows])
        D = np.array([[float(r["Dx"]), float(r["Dy"])] for r in rows])
        d = (C - D) / 2
        model = np.array([reg.shift(p) for p in self.g])
        self.resid = np.hypot(*(d - model).T)
        self.good = self.resid <= gate_px
        self.bad_radius, self.support_radius, self.min_nb = bad_radius, support_radius, min_nb

    def reliable(self, p):
        dist = np.hypot(self.g[:, 0] - p[0], self.g[:, 1] - p[1])
        if (~self.good[dist <= self.bad_radius]).any():
            return False
        return bool((self.good & (dist <= self.support_radius)).sum() >= self.min_nb)

    def area_share(self, W, H, step=250):
        xs, ys = np.arange(step / 2, W, step), np.arange(step / 2, H, step)
        ok = [self.reliable((x, y)) for y in ys for x in xs]
        return float(np.mean(ok))


class PairSampler:
    def __init__(self, sar, rgb, sar_to_rgb, reg, mask, ds=4, patch_orig=1024, win_orig=2048,
                 val_cells=((2, 1), (1, 2)), grid=3, border=8):
        """sar, rgb: uint8 grey maps at full resolution (as returned by load_all).
        sar_to_rgb(x, y) -> (gx, gy): nominal SAR pixel -> RGB pixel from the geotransforms."""
        self.ds, self.reg, self.mask, self.sar_to_rgb = ds, reg, mask, sar_to_rgb
        self.sar, self.rgb = block_mean(sar, ds), block_mean(rgb, ds)
        self.P, self.W, self.bd = patch_orig // ds, win_orig // ds, border
        Hr, Wr = self.rgb.shape
        cw, ch = Wr / grid, Hr / grid
        self.val_rects = [(i * cw, j * ch, (i + 1) * cw, (j + 1) * ch) for i, j in val_cells]
        if self.W > min(cw, ch):
            raise ValueError("window larger than a val cell; use a smaller window or grid")

    def _hits_val(self, w0):
        W = self.W
        return any(w0[0] < x1 and w0[0] + W > x0 and w0[1] < y1 and w0[1] + W > y0
                   for x0, y0, x1, y1 in self.val_rects)

    @staticmethod
    def _aug(x, rng, speckle):
        x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
        if speckle:
            L = rng.uniform(4, 16)
            x = x * rng.gamma(L, 1.0 / L, x.shape)
        return np.clip(x, 0, 1).astype(np.float32)

    def sample(self, rng, split="train", aug=None):
        aug = (split == "train") if aug is None else aug
        P, W, ds, bd = self.P, self.W, self.ds, self.bd
        Hs, Ws = self.sar.shape
        Hr, Wr = self.rgb.shape
        for _ in range(50000):
            if split == "val":
                x0, y0, x1, y1 = self.val_rects[rng.integers(len(self.val_rects))]
                u0, v0 = int(rng.uniform(x0, x1 - P)), int(rng.uniform(y0, y1 - P))
            else:
                u0, v0 = int(rng.integers(0, Ws - P + 1)), int(rng.integers(0, Hs - P + 1))
            if not (0 <= u0 <= Ws - P and 0 <= v0 <= Hs - P):
                continue
            half = (P - 1) / 2 * ds
            c = np.array([u0 * ds + (ds - 1) / 2 + half, v0 * ds + (ds - 1) / 2 + half])
            g = np.array(self.sar_to_rgb(*c), float)                 # nominal position
            if not self.mask.reliable(g):
                continue
            gt = g + self.reg.shift(g)                               # corrected, in memory only
            tl = (gt - half - (ds - 1) / 2) / ds                     # true top-left, RGB ds frame
            lo = np.maximum(tl - (W - P - bd), 0.0)
            hi = np.minimum(tl - bd, [Wr - W, Hr - W])
            if split == "val":
                lo = np.maximum(lo, [x0, y0])
                hi = np.minimum(hi, [x1 - W, y1 - W])
            a, b = np.ceil(lo).astype(int), np.floor(hi).astype(int)
            if (b < a).any():
                continue
            w0 = np.array([rng.integers(a[0], b[0] + 1), rng.integers(a[1], b[1] + 1)])
            if split == "train" and self._hits_val(w0):
                continue
            patch = self.sar[v0:v0 + P, u0:u0 + P]
            if (patch < 1).mean() > 0.05 or patch.std() < 8:
                continue
            win = self.rgb[w0[1]:w0[1] + W, w0[0]:w0[0] + W]
            p, w = patch.astype(np.float32) / 255.0, win.astype(np.float32) / 255.0
            if aug:
                p, w = self._aug(p, rng, True), self._aug(w, rng, False)
            return dict(patch=p, window=w, label=(tl - w0).astype(np.float32), g=g, w0=w0)
        raise RuntimeError("could not draw a valid sample (mask too strict or val cells unreliable)")

    def fixed_val(self, n, seed=123):
        rng = np.random.default_rng(seed)
        return [self.sample(rng, "val", aug=False) for _ in range(n)]


# ------------------------------------------------------------------------------------------
def selftest():
    """Synthetic maps where the true shift is known; checks labels, split and mask."""
    rng = np.random.default_rng(0)
    N, ds = 3200, 4
    rgb = np.kron(rng.integers(0, 256, (N // 4, N // 4), dtype=np.uint8), np.ones((4, 4), np.uint8))
    dx, dy = -44, 4                       # SAR[y, x] = RGB[y + dy, x + dx]   (true shift)
    sar = np.zeros_like(rgb)
    sar[0:N - dy, -dx:N] = rgb[dy:N, 0:N + dx]
    tmp = Path(tempfile.mkdtemp())
    json.dump(dict(coef=[[dx, dy]], g0=[0, 0]), open(tmp / "reg.json", "w"))
    with open(tmp / "s.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "y0", "gx", "gy", "Cx", "Cy", "Dx", "Dy", "Cinl", "Dinl"])
        for gx in range(300, N, 400):
            for gy in range(300, N, 400):
                bad = gx < 1000 and gy < 2200                         # a "west block"
                s = np.array([120.0, -28.0]) if bad else np.array([dx, dy]) + rng.normal(0, 1.5, 2)
                w.writerow([0, 0, gx, gy, s[0], s[1], -s[0], -s[1], 500, 500])
    reg = Registration(tmp / "reg.json")
    mask = ReliabilityMask(tmp / "s.csv", reg)
    sp = PairSampler(sar, rgb, lambda x, y: (x, y), reg, mask, ds=ds, patch_orig=256, win_orig=512)
    print(f"mask: {mask.good.sum()}/{len(mask.good)} calibration points good, "
          f"{100 * mask.area_share(N, N):.0f}% of the map reliable")

    bad_hit = exact = 0
    P = sp.P
    n = 0
    for split in ("train", "val"):
        for _ in range(150):
            s = sp.sample(rng, split, aug=False)       # exact content comparison needs no augmentation
            n += 1
            lx, ly = s["label"]
            ix, iy = int(round(lx)), int(round(ly))
            crop = s["window"][iy:iy + P, ix:ix + P]
            exact += int(abs(lx - ix) < 1e-4 and abs(ly - iy) < 1e-4 and np.allclose(crop, s["patch"], atol=1e-6))
            bad_hit += int(s["g"][0] < 1000 and s["g"][1] < 2200)
    print(f"label geometry exact: {exact}/{n}   samples inside the bad block: {bad_hit}")
    # split hygiene: train windows never touch a val cell; val windows lie fully inside one
    rng2 = np.random.default_rng(5)
    leak = outside = 0
    for _ in range(300):
        leak += int(sp._hits_val(sp.sample(rng2, "train")["w0"]))
        w0 = sp.sample(rng2, "val")["w0"]
        outside += int(not any(x0 <= w0[0] and w0[0] + sp.W <= x1 and y0 <= w0[1] and w0[1] + sp.W <= y1
                               for x0, y0, x1, y1 in sp.val_rects))
    print(f"train windows touching val cells: {leak}/300   val windows not inside a val cell: {outside}/300")
    assert exact == n and bad_hit == 0 and leak == 0 and outside == 0, "SELFTEST FAILED"
    print("SELFTEST OK")


if __name__ == "__main__":
    selftest()
