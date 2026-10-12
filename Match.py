
# # sar_path = r"C:\Users\palak\Downloads\sar_grd_db.tif"
# # rgb_path = r"C:\Users\palak\Downloads\sar_rgb_output\rgb_sar_aligned.tif"

# # # --- MatchAnything code + weights ---
# # repo_path = r"C:\Users\palak\AppData\Local\Programs\Microsoft VS Code\MatchAnything_download\imcui\third_party\MatchAnything" 

# #!/usr/bin/env python3
# """
# sar_rgb_match_check.py

# Sanity-check MatchAnything (ELoFTR variant) on ONE SAR map and ONE RGB map.

# What it does
#   1. Loads the SAR map and the RGB map.
#   2. Randomly samples a square patch from the SAR map.
#   3. Slides overlapping search windows over the RGB map; the patch is matched against
#      every window with MatchAnything-ELoFTR; the window with the most RANSAC inliers wins.
#   4. Estimates patch -> RGB-map transform and draws:
#        - the predicted location of the patch as a red box on the RGB map
#        - (optional, aligned = True) the expected location as a green box + pixel error
#        - correspondence lines between the patch and the best RGB window

# Why windows instead of the whole map?
#   The matcher is scale sensitive. If the RGB map is much larger than the patch, resizing
#   the whole map to ~832 px would shrink it far more than the patch. Windows keep both
#   images at the same scale (same resize factor is applied to patch and window).

# Assumption: SAR and RGB maps have roughly the same ground resolution (pixel size).
# If they do not, set sar_scale, or aligned = True when both maps cover the same extent.

# HOW TO USE: edit the CONFIG block below (paths and options), then run
#     python sar_rgb_match_check.py
# """
# import os
# import sys
# from pathlib import Path
# from types import SimpleNamespace
# import rasterio

# # OpenCV refuses images above ~1 gigapixel by default; lift the limit (must be set before importing cv2)
# os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

# import cv2
# import matplotlib
# import numpy as np
# from PIL import Image

# Image.MAX_IMAGE_PIXELS = None  # allow huge maps

# # ======================================================================================
# # CONFIG - edit these variables, then run:  python sar_rgb_match_check.py
# # (use r"..." strings so Windows backslashes work, e.g. r"D:\data\sar_map.tif")
# # ======================================================================================
# # --- your data ---

# import sys
# from pathlib import Path

# ELOFTR_SRC_PARENT = Path(
#     r"C:\Users\palak\Documents\Plk\DIAT\Semester 3\DSMAC\Code\EfficientLoFTR-main"
# )

# if str(ELOFTR_SRC_PARENT) not in sys.path:
#     sys.path.insert(0, str(ELOFTR_SRC_PARENT))



# sar_path = r"C:\Users\palak\Downloads\sar_rgb_output\sar_on_rgb_grid.tif"
# rgb_path = r"C:\Users\palak\Downloads\sar_rgb_output\rgb_sar_aligned.tif"

# # --- MatchAnything code + weights ---
# # repo_path = r"C:\Users\palak\AppData\Local\Programs\Microsoft VS Code\MatchAnything_download\imcui\third_party\MatchAnything" 
# repo_path = r"C:\Users\palak\Documents\Plk\DIAT\Semester 3\DSMAC\Code\MatchAnything_restore"
# ckpt_path = None   # None = auto: finds weights inside repo_path, else downloads them (needs: pip install gdown)
# matcher = "matchanything"   # "matchanything", or "orb" (debug only: tests plumbing without weights)
# device = None               # None = cuda if available, else cpu

# # --- output ---
# out_path = "match_result.png"   # with trials > 1: match_result_00.png, _01.png, ...
# no_show = False                 # True = only save, do not open a matplotlib window

# # --- patch / search ---
# patch_size = 2048      # SAR patch side in pixels (after sar_scale)
# patch_xy = None         # e.g. (1200, 800) to fix the patch top-left instead of random
# trials = 1              # number of random patches
# seed = None             # None = random (the seed used is printed so you can reproduce it)
# window = 0              # RGB search window side in px (0 = 2 x patch_size)
# stride = 0              # window stride in px (0 = window / 2)
# img_resize = 832        # long side of each search window after resize (832 = training size)
# match_thr = 0.1         # coarse match confidence threshold (raise = fewer, cleaner matches)

# # --- map geometry ---
# aligned = False         # True if both maps cover the same area: SAR is rescaled to the RGB size
#                         # and the script reports the error against the expected location
# sar_scale = None        # resize SAR by this factor first (None = 1.0, or auto when aligned = True)
# search_radius = 0       # with aligned = True: only search windows within this many px of the truth
# sar_stretch = False     # True = force percentile contrast stretch on the SAR map
# downscale = 1           # shrink BOTH maps by this integer factor after loading (2, 4, ...): faster, less memory

# # --- verification ---
# model = "similarity"    # "similarity" (rotation+scale+shift) or "homography"
# ransac_thr = 3.0        # RANSAC reprojection threshold in RGB-map pixels
# min_inliers = 12        # fewer inliers than this = "no reliable match"
# success_px = 10.0       # with aligned = True: centre error counted as success
# # ======================================================================================


# # --------------------------------------------------------------------------------------
# # image IO
# # --------------------------------------------------------------------------------------
# def to_u8(a, stretch=True):
#     """Convert any dtype to uint8. Non-uint8 data is percentile-stretched (2-98%).
#     Works in row chunks so huge maps do not need a full float32 copy."""
#     if a.dtype == np.uint8 and not stretch:
#         return a
#     step = max(1, int(np.sqrt(a.size / 4_000_000)))  # ~4M-pixel sample for the percentiles
#     sample = a[::step, ::step].astype(np.float32)
#     vals = sample[np.isfinite(sample) & (sample != 0)]  # 0 is usually nodata
#     if vals.size == 0:
#         return np.zeros(a.shape, np.uint8)
#     lo, hi = np.percentile(vals, (2, 98))
#     if hi <= lo:
#         hi = lo + 1e-6
#     out = np.empty(a.shape, np.uint8)
#     for r in range(0, a.shape[0], 2048):
#         blk = a[r:r + 2048].astype(np.float32)
#         ok = np.isfinite(blk) & (blk != 0)
#         o = np.clip((blk - lo) / (hi - lo), 0, 1) * 255.0
#         o[~ok] = 0
#         out[r:r + 2048] = o.astype(np.uint8)
#     return out




# import rasterio

# def load_sar(path, force_stretch=True):

#     with rasterio.open(path) as src:

#         img = src.read(1).astype(np.float32)
#         nodata = src.nodata

#         print("[SAR]")
#         print("  dtype:", src.dtypes[0])
#         print("  shape:", img.shape)
#         print("  nodata:", nodata)
#         print("  raw min/max:", np.nanmin(img), np.nanmax(img))

#         # Valid pixel mask
#         valid = np.isfinite(img)

#         if nodata is not None:
#             valid &= (img != nodata)

#         # zero = background
#         valid &= (img != 0)

#         vals = img[valid]

#         if vals.size == 0:
#             raise RuntimeError("No valid SAR pixels found!")

#         print("  valid pixels:", vals.size)
#         print("  valid min/max:", vals.min(), vals.max())

#         lo, hi = np.percentile(vals, [2, 98])

#         print("  valid percentiles:",
#               np.percentile(vals, [1, 2, 50, 98, 99]))

#         # Output directly as uint8
#         out = np.zeros(img.shape, dtype=np.uint8)

#         # Process by rows to avoid huge temporary arrays
#         for r in range(0, img.shape[0], 1024):

#             block = img[r:r+1024]
#             mask = valid[r:r+1024]

#             scaled = np.zeros(block.shape, dtype=np.float32)

#             scaled[mask] = (
#                 (block[mask] - np.float32(lo))
#                 / np.float32(hi - lo)
#             )

#             scaled = np.clip(scaled, 0.0, 1.0)

#             out[r:r+1024][mask] = (
#                 scaled[mask] * 255
#             ).astype(np.uint8)

#         print("  stretch:", lo, "->", hi)

#         return out

# def load_rgb(path):
#     img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise FileNotFoundError(path)
#     if img.ndim == 2:
#         img = np.stack([img] * 3, -1)
#     img = img[..., :3]  # drop alpha / extra bands
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if img.dtype != np.uint8:
#         img = to_u8(img)
#     return img


# # --------------------------------------------------------------------------------------
# # matchers
# # --------------------------------------------------------------------------------------
# def _prep(gray, scale):
#     """Resize by `scale`, floor to a multiple of 32, pad to a square (as in the official demo)."""
#     h, w = gray.shape
#     w1, h1 = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
#     w2, h2 = max(32, w1 // 32 * 32), max(32, h1 // 32 * 32)
#     img = np.asarray(Image.fromarray(gray).resize((w2, h2), Image.LANCZOS), dtype=np.float32)
#     s = max(h2, w2)
#     pad = np.zeros((s, s), np.float32)
#     pad[:h2, :w2] = img
#     mask = np.zeros((s, s), bool)
#     mask[:h2, :w2] = True
#     return dict(pad=pad, mask=mask, img=img, sx=w / w2, sy=h / h2, size_res=(h1, w1))


# class MatchAnythingELoFTR:
#     """Thin wrapper that mirrors the official HF-demo wrapper (imcui/hloc/matchers/matchanything.py)."""

#     def __init__(self, repo, ckpt, device=None, img_resize=832, thr=0.1):
#         import torch
#         import torch.nn.functional as F

#         self.torch, self.F = torch, F
#         repo = Path(repo).resolve()
#         sys.path.append(str(repo))
#         sys.path.append(str(repo.parent))
#         try:
#             from MatchAnything.src.config.default import get_cfg_defaults
#             from MatchAnything.src.lightning.lightning_loftr import PL_LoFTR
#         except ImportError:
#             from src.config.default import get_cfg_defaults
#             from src.lightning.lightning_loftr import PL_LoFTR

#         # cfg = get_cfg_defaults()
#         # cfg.merge_from_file(str(repo / "configs/models/eloftr_model.py"))

       
#         from src.config.default import _CN as cfg

#         model_config = repo / "configs" / "models" / "eloftr_model.py"
#         exec(compile(
#             model_config.read_text(encoding="utf-8"),
#             str(model_config),
#             "exec"
#         ))


#         cfg = get_cfg_defaults()

#         model_config = repo / "configs/models/eloftr_model.py"
#         exec(compile(model_config.read_text(encoding="utf-8"), str(model_config), "exec"))

#         if cfg.LOFTR.COARSE.ROPE:
#             assert cfg.DATASET.NPE_NAME is not None
#         if cfg.DATASET.NPE_NAME == "megadepth":
#             cfg.LOFTR.COARSE.NPE = [832, 832, img_resize, img_resize]
#         cfg.METHOD = "matchanything_eloftr"
#         cfg.LOFTR.MATCH_COARSE.THR = thr

#         self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
#         self.net = PL_LoFTR(cfg, pretrained_ckpt=str(ckpt), test_mode=True).matcher
#         self.net.eval().to(self.device)

#     def match(self, g0, g1, scale):
#         torch, F = self.torch, self.F
#         p0, p1 = _prep(g0, scale), _prep(g1, scale)

#         def rgb_like(p):  # (1,3,H,W) in [0,1]
#             return torch.from_numpy(p["img"] / 255.0)[None, None].repeat(1, 3, 1, 1)

#         batch = {
#             "image0": torch.from_numpy(p0["pad"])[None, None] / 255.0,
#             "image1": torch.from_numpy(p1["pad"])[None, None] / 255.0,
#             "image0_rgb_origin": rgb_like(p0),
#             "image1_rgb_origin": rgb_like(p1),
#             "origin_img_size0": torch.tensor(p0["size_res"])[None],
#             "origin_img_size1": torch.tensor(p1["size_res"])[None],
#         }
#         for k, p in (("mask0", p0), ("mask1", p1)):
#             m = F.interpolate(torch.from_numpy(p["mask"])[None, None].float(), scale_factor=0.125, mode="nearest")
#             batch[k] = m[0].bool()  # (1, S/8, S/8)
#         batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

#         with torch.no_grad():
#             self.net(batch)
#         k0 = batch["mkpts0_f"].float().cpu().numpy() * np.array([p0["sx"], p0["sy"]], np.float32)
#         k1 = batch["mkpts1_f"].float().cpu().numpy() * np.array([p1["sx"], p1["sy"]], np.float32)
#         return k0, k1, batch["mconf"].float().cpu().numpy()


# class OrbMatcher:
#     """Debug-only stand-in so the pipeline/plots can be tested without weights."""

#     def __init__(self):
#         self.orb = cv2.ORB_create(4000)
#         self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     def match(self, g0, g1, scale):
#         k0, d0 = self.orb.detectAndCompute(g0, None)
#         k1, d1 = self.orb.detectAndCompute(g1, None)
#         if d0 is None or d1 is None:
#             return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
#         ms = self.bf.match(d0, d1)
#         p0 = np.float32([k0[m.queryIdx].pt for m in ms]).reshape(-1, 2)
#         p1 = np.float32([k1[m.trainIdx].pt for m in ms]).reshape(-1, 2)
#         conf = np.float32([1 - m.distance / 256 for m in ms])
#         return p0, p1, conf


# GDRIVE_WEIGHTS_ID = "12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d"  # weights.zip linked in the MatchAnything README


# def _find_ckpt(repo):
#     return next(Path(repo).rglob("matchanything_eloftr.ckpt"), None)


# def resolve_ckpt(ckpt, repo):
#     """Use ckpt_path if it exists; otherwise look inside the repo folder, then try to download."""
#     if ckpt and Path(ckpt).is_file():
#         return str(ckpt)
#     if ckpt:
#         print(f"[weights] ckpt_path {ckpt!r} not found, searching / downloading instead")
#     found = _find_ckpt(repo)
#     if found:
#         print(f"[weights] using {found}")
#         return str(found)

#     # 1) Hugging Face (the repo the official demo points to; file names not verified)
#     try:
#         from huggingface_hub import hf_hub_download

#         for fn in ("weights/matchanything_eloftr.ckpt", "matchanything_eloftr.ckpt"):
#             try:
#                 p = hf_hub_download("LittleFrog/MatchAnything_checkpoints", fn)
#                 print(f"[weights] downloaded from Hugging Face: {p}")
#                 return p
#             except Exception:
#                 continue
#     except ImportError:
#         pass

#     # 2) Google Drive zip from the README (needs: pip install gdown)
#     try:
#         import zipfile

#         import gdown

#         zip_path = Path(repo) / "weights.zip"
#         print("[weights] downloading weights.zip from Google Drive ...")
#         gdown.download(id=GDRIVE_WEIGHTS_ID, output=str(zip_path), quiet=False)
#         with zipfile.ZipFile(zip_path) as z:
#             z.extractall(repo)
#         zip_path.unlink()
#         found = _find_ckpt(repo)
#         if found:
#             print(f"[weights] using {found}")
#             return str(found)
#     except ImportError:
#         print("[weights] 'gdown' is not installed: run  pip install gdown  and try again")
#     except Exception as e:
#         print(f"[weights] Google Drive download failed: {e}")

#     raise SystemExit(
#         "Could not find or download the checkpoint. Download weights.zip manually from the MatchAnything "
#         "README, unzip it into the repo folder, and set ckpt_path in the CONFIG block."
#     )


# # --------------------------------------------------------------------------------------
# # geometry helpers
# # --------------------------------------------------------------------------------------
# def sample_patch(sar, size, rng, fixed_xy=None, max_tries=300):
#     h, w = sar.shape
#     if fixed_xy is not None:
#         return int(fixed_xy[0]), int(fixed_xy[1])
#     x = y = 0
#     for _ in range(max_tries):
#         x = int(rng.integers(0, w - size + 1))
#         y = int(rng.integers(0, h - size + 1))
#         p = sar[y:y + size, x:x + size]
#         if (p == 0).mean() < 0.2 and p.std() > 8:  # skip nodata / flat areas
#             return x, y
#     print("[warn] could not find a textured patch; using the last random one")
#     return x, y


# def make_windows(hh, ww, win, stride):
#     wh, wd = min(win, hh), min(win, ww)
#     ys = list(range(0, hh - wh + 1, stride))
#     xs = list(range(0, ww - wd + 1, stride))
#     if ys[-1] != hh - wh:
#         ys.append(hh - wh)
#     if xs[-1] != ww - wd:
#         xs.append(ww - wd)
#     return [(x, y, wd, wh) for y in ys for x in xs]


# def estimate(p0, p1, model, thr):
#     if len(p0) < 4:
#         return None, None
#     p0, p1 = p0.astype(np.float32), p1.astype(np.float32)
#     if model == "homography":
#         method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
#         H, m = cv2.findHomography(p0, p1, method, thr, maxIters=10000, confidence=0.9999)
#     else:  # similarity: rotation + uniform scale + translation (suits co-registered maps)
#         M, m = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=thr,
#                                            maxIters=10000, confidence=0.9999)
#         H = None if M is None else np.vstack([M, [0, 0, 1]])
#     if H is None or m is None:
#         return None, None
#     return H, m.ravel().astype(bool)


# def project(H, pts):
#     return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


# # --------------------------------------------------------------------------------------
# # plotting
# # --------------------------------------------------------------------------------------
# def show_map(ax, img, title, max_side=2000):
#     h, w = img.shape[:2]
#     s = min(1.0, max_side / max(h, w))
#     small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
#     ax.imshow(small, cmap="gray" if img.ndim == 2 else None, extent=(0, w, h, 0))
#     ax.set_xlim(0, w)
#     ax.set_ylim(h, 0)
#     ax.set_title(title, fontsize=10)


# def box(ax, corners, color, label, ls="-", lw=2.0, off=(0, 0)):
#     c = np.asarray(corners) + np.asarray(off)
#     c = np.vstack([c, c[:1]])
#     ax.plot(c[:, 0], c[:, 1], ls, color=color, lw=lw, label=label)


# def plot_result(sar, rgb, patch_xy, size, best, gt_corners, title, out_path, show, max_lines=150):
#     import matplotlib.pyplot as plt

#     x0, y0 = patch_xy
#     patch = sar[y0:y0 + size, x0:x0 + size]
#     patch_corners = np.float32([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])

#     fig = plt.figure(figsize=(16, 11), constrained_layout=True)
#     gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
#     ax_s, ax_r, ax_m = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

#     show_map(ax_s, sar, "SAR map: sampled patch (yellow)")
#     box(ax_s, patch_corners, "yellow", "sampled patch")

#     show_map(ax_r, rgb, "RGB map: predicted location (red)" + (", expected (green)" if gt_corners is not None else ""))
#     if best is not None:
#         wx, wy, ww, wh = best["win"]
#         box(ax_r, [[wx, wy], [wx + ww, wy], [wx + ww, wy + wh], [wx, wy + wh]], "cyan", "best search window", "--", 1.2)
#         if best["quad"] is not None:
#             box(ax_r, best["quad"], "red", "prediction")
#     if gt_corners is not None:
#         box(ax_r, gt_corners, "lime", "expected (aligned)", "--", 1.5)
#     ax_r.legend(loc="upper right", fontsize=8)

#     # --- correspondence panel: SAR patch | best RGB window ----------------------------
#     if best is not None:
#         wx, wy, ww, wh = best["win"]
#         crop = rgb[wy:wy + wh, wx:wx + ww]
#         gap = 30
#         hc, wc = max(size, wh), size + gap + ww
#         canvas = np.full((hc, wc, 3), 255, np.uint8)
#         canvas[:size, :size] = np.repeat(patch[..., None], 3, 2)
#         canvas[:wh, size + gap:size + gap + ww] = crop
#         ax_m.imshow(canvas)
#         ax_m.axis("off")
#         k0, k1, conf = best["k0"], best["k1"], best["conf"]
#         inl = np.flatnonzero(best["inliers"])
#         if len(inl) > max_lines:
#             inl = np.random.default_rng(0).choice(inl, max_lines, replace=False)
#         cmap = plt.cm.viridis
#         cn = (conf - conf.min()) / (np.ptp(conf) + 1e-9) if len(conf) else conf
#         for i in inl:
#             a = k0[i] - [x0, y0]  # patch coords
#             b = k1[i] + [size + gap, 0]  # window coords (k1 already window-local)
#             ax_m.plot([a[0], b[0]], [a[1], b[1]], "-", color=cmap(cn[i]), lw=0.7, alpha=0.8)
#             ax_m.plot([a[0], b[0]], [a[1], b[1]], "o", ms=2, color=cmap(cn[i]))
#         off = (size + gap - wx, -wy)
#         if best["quad"] is not None:
#             box(ax_m, best["quad"], "red", "prediction", off=off)
#         if gt_corners is not None:
#             box(ax_m, gt_corners, "lime", "expected", "--", 1.5, off=off)
#         ax_m.set_title(f"Correspondences (inliers drawn, max {max_lines}; colour = confidence)", fontsize=10)
#     else:
#         ax_m.axis("off")
#         ax_m.text(0.5, 0.5, "No transform found (too few matches)", ha="center", va="center", fontsize=14)

#     fig.suptitle(title, fontsize=12)
#     fig.savefig(out_path, dpi=130)
#     print(f"[saved] {out_path}")
#     if show:
#         plt.show()
#     plt.close(fig)


# # --------------------------------------------------------------------------------------
# # main
# # --------------------------------------------------------------------------------------
# def run_trial(i, args, matcher, sar, rgb, rgb_gray, seed):
#     rng = np.random.default_rng(seed + i)
#     size = args.patch_size
#     x0, y0 = sample_patch(sar, size, rng, args.patch_xy)
#     patch = sar[y0:y0 + size, x0:x0 + size]
#     hh, ww = rgb_gray.shape
#     win = args.window or 2 * size
#     wins = make_windows(hh, ww, win, args.stride or max(1, win // 2))
#     print(f"[trial {i}] patch at (x={x0}, y={y0}) size={size}; searching {len(wins)} window(s) of ~{min(win, ww)}x{min(win, hh)}")

#     if args.search_radius and args.aligned:
#         cx, cy = x0 + size / 2, y0 + size / 2
#         wins = [w for w in wins if abs(w[0] + w[2] / 2 - cx) <= args.search_radius and abs(w[1] + w[3] / 2 - cy) <= args.search_radius]
#         print(f"         restricted to {len(wins)} window(s) near ground truth (search_radius)")

#     best = None
#     for wi, (wx, wy, wd, wh) in enumerate(wins):
#         crop = rgb_gray[wy:wy + wh, wx:wx + wd]
#         scale = args.img_resize / max(wd, wh)  # same scale for patch and window
#         k0, k1, conf = matcher.match(patch, crop, scale)
#         if len(k0) < args.min_inliers:
#             continue
#         H, inl = estimate(k0, k1 + [wx, wy], args.model, args.ransac_thr)  # patch coords -> RGB-map coords
#         if H is None:
#             continue
#         # k0 is patch-local: the transform maps patch-local coords to map coords
#         n_in = int(inl.sum())
#         if best is None or n_in > best["n"]:
#             best = dict(n=n_in, win=(wx, wy, wd, wh), H=H, inliers=inl, k0=k0 + [x0, y0], k1=k1, conf=conf, n_matches=len(k0))
#         if (wi + 1) % 20 == 0:
#             print(f"         {wi + 1}/{len(wins)} windows done (best inliers so far: {best['n'] if best else 0})")

#     result = dict(ok=False, center_err=None, inliers=0)
#     gt_corners = None
#     local_corners = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
#     if args.aligned:
#         gt_corners = local_corners + [x0, y0]  # identity after SAR rescale to RGB resolution

#     title = f"patch ({x0},{y0}) {size}px"
#     if best is None or best["n"] < args.min_inliers:
#         print("         -> no reliable match (inliers below min_inliers)")
#         title += " | NO RELIABLE MATCH"
#         best_plot = None if best is None else dict(best, quad=None)
#     else:
#         H = best["H"]
#         quad = project(H, local_corners)
#         best["quad"] = quad
#         best_plot = best
#         s = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
#         rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
#         pred_c = project(H, [[size / 2, size / 2]])[0]
#         result.update(inliers=best["n"])
#         title += f" | inliers {best['n']}/{best['n_matches']} | scale {s:.2f} rot {rot:.1f} deg"
#         msg = f"         -> inliers {best['n']}/{best['n_matches']}, scale {s:.3f}, rot {rot:.2f} deg, centre at ({pred_c[0]:.1f},{pred_c[1]:.1f})"
#         if args.aligned:
#             err = float(np.linalg.norm(pred_c - [x0 + size / 2, y0 + size / 2]))
#             corner_err = float(np.linalg.norm(quad - gt_corners, axis=1).mean())
#             result.update(center_err=err, ok=err <= args.success_px)
#             title += f" | centre err {err:.1f}px (corner {corner_err:.1f}px) {'OK' if result['ok'] else 'FAIL'}"
#             msg += f"; centre error {err:.1f}px, mean corner error {corner_err:.1f}px -> {'OK' if result['ok'] else 'FAIL'}"
#         print(msg)

#     out = Path(args.out)
#     out_path = out if args.trials == 1 else out.with_name(f"{out.stem}_{i:02d}{out.suffix}")
#     plot_result(sar, rgb, (x0, y0), size, best_plot, gt_corners, title, out_path, show=(args.trials == 1 and not args.no_show))
#     return result


# def main():
#     args = SimpleNamespace(
#         sar=sar_path, rgb=rgb_path, matcher=matcher, repo=repo_path, ckpt=ckpt_path, device=device,
#         img_resize=img_resize, match_thr=match_thr, patch_size=patch_size, patch_xy=patch_xy,
#         window=window, stride=stride, sar_scale=sar_scale, aligned=aligned, search_radius=search_radius,
#         model=model, ransac_thr=ransac_thr, min_inliers=min_inliers, success_px=success_px,
#         sar_stretch=sar_stretch, downscale=downscale, trials=trials, seed=seed, out=out_path, no_show=no_show,
#     )

#     for name, p in (("sar_path", args.sar), ("rgb_path", args.rgb)):
#         if not Path(p).is_file():
#             raise SystemExit(f"{name} = {p!r} does not exist. Edit the CONFIG block at the top of this script.")
#     if args.matcher == "matchanything" and not Path(args.repo).is_dir():
#         raise SystemExit(f"repo_path = {args.repo!r} is not a folder. Edit the CONFIG block at the top of this script.")

#     if args.no_show or args.trials > 1:
#         matplotlib.use("Agg")

#     seed_ = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 10**9)
#     print(f"[seed] {seed_} (set seed = {seed_} in the CONFIG block to reproduce)")

#     sar = load_sar(args.sar, args.sar_stretch)
#     rgb = load_rgb(args.rgb)
#     print(f"[load] SAR {sar.shape[1]}x{sar.shape[0]}  RGB {rgb.shape[1]}x{rgb.shape[0]}")
#     if args.downscale and args.downscale > 1:
#         f = 1.0 / args.downscale
#         sar = cv2.resize(sar, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
#         rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
#         print(f"[downscale] both maps / {args.downscale} -> SAR {sar.shape[1]}x{sar.shape[0]}  RGB {rgb.shape[1]}x{rgb.shape[0]}")
#     rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

#     s = args.sar_scale
#     if s is None and args.aligned:
#         s = rgb.shape[1] / sar.shape[1]
#     if s and abs(s - 1) > 1e-3:
#         interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
#         sar = cv2.resize(sar, None, fx=s, fy=s, interpolation=interp)
#         print(f"[scale] SAR rescaled by {s:.4f} -> {sar.shape[1]}x{sar.shape[0]}")
#         if args.aligned and abs(sar.shape[0] - rgb.shape[0]) > 0.02 * rgb.shape[0]:
#             print("[warn] aspect ratios differ; maps probably do not cover the same extent, aligned errors are not meaningful")

#     P = args.patch_size
#     if P > min(sar.shape) or P > min(rgb_gray.shape):
#         raise SystemExit(f"patch_size = {P} is larger than a map (SAR {sar.shape}, RGB {rgb_gray.shape})")

#     if args.matcher == "orb":
#         matcher_obj = OrbMatcher()
#     else:
#         matcher_obj = MatchAnythingELoFTR(args.repo, resolve_ckpt(args.ckpt, args.repo), args.device, args.img_resize, args.match_thr)

#     results = [run_trial(i, args, matcher_obj, sar, rgb, rgb_gray, seed_) for i in range(args.trials)]

#     if args.trials > 1:
#         found = [r for r in results if r["inliers"] >= args.min_inliers]
#         print(f"\n[summary] {len(found)}/{args.trials} patches produced a reliable transform")
#         if args.aligned:
#             ok = sum(r["ok"] for r in results)
#             errs = [r["center_err"] for r in results if r["center_err"] is not None]
#             print(f"[summary] success (centre error <= {args.success_px}px): {ok}/{args.trials}"
#                   + (f", median centre error {np.median(errs):.1f}px" if errs else ""))


# if __name__ == "__main__":
#     main()









# # import rasterio

# # sar_path = r"C:\Users\palak\Downloads\sar_grd_db.tif"
# # rgb_path = r"C:\Users\palak\Downloads\sar_rgb_output\rgb_sar_aligned.tif"

# # for name, path in [("SAR", sar_path), ("RGB", rgb_path)]:

# #     with rasterio.open(path) as src:

# #         print("\n======================")
# #         print(name)
# #         print("======================")

# #         print("CRS:", src.crs)
# #         print("Size:", src.width, "x", src.height)

# #         print("Resolution:", src.res)

# #         print("Bounds:")
# #         print(" ", src.bounds)

# #         print("Transform:")
# #         print(src.transform)

# #         print("Center:")
# #         print(src.bounds.left + (src.bounds.right-src.bounds.left)/2,
# #               src.bounds.bottom + (src.bounds.top-src.bounds.bottom)/2)


# # import rasterio
# # from rasterio.warp import reproject, Resampling
# # import numpy as np

# # sar_path = r"C:\Users\palak\Downloads\sar_rgb_output\sar_on_rgb_grid.tif"
# # rgb_path = r"C:\Users\palak\Downloads\sar_rgb_output\rgb_sar_aligned.tif"

# # out_path = r"C:\Users\palak\Downloads\sar_rgb_output\sar_on_rgb_grid.tif"


# # with rasterio.open(sar_path) as sar, rasterio.open(rgb_path) as rgb:

# #     print("SAR:", sar.width, sar.height)
# #     print("RGB:", rgb.width, rgb.height)

# #     # Output array has EXACTLY the RGB dimensions
# #     sar_resampled = np.empty(
# #         (rgb.height, rgb.width),
# #         dtype=np.float32
# #     )

# #     reproject(
# #         source=rasterio.band(sar, 1),
# #         destination=sar_resampled,

# #         src_transform=sar.transform,
# #         src_crs=sar.crs,

# #         dst_transform=rgb.transform,
# #         dst_crs=rgb.crs,

# #         src_nodata=sar.nodata,
# #         dst_nodata=0,

# #         resampling=Resampling.bilinear
# #     )

# #     profile = rgb.profile.copy()

# #     profile.update(
# #         driver="GTiff",
# #         dtype="float32",
# #         count=1,
# #         nodata=0,
# #         compress="deflate"
# #     )

# #     with rasterio.open(out_path, "w", **profile) as dst:
# #         dst.write(sar_resampled, 1)

# # print("\nSaved:")
# # print(out_path)































#!/usr/bin/env python3
"""
Match.py  (sar_rgb_match_check)

Sanity-check MatchAnything (ELoFTR variant) on ONE SAR map and ONE RGB map.

  1. Loads the SAR map and the RGB map.
  2. Samples a square patch from the SAR map.
  3. Slides overlapping search windows over the RGB map; the patch is matched against
     every window with MatchAnything-ELoFTR; the window with most RANSAC inliers wins.
  4. Estimates patch -> RGB-map transform and plots the result.

Edit the CONFIG block below, then run:  python Match.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# OpenCV refuses huge images by default; lift the limit (must be set before importing cv2)
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import cv2
import matplotlib
import numpy as np
import rasterio
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ======================================================================================
# CONFIG
# ======================================================================================
sar_path = r"C:\Users\palak\Downloads\sar_rgb_output\sar_on_rgb_grid.tif"
rgb_path = r"C:\Users\palak\Downloads\sar_rgb_output\rgb_sar_aligned.tif"

repo_path = r"C:\Users\palak\Documents\Plk\DIAT\Semester 3\DSMAC\Code\MatchAnything_restore"
ckpt_path = None            # None = auto-find inside repo_path
matcher = "matchanything"   # or "orb" (debug only)
device = None               # None = cuda if available, else cpu

out_path = "match_result.png"
no_show = False

patch_size = 2048
patch_xy = None
trials = 1
seed = None
window = 0                  # 0 = 2 x patch_size
stride = 0                  # 0 = window / 2
img_resize = 832
match_thr = 0.1

aligned = False
sar_scale = None
search_radius = 0
sar_stretch = False
downscale = 1

model = "similarity"        # or "homography"
ransac_thr = 3.0
min_inliers = 12
success_px = 10.0
# ======================================================================================


# --------------------------------------------------------------------------------------
# image IO
# --------------------------------------------------------------------------------------
def to_u8(a, stretch=True):
    """Convert any dtype to uint8 with a 2-98% percentile stretch (chunked for huge maps)."""
    if a.dtype == np.uint8 and not stretch:
        return a
    step = max(1, int(np.sqrt(a.size / 4_000_000)))
    sample = a[::step, ::step].astype(np.float32)
    vals = sample[np.isfinite(sample) & (sample != 0)]
    if vals.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(vals, (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    out = np.empty(a.shape, np.uint8)
    for r in range(0, a.shape[0], 2048):
        blk = a[r:r + 2048].astype(np.float32)
        ok = np.isfinite(blk) & (blk != 0)
        o = np.clip((blk - lo) / (hi - lo), 0, 1) * 255.0
        o[~ok] = 0
        out[r:r + 2048] = o.astype(np.uint8)
    return out


def load_sar(path, force_stretch=True):
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)
        nodata = src.nodata

    print("[SAR]")
    print("  shape:", img.shape, " nodata:", nodata)

    valid = np.isfinite(img) & (img != 0)
    if nodata is not None:
        valid &= (img != nodata)

    vals = img[valid]
    if vals.size == 0:
        raise RuntimeError("No valid SAR pixels found!")

    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    print(f"  valid pixels: {vals.size}, stretch: {lo:.3f} -> {hi:.3f}")

    out = np.zeros(img.shape, dtype=np.uint8)
    for r in range(0, img.shape[0], 1024):
        block = img[r:r + 1024]
        mask = valid[r:r + 1024]
        scaled = np.zeros(block.shape, dtype=np.float32)
        scaled[mask] = (block[mask] - np.float32(lo)) / np.float32(hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        out[r:r + 1024][mask] = (scaled[mask] * 255).astype(np.uint8)
    return out


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = to_u8(img)
    return img


# --------------------------------------------------------------------------------------
# matchers
# --------------------------------------------------------------------------------------
def _prep(gray, scale):
    """Resize by `scale`, floor to a multiple of 32, pad to a square (as in the official demo)."""
    h, w = gray.shape
    w1, h1 = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    w2, h2 = max(32, w1 // 32 * 32), max(32, h1 // 32 * 32)
    img = np.asarray(Image.fromarray(gray).resize((w2, h2), Image.LANCZOS), dtype=np.float32)
    s = max(h2, w2)
    pad = np.zeros((s, s), np.float32)
    pad[:h2, :w2] = img
    mask = np.zeros((s, s), bool)
    mask[:h2, :w2] = True
    return dict(pad=pad, mask=mask, img=img, sx=w / w2, sy=h / h2, size_res=(h1, w1))


def _purge_src_modules():
    """Drop any previously imported top-level 'src' package (e.g. from EfficientLoFTR)
    so that 'src' resolves to MatchAnything's own package."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]


class MatchAnythingELoFTR:
    def __init__(self, repo, ckpt, device=None, img_resize=832, thr=0.1):
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        repo = Path(repo).resolve()

        # MatchAnything's own `src` package MUST win over any other `src` (EfficientLoFTR etc.)
        _purge_src_modules()
        sys.path[:] = [p for p in sys.path if "EfficientLoFTR" not in p]
        for p in (str(repo.parent), str(repo)):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, str(repo.parent))
        sys.path.insert(0, str(repo))  # repo first, so `src` = repo/src

        from src.config.default import get_cfg_defaults
        from src.lightning.lightning_loftr import PL_LoFTR

        cfg = get_cfg_defaults()

        # The model config is a python script that edits `cfg`; run it in an explicit namespace
        model_config = repo / "configs" / "models" / "eloftr_model.py"
        ns = {"cfg": cfg, "__file__": str(model_config)}
        exec(compile(model_config.read_text(encoding="utf-8"), str(model_config), "exec"), ns)
        cfg = ns["cfg"]

        if cfg.LOFTR.COARSE.ROPE:
            assert cfg.DATASET.NPE_NAME is not None
        if cfg.DATASET.NPE_NAME == "megadepth":
            cfg.LOFTR.COARSE.NPE = [832, 832, img_resize, img_resize]
        cfg.METHOD = "matchanything_eloftr"
        cfg.LOFTR.MATCH_COARSE.THR = thr

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = PL_LoFTR(cfg, pretrained_ckpt=str(ckpt), test_mode=True).matcher
        self.net.eval().to(self.device)

    def match(self, g0, g1, scale):
        torch, F = self.torch, self.F
        p0, p1 = _prep(g0, scale), _prep(g1, scale)

        def rgb_like(p):
            return torch.from_numpy(p["img"] / 255.0)[None, None].repeat(1, 3, 1, 1)

        batch = {
            "image0": torch.from_numpy(p0["pad"])[None, None] / 255.0,
            "image1": torch.from_numpy(p1["pad"])[None, None] / 255.0,
            "image0_rgb_origin": rgb_like(p0),
            "image1_rgb_origin": rgb_like(p1),
            "origin_img_size0": torch.tensor(p0["size_res"])[None],
            "origin_img_size1": torch.tensor(p1["size_res"])[None],
        }
        for k, p in (("mask0", p0), ("mask1", p1)):
            m = F.interpolate(torch.from_numpy(p["mask"])[None, None].float(), scale_factor=0.125, mode="nearest")
            batch[k] = m[0].bool()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            self.net(batch)
        k0 = batch["mkpts0_f"].float().cpu().numpy() * np.array([p0["sx"], p0["sy"]], np.float32)
        k1 = batch["mkpts1_f"].float().cpu().numpy() * np.array([p1["sx"], p1["sy"]], np.float32)
        return k0, k1, batch["mconf"].float().cpu().numpy()


class OrbMatcher:
    """Debug-only stand-in so the pipeline/plots can be tested without weights."""

    def __init__(self):
        self.orb = cv2.ORB_create(4000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, g0, g1, scale):
        k0, d0 = self.orb.detectAndCompute(g0, None)
        k1, d1 = self.orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        ms = self.bf.match(d0, d1)
        p0 = np.float32([k0[m.queryIdx].pt for m in ms]).reshape(-1, 2)
        p1 = np.float32([k1[m.trainIdx].pt for m in ms]).reshape(-1, 2)
        conf = np.float32([1 - m.distance / 256 for m in ms])
        return p0, p1, conf


def resolve_ckpt(ckpt, repo):
    if ckpt and Path(ckpt).is_file():
        return str(ckpt)
    found = next(Path(repo).rglob("matchanything_eloftr.ckpt"), None)
    if found:
        print(f"[weights] using {found}")
        return str(found)
    raise SystemExit("matchanything_eloftr.ckpt not found. Set ckpt_path in the CONFIG block.")


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def sample_patch(sar, size, rng, fixed_xy=None, max_tries=300):
    h, w = sar.shape
    if fixed_xy is not None:
        return int(fixed_xy[0]), int(fixed_xy[1])
    x = y = 0
    for _ in range(max_tries):
        x = int(rng.integers(0, w - size + 1))
        y = int(rng.integers(0, h - size + 1))
        p = sar[y:y + size, x:x + size]
        if (p == 0).mean() < 0.2 and p.std() > 8:
            return x, y
    print("[warn] could not find a textured patch; using the last random one")
    return x, y


def make_windows(hh, ww, win, stride):
    wh, wd = min(win, hh), min(win, ww)
    ys = list(range(0, hh - wh + 1, stride))
    xs = list(range(0, ww - wd + 1, stride))
    if ys[-1] != hh - wh:
        ys.append(hh - wh)
    if xs[-1] != ww - wd:
        xs.append(ww - wd)
    return [(x, y, wd, wh) for y in ys for x in xs]


def estimate(p0, p1, model, thr):
    if len(p0) < 4:
        return None, None
    p0, p1 = p0.astype(np.float32), p1.astype(np.float32)
    if model == "homography":
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, m = cv2.findHomography(p0, p1, method, thr, maxIters=10000, confidence=0.9999)
    else:
        M, m = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                           maxIters=10000, confidence=0.9999)
        H = None if M is None else np.vstack([M, [0, 0, 1]])
    if H is None or m is None:
        return None, None
    return H, m.ravel().astype(bool)


def project(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def show_map(ax, img, title, max_side=2000):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    ax.imshow(small, cmap="gray" if img.ndim == 2 else None, extent=(0, w, h, 0))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(title, fontsize=10)


def box(ax, corners, color, label, ls="-", lw=2.0, off=(0, 0)):
    c = np.asarray(corners) + np.asarray(off)
    c = np.vstack([c, c[:1]])
    ax.plot(c[:, 0], c[:, 1], ls, color=color, lw=lw, label=label)


def plot_result(sar, rgb, patch_xy, size, best, gt_corners, title, out_path, show, max_lines=150):
    import matplotlib.pyplot as plt

    x0, y0 = patch_xy
    patch = sar[y0:y0 + size, x0:x0 + size]
    patch_corners = np.float32([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_s, ax_r, ax_m = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    show_map(ax_s, sar, "SAR map: sampled patch (yellow)")
    box(ax_s, patch_corners, "yellow", "sampled patch")

    show_map(ax_r, rgb, "RGB map: predicted location (red)" + (", expected (green)" if gt_corners is not None else ""))
    if best is not None:
        wx, wy, ww, wh = best["win"]
        box(ax_r, [[wx, wy], [wx + ww, wy], [wx + ww, wy + wh], [wx, wy + wh]], "cyan", "best search window", "--", 1.2)
        if best.get("quad") is not None:
            box(ax_r, best["quad"], "red", "prediction")
    if gt_corners is not None:
        box(ax_r, gt_corners, "lime", "expected (aligned)", "--", 1.5)
    ax_r.legend(loc="upper right", fontsize=8)

    if best is not None:
        wx, wy, ww, wh = best["win"]
        crop = rgb[wy:wy + wh, wx:wx + ww]
        gap = 30
        hc, wc = max(size, wh), size + gap + ww
        canvas = np.full((hc, wc, 3), 255, np.uint8)
        canvas[:size, :size] = np.repeat(patch[..., None], 3, 2)
        canvas[:wh, size + gap:size + gap + ww] = crop
        ax_m.imshow(canvas)
        ax_m.axis("off")
        k0, k1, conf = best["k0"], best["k1"], best["conf"]
        inl = np.flatnonzero(best["inliers"])
        if len(inl) > max_lines:
            inl = np.random.default_rng(0).choice(inl, max_lines, replace=False)
        cmap = plt.cm.viridis
        cn = (conf - conf.min()) / (np.ptp(conf) + 1e-9) if len(conf) else conf
        for i in inl:
            a = k0[i] - [x0, y0]
            b = k1[i] + [size + gap, 0]
            ax_m.plot([a[0], b[0]], [a[1], b[1]], "-", color=cmap(cn[i]), lw=0.7, alpha=0.8)
            ax_m.plot([a[0], b[0]], [a[1], b[1]], "o", ms=2, color=cmap(cn[i]))
        off = (size + gap - wx, -wy)
        if best.get("quad") is not None:
            box(ax_m, best["quad"], "red", "prediction", off=off)
        if gt_corners is not None:
            box(ax_m, gt_corners, "lime", "expected", "--", 1.5, off=off)
        ax_m.set_title(f"Correspondences (inliers drawn, max {max_lines}; colour = confidence)", fontsize=10)
    else:
        ax_m.axis("off")
        ax_m.text(0.5, 0.5, "No transform found (too few matches)", ha="center", va="center", fontsize=14)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def run_trial(i, args, matcher_obj, sar, rgb, rgb_gray, seed):
    rng = np.random.default_rng(seed + i)
    size = args.patch_size
    x0, y0 = sample_patch(sar, size, rng, args.patch_xy)
    patch = sar[y0:y0 + size, x0:x0 + size]
    hh, ww = rgb_gray.shape
    win = args.window or 2 * size
    wins = make_windows(hh, ww, win, args.stride or max(1, win // 2))
    print(f"[trial {i}] patch at (x={x0}, y={y0}) size={size}; searching {len(wins)} window(s) of ~{min(win, ww)}x{min(win, hh)}")

    if args.search_radius and args.aligned:
        cx, cy = x0 + size / 2, y0 + size / 2
        wins = [w for w in wins if abs(w[0] + w[2] / 2 - cx) <= args.search_radius
                and abs(w[1] + w[3] / 2 - cy) <= args.search_radius]
        print(f"         restricted to {len(wins)} window(s) near ground truth (search_radius)")

    best = None
    for wi, (wx, wy, wd, wh) in enumerate(wins):
        crop = rgb_gray[wy:wy + wh, wx:wx + wd]
        scale = args.img_resize / max(wd, wh)
        k0, k1, conf = matcher_obj.match(patch, crop, scale)
        if len(k0) < args.min_inliers:
            continue
        H, inl = estimate(k0, k1 + [wx, wy], args.model, args.ransac_thr)
        if H is None:
            continue
        n_in = int(inl.sum())
        if best is None or n_in > best["n"]:
            best = dict(n=n_in, win=(wx, wy, wd, wh), H=H, inliers=inl,
                        k0=k0 + [x0, y0], k1=k1, conf=conf, n_matches=len(k0))
        if (wi + 1) % 20 == 0:
            print(f"         {wi + 1}/{len(wins)} windows done (best inliers so far: {best['n'] if best else 0})")

    result = dict(ok=False, center_err=None, inliers=0)
    gt_corners = None
    local_corners = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    if args.aligned:
        gt_corners = local_corners + [x0, y0]

    title = f"patch ({x0},{y0}) {size}px"
    if best is None or best["n"] < args.min_inliers:
        print("         -> no reliable match (inliers below min_inliers)")
        title += " | NO RELIABLE MATCH"
        best_plot = None if best is None else dict(best, quad=None)
    else:
        H = best["H"]
        quad = project(H, local_corners)
        best["quad"] = quad
        best_plot = best
        s = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        pred_c = project(H, [[size / 2, size / 2]])[0]
        result.update(inliers=best["n"])
        title += f" | inliers {best['n']}/{best['n_matches']} | scale {s:.2f} rot {rot:.1f} deg"
        msg = (f"         -> inliers {best['n']}/{best['n_matches']}, scale {s:.3f}, rot {rot:.2f} deg, "
               f"centre at ({pred_c[0]:.1f},{pred_c[1]:.1f})")
        if args.aligned:
            err = float(np.linalg.norm(pred_c - [x0 + size / 2, y0 + size / 2]))
            corner_err = float(np.linalg.norm(quad - gt_corners, axis=1).mean())
            result.update(center_err=err, ok=err <= args.success_px)
            title += f" | centre err {err:.1f}px (corner {corner_err:.1f}px) {'OK' if result['ok'] else 'FAIL'}"
            msg += f"; centre error {err:.1f}px, mean corner error {corner_err:.1f}px -> {'OK' if result['ok'] else 'FAIL'}"
        print(msg)

    out = Path(args.out)
    out_file = out if args.trials == 1 else out.with_name(f"{out.stem}_{i:02d}{out.suffix}")
    plot_result(sar, rgb, (x0, y0), size, best_plot, gt_corners, title, out_file,
                show=(args.trials == 1 and not args.no_show))
    return result


def main():
    args = SimpleNamespace(
        sar=sar_path, rgb=rgb_path, matcher=matcher, repo=repo_path, ckpt=ckpt_path, device=device,
        img_resize=img_resize, match_thr=match_thr, patch_size=patch_size, patch_xy=patch_xy,
        window=window, stride=stride, sar_scale=sar_scale, aligned=aligned, search_radius=search_radius,
        model=model, ransac_thr=ransac_thr, min_inliers=min_inliers, success_px=success_px,
        sar_stretch=sar_stretch, downscale=downscale, trials=trials, seed=seed, out=out_path, no_show=no_show,
    )

    for name, p in (("sar_path", args.sar), ("rgb_path", args.rgb)):
        if not Path(p).is_file():
            raise SystemExit(f"{name} = {p!r} does not exist.")
    if args.matcher == "matchanything" and not Path(args.repo).is_dir():
        raise SystemExit(f"repo_path = {args.repo!r} is not a folder.")

    if args.no_show or args.trials > 1:
        matplotlib.use("Agg")

    seed_ = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 10**9)
    print(f"[seed] {seed_} (set seed = {seed_} in the CONFIG block to reproduce)")

    sar = load_sar(args.sar, args.sar_stretch)
    rgb = load_rgb(args.rgb)
    print(f"[load] SAR {sar.shape[1]}x{sar.shape[0]}  RGB {rgb.shape[1]}x{rgb.shape[0]}")

    if args.downscale and args.downscale > 1:
        f = 1.0 / args.downscale
        sar = cv2.resize(sar, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        print(f"[downscale] both maps / {args.downscale}")
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    s = args.sar_scale
    if s is None and args.aligned:
        s = rgb.shape[1] / sar.shape[1]
    if s and abs(s - 1) > 1e-3:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        sar = cv2.resize(sar, None, fx=s, fy=s, interpolation=interp)
        print(f"[scale] SAR rescaled by {s:.4f} -> {sar.shape[1]}x{sar.shape[0]}")

    P = args.patch_size
    if P > min(sar.shape) or P > min(rgb_gray.shape):
        raise SystemExit(f"patch_size = {P} is larger than a map (SAR {sar.shape}, RGB {rgb_gray.shape})")

    if args.matcher == "orb":
        matcher_obj = OrbMatcher()
    else:
        matcher_obj = MatchAnythingELoFTR(args.repo, resolve_ckpt(args.ckpt, args.repo),
                                          args.device, args.img_resize, args.match_thr)

    results = [run_trial(i, args, matcher_obj, sar, rgb, rgb_gray, seed_) for i in range(args.trials)]

    if args.trials > 1:
        found = [r for r in results if r["inliers"] >= args.min_inliers]
        print(f"\n[summary] {len(found)}/{args.trials} patches produced a reliable transform")
        if args.aligned:
            ok = sum(r["ok"] for r in results)
            errs = [r["center_err"] for r in results if r["center_err"] is not None]
            print(f"[summary] success (centre error <= {args.success_px}px): {ok}/{args.trials}"
                  + (f", median centre error {np.median(errs):.1f}px" if errs else ""))


if __name__ == "__main__":
    main()