# ================================================================
# SWIN_CNN_FINAL_COMPLETE.py
# Swin Transformer baseline (NO temporal module)
# Temporal info is stacked as channels
# ================================================================

import os, random, warnings
import numpy as np
import rasterio
from rasterio.enums import Resampling
import rasterio.warp
from rasterio import features
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================
class Config:
    lulc_2010 = r"data/processed/lulc_2010.tif"
    lulc_2015 = r"data/processed/lulc_2015.tif"
    lulc_2020 = r"data/processed/lulc_2020.tif"

    drivers_dir = r"data/processed/drivers"
    malawi_shp  = r"data/raw/malawi.shp"

    out_dir = r"results/swin_cnn"

    model_name = "swinv2_tiny_window8_256"
    pretrained = False

    n_classes   = 5
    patch_size  = 256
    batch_size  = 1
    epochs      = 25
    lr          = 3e-5
    max_patches = 20000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed   = 42

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# ================================================================
# UTILS (same as your CNN)
# ================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()

def align(path, ref_prof, resampling):
    with rasterio.open(path) as src:
        dst = np.zeros((ref_prof["height"], ref_prof["width"]), np.float32)
        rasterio.warp.reproject(
            rasterio.band(src, 1), dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_prof["transform"],
            dst_crs=ref_prof["crs"],
            resampling=resampling
        )
        return dst

def rasterize_mask(shp, prof):
    gdf = gpd.read_file(shp)
    geom = gdf.geometry.unary_union
    return features.rasterize(
        [(mapping(geom), 1)],
        out_shape=(prof["height"], prof["width"]),
        transform=prof["transform"],
        fill=0
    ).astype(bool)

def sanitize_lulc(arr):
    arr = arr.astype(np.int64)
    arr[arr < 0] = 0
    arr[arr >= cfg.n_classes] = 0
    return arr

def one_hot(lbl, C):
    out = np.zeros((C, *lbl.shape), np.float32)
    for c in range(C):
        out[c] = (lbl == c)
    return out

# ================================================================
# DATASET (stack time in channels)
# ================================================================
class TemporalDataset(Dataset):
    def __init__(self, t0, t1, t2, drivers, mask):
        H, W = t0.shape
        ps = cfg.patch_size

        ys, xs = np.where(mask)
        valid = (ys + ps < H) & (xs + ps < W)
        ys, xs = ys[valid], xs[valid]

        idx = np.random.choice(len(ys), min(cfg.max_patches, len(ys)), replace=False)
        self.coords = list(zip(ys[idx], xs[idx]))

        self.t0, self.t1, self.t2 = t0, t1, t2
        self.drivers = drivers

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        y, x = self.coords[i]
        ps = cfg.patch_size

        def pack(lbl):
            oh = one_hot(lbl[y:y+ps, x:x+ps], cfg.n_classes)
            drv = self.drivers[:, y:y+ps, x:x+ps]
            return np.concatenate([oh, drv], axis=0)

        # 🔥 key difference: concatenate time along channels
        x_input = np.concatenate([
            pack(self.t0),
            pack(self.t1)
        ], axis=0)

        x_input = np.nan_to_num(x_input)

        y_out = self.t2[y:y+ps, x:x+ps]

        return torch.from_numpy(x_input).float(), torch.from_numpy(y_out).long()

# ================================================================
# MODEL (Swin only)
# ================================================================
class SwinCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=in_ch,
            num_classes=0,
            global_pool=""
        )

        self.head = nn.Conv2d(768, cfg.n_classes, 1)

    def forward(self, x):
        feat = self.encoder.forward_features(x)

        # handle NHWC vs NCHW
        if feat.shape[-1] == 768:
            feat = feat.permute(0, 3, 1, 2)

        out = self.head(feat)

        return F.interpolate(
            out,
            (cfg.patch_size, cfg.patch_size),
            mode="bilinear",
            align_corners=False
        )

# ================================================================
# METRICS (same)
# ================================================================
def compute_metrics(logits, gt):
    pred = logits.argmax(1)
    acc = (pred == gt).float().mean().item()

    ious = []
    for c in range(cfg.n_classes):
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        denom = tp + fp + fn
        if denom > 0:
            ious.append((tp / denom).item())

    return acc, np.mean(ious)

# ================================================================
# TRAINING
# ================================================================
def main():
    set_seed(cfg.seed)
    print("Loading data...")

    t0, prof = read_raster(cfg.lulc_2010)
    t1 = align(cfg.lulc_2015, prof, Resampling.nearest)
    t2 = align(cfg.lulc_2020, prof, Resampling.nearest)

    t0 = sanitize_lulc(t0)
    t1 = sanitize_lulc(t1)
    t2 = sanitize_lulc(t2)

    mask = rasterize_mask(cfg.malawi_shp, prof)

    drivers = []
    for f in sorted(os.listdir(cfg.drivers_dir)):
        if f.endswith(".tif"):
            drivers.append(align(os.path.join(cfg.drivers_dir,f),
                                 prof, Resampling.bilinear))

    drivers = np.stack(drivers)

    for i in range(drivers.shape[0]):
        d = drivers[i]
        if d.max() > d.min():
            drivers[i] = (d - d.min()) / (d.max() - d.min())

    drivers = np.nan_to_num(drivers)

    in_ch = (cfg.n_classes + drivers.shape[0]) * 2

    model = SwinCNN(in_ch).to(cfg.device)

    weights = torch.tensor([1.0,1.5,2.0,1.5,1.5], device=cfg.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds = TemporalDataset(t0, t1, t2, drivers, mask)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    print("Training Swin-CNN...")

    for ep in range(cfg.epochs):
        model.train()
        tl = ta = ti = 0

        for x, y in dl:
            x, y = x.to(cfg.device), y.to(cfg.device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            opt.step()

            acc, miou = compute_metrics(out, y)

            tl += loss.item()
            ta += acc
            ti += miou

        print(f"Epoch {ep+1:02d} | Loss {tl/len(dl):.4f} | Acc {ta/len(dl)*100:.2f}% | mIoU {ti/len(dl):.3f}")

    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir,"SWIN_CNN.pth"))

    print("Training complete.")

# ================================================================
if __name__ == "__main__":
    main()