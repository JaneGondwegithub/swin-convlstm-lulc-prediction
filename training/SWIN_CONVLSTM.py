# ================================================================
# SWIN_CONVLSTM_FINAL_COMPLETE.py
# Stable SwinV2 + ConvLSTM baseline (with metrics)
# - keeps your simple training loop
# - masks pixels outside Malawi in both input and loss
# - handles timm Swin feature outputs in either NCHW or NHWC
# ================================================================

import os
import random
import warnings

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
    malawi_shp = r"data/raw/malawi.shp"

    out_dir = r"results/swin_convlstm"

    # Swin settings
    model_name = "swinv2_tiny_window8_256"
    pretrained = False   # safer with custom in_chans
    n_classes = 5
    ignore_index = 255

    # Data / training
    patch_size = 256      # must stay 256 for swinv2_tiny_window8_256
    batch_size = 1        # safer for GPU memory
    epochs = 25
    lr = 3e-5
    weight_decay = 1e-4
    max_patches = 20000

    # Model dims
    stage_dim = 64
    feat_dim = 256

    # Fixed CE weights (same idea as your CNN baseline)
    class_weights = (1.0, 1.5, 2.0, 1.5, 1.5)

    num_workers = 0   # safer on Windows
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)


# ================================================================
# UTILS
# ================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def align(path, ref_prof, resampling):
    with rasterio.open(path) as src:
        dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_prof["transform"],
            dst_crs=ref_prof["crs"],
            dst_nodata=np.nan,
            resampling=resampling
        )
        return dst


def rasterize_mask(shp, prof):
    gdf = gpd.read_file(shp)
    if gdf.empty:
        raise ValueError(f"Shapefile is empty: {shp}")

    raster_crs = prof["crs"]
    if gdf.crs is not None and raster_crs is not None and str(gdf.crs) != str(raster_crs):
        gdf = gdf.to_crs(raster_crs)

    geom = gdf.geometry.unary_union

    mask = features.rasterize(
        [(mapping(geom), 1)],
        out_shape=(prof["height"], prof["width"]),
        transform=prof["transform"],
        fill=0,
        dtype="uint8"
    ).astype(bool)

    return mask


def sanitize_lulc(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr.astype(np.int64)
    arr[arr < 0] = 0
    arr[arr >= cfg.n_classes] = 0
    return arr


def normalize_drivers(drivers):
    drivers = drivers.astype(np.float32).copy()

    for i in range(drivers.shape[0]):
        d = drivers[i]
        finite = np.isfinite(d)

        if not finite.any():
            drivers[i] = np.zeros_like(d, dtype=np.float32)
            continue

        vals = d[finite]
        dmin, dmax = vals.min(), vals.max()

        out = np.zeros_like(d, dtype=np.float32)
        if dmax > dmin:
            out[finite] = (d[finite] - dmin) / (dmax - dmin)

        drivers[i] = out

    drivers = np.nan_to_num(drivers, nan=0.0, posinf=0.0, neginf=0.0)
    return drivers


def one_hot(lbl, num_classes):
    out = np.zeros((num_classes, *lbl.shape), dtype=np.float32)
    for c in range(num_classes):
        out[c] = (lbl == c).astype(np.float32)
    return out


# ================================================================
# DATASET
# ================================================================
class TemporalDataset(Dataset):
    def __init__(self, t0, t1, t2, drivers, mask):
        H, W = t0.shape
        ps = cfg.patch_size

        ys, xs = np.where(mask)

        # Top-left patch origins that still fit fully inside the raster
        valid = (ys + ps <= H) & (xs + ps <= W)
        ys, xs = ys[valid], xs[valid]

        if len(ys) == 0:
            raise ValueError("No valid patch origins found. Check patch_size and mask.")

        n = min(cfg.max_patches, len(ys))
        idx = np.random.choice(len(ys), n, replace=False)

        self.coords = list(zip(ys[idx], xs[idx]))
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.drivers = drivers
        self.mask = mask

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        y, x = self.coords[i]
        ps = cfg.patch_size

        mask_patch = self.mask[y:y + ps, x:x + ps]

        def pack(lbl):
            lbl_patch = lbl[y:y + ps, x:x + ps]
            oh = one_hot(lbl_patch, cfg.n_classes)
            drv = self.drivers[:, y:y + ps, x:x + ps]

            stacked = np.concatenate([oh, drv], axis=0)
            stacked *= mask_patch[None, :, :].astype(np.float32)  # zero outside Malawi
            return stacked

        x_seq = np.stack([
            pack(self.t0),
            pack(self.t1)
        ], axis=0)

        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y_out = self.t2[y:y + ps, x:x + ps].copy().astype(np.int64)
        y_out[~mask_patch] = cfg.ignore_index

        return torch.from_numpy(x_seq), torch.from_numpy(y_out)


# ================================================================
# MODEL
# ================================================================
class ConvLSTM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch * 4, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        h = torch.zeros_like(x[:, 0])
        c = torch.zeros_like(h)

        for t in range(x.size(1)):
            y = self.conv(torch.cat([x[:, t], h], dim=1))
            i, f, o, g = torch.chunk(y, 4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g
            h = o * torch.tanh(c)

        return h


class SwinConvLSTM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=in_ch,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        self.stage_channels = list(self.encoder.feature_info.channels())
        if len(self.stage_channels) != 4:
            raise ValueError(f"Expected 4 Swin stages, got {len(self.stage_channels)}")

        self.lateral = nn.ModuleList([
            nn.Conv2d(ch, cfg.stage_dim, kernel_size=1)
            for ch in self.stage_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.stage_dim * 4, cfg.feat_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.temporal = ConvLSTM(cfg.feat_dim)

        self.head = nn.Sequential(
            nn.Conv2d(cfg.feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, cfg.n_classes, kernel_size=1)
        )

    @staticmethod
    def _to_nchw(feat, expected_c):
        if feat.ndim != 4:
            raise ValueError(f"Unexpected feature shape: {tuple(feat.shape)}")

        # Already NCHW
        if feat.shape[1] == expected_c:
            return feat.contiguous()

        # NHWC -> NCHW
        if feat.shape[-1] == expected_c:
            return feat.permute(0, 3, 1, 2).contiguous()

        raise ValueError(
            f"Could not infer feature layout for shape {tuple(feat.shape)} and channels {expected_c}"
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        seq_feats = []

        for t in range(T):
            feats = self.encoder(x[:, t])

            stage_feats = []
            for i, feat in enumerate(feats):
                feat = self._to_nchw(feat, self.stage_channels[i])
                feat = self.lateral[i](feat)
                stage_feats.append(feat)

            # fuse at the highest-resolution Swin stage size (usually 64x64 for 256 input)
            base_h, base_w = stage_feats[0].shape[-2:]

            resized = []
            for feat in stage_feats:
                if feat.shape[-2:] != (base_h, base_w):
                    feat = F.interpolate(
                        feat,
                        size=(base_h, base_w),
                        mode="bilinear",
                        align_corners=False
                    )
                resized.append(feat)

            fused = self.fuse(torch.cat(resized, dim=1))
            seq_feats.append(fused)

        seq_feats = torch.stack(seq_feats, dim=1)   # [B, T, feat_dim, h, w]
        h = self.temporal(seq_feats)
        out = self.head(h)

        return F.interpolate(
            out,
            size=(cfg.patch_size, cfg.patch_size),
            mode="bilinear",
            align_corners=False
        )


# ================================================================
# METRICS
# ================================================================
def compute_metrics(logits, gt):
    pred = logits.argmax(dim=1)

    valid = (gt != cfg.ignore_index)
    if valid.sum().item() == 0:
        return 0.0, 0.0

    pred = pred[valid]
    gt = gt[valid]

    acc = (pred == gt).float().mean().item()

    ious = []
    for c in range(cfg.n_classes):
        tp = ((pred == c) & (gt == c)).sum().float()
        fp = ((pred == c) & (gt != c)).sum().float()
        fn = ((pred != c) & (gt == c)).sum().float()

        denom = tp + fp + fn
        if denom.item() > 0:
            ious.append((tp / denom).item())

    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return acc, miou


# ================================================================
# TRAINING
# ================================================================
def main():
    set_seed(cfg.seed)

    if cfg.model_name == "swinv2_tiny_window8_256" and cfg.patch_size != 256:
        raise ValueError(
            "patch_size must be 256 for swinv2_tiny_window8_256"
        )

    print("Loading data...")

    t0, prof = read_raster(cfg.lulc_2010)
    t1 = align(cfg.lulc_2015, prof, Resampling.nearest)
    t2 = align(cfg.lulc_2020, prof, Resampling.nearest)

    t0 = sanitize_lulc(t0)
    t1 = sanitize_lulc(t1)
    t2 = sanitize_lulc(t2)

    print("Rasterizing Malawi mask...")
    mask = rasterize_mask(cfg.malawi_shp, prof)

    print("Loading drivers...")
    driver_files = sorted([
        f for f in os.listdir(cfg.drivers_dir)
        if f.lower().endswith(".tif")
    ])

    if len(driver_files) == 0:
        raise ValueError(f"No .tif files found in {cfg.drivers_dir}")

    drivers = []
    for f in driver_files:
        arr = align(os.path.join(cfg.drivers_dir, f), prof, Resampling.bilinear)
        drivers.append(arr)

    drivers = np.stack(drivers, axis=0)
    drivers = normalize_drivers(drivers)

    in_ch = cfg.n_classes + drivers.shape[0]
    model = SwinConvLSTM(in_ch).to(cfg.device)

    weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=cfg.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=cfg.ignore_index)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    ds = TemporalDataset(t0, t1, t2, drivers, mask)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(f"Total sampled patches: {len(ds)}")
    print(f"Drivers            : {drivers.shape[0]}")
    print(f"Input channels     : {in_ch}")
    print(f"Device             : {cfg.device}")
    print("Training Swin + ConvLSTM...")

    loss_hist, acc_hist, miou_hist = [], [], []

    for ep in range(cfg.epochs):
        model.train()

        tl = 0.0
        ta = 0.0
        ti = 0.0

        for x, y in dl:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            acc, miou = compute_metrics(out.detach(), y.detach())

            tl += float(loss.item())
            ta += acc
            ti += miou

        ep_loss = tl / len(dl)
        ep_acc = ta / len(dl)
        ep_miou = ti / len(dl)

        loss_hist.append(ep_loss)
        acc_hist.append(ep_acc)
        miou_hist.append(ep_miou)

        print(
            f"Epoch {ep+1:02d} | "
            f"Loss {ep_loss:.4f} | "
            f"Acc {ep_acc*100:.2f}% | "
            f"mIoU {ep_miou:.4f}"
        )

    # Save model
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": cfg.model_name,
            "n_classes": cfg.n_classes,
            "patch_size": cfg.patch_size,
            "in_ch": in_ch,
            "driver_count": int(drivers.shape[0])
        },
        os.path.join(cfg.out_dir, "SWIN_CONVLSTM.pth")
    )

    # Plot curves
    epochs = np.arange(1, len(loss_hist) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, loss_hist, label="Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, acc_hist, label="Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    axes[2].plot(epochs, miou_hist, label="mIoU")
    axes[2].set_title("mIoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mIoU")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_curves.png"), dpi=300)
    plt.close()

    print("Training complete.")


# ================================================================
if __name__ == "__main__":
    main()