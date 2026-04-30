# ================================================================
# UNET_CONVLSTM_FINAL.py
# U-Net + ConvLSTM baseline (same data pipeline as CNN baseline)
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

    n_classes   = 5
    patch_size  = 256
    batch_size  = 2
    epochs      = 25
    lr          = 1e-4
    max_patches = 20000
    feat_dim    = 256
    base_ch     = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed   = 42

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# ================================================================
# UTILS
# ================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
# DATASET
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

        x_seq = np.stack([
            pack(self.t0),
            pack(self.t1)
        ], axis=0)

        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1.0, neginf=0.0)

        y_out = self.t2[y:y+ps, x:x+ps]
        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_out).long()

# ================================================================
# MODEL
# ================================================================
class ConvLSTM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch * 4, 3, padding=1)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
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


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64, feat_dim=256):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base_ch)          # 64
        self.down1 = Down(base_ch, base_ch * 2)          # 128
        self.down2 = Down(base_ch * 2, base_ch * 4)      # 256
        self.down3 = Down(base_ch * 4, base_ch * 8)      # 512
        self.down4 = Down(base_ch * 8, feat_dim)         # 256 bottleneck

    def forward(self, x):
        x1 = self.inc(x)     # [B, 64, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 128, 128]
        x3 = self.down2(x2)  # [B, 256, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 32, 32]
        x5 = self.down4(x4)  # [B, 256, 16, 16]
        return x1, x2, x3, x4, x5


class UNetConvLSTM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = UNetEncoder(
            in_ch=in_ch,
            base_ch=cfg.base_ch,
            feat_dim=cfg.feat_dim
        )

        self.temporal = ConvLSTM(cfg.feat_dim)

        self.up1 = Up(cfg.feat_dim, cfg.base_ch * 8, cfg.base_ch * 4)  # 256 + 512 -> 256
        self.up2 = Up(cfg.base_ch * 4, cfg.base_ch * 4, cfg.base_ch * 2)  # 256 + 256 -> 128
        self.up3 = Up(cfg.base_ch * 2, cfg.base_ch * 2, cfg.base_ch)      # 128 + 128 -> 64
        self.up4 = Up(cfg.base_ch, cfg.base_ch, cfg.base_ch)               # 64 + 64 -> 64

        self.head = nn.Conv2d(cfg.base_ch, cfg.n_classes, kernel_size=1)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        feats = []
        skips = None

        for t in range(x.size(1)):
            x1, x2, x3, x4, x5 = self.encoder(x[:, t])
            feats.append(x5)
            skips = (x1, x2, x3, x4)  # use skip features from latest time step

        feats = torch.stack(feats, dim=1)   # [B, T, C, H, W]
        h = self.temporal(feats)

        x1, x2, x3, x4 = skips
        y = self.up1(h, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)

        out = self.head(y)

        out = F.interpolate(
            out,
            size=(cfg.patch_size, cfg.patch_size),
            mode="bilinear",
            align_corners=False
        )
        return out

# ================================================================
# METRICS
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

    return acc, np.mean(ious) if len(ious) > 0 else 0.0

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

    t0 = np.nan_to_num(t0)
    t1 = np.nan_to_num(t1)
    t2 = np.nan_to_num(t2)

    mask = rasterize_mask(cfg.malawi_shp, prof)

    drivers = []
    for f in sorted(os.listdir(cfg.drivers_dir)):
        if f.endswith(".tif"):
            drivers.append(
                align(os.path.join(cfg.drivers_dir, f), prof, Resampling.bilinear)
            )

    drivers = np.stack(drivers)

    for i in range(drivers.shape[0]):
        d = drivers[i]
        if d.max() > d.min():
            drivers[i] = (d - d.min()) / (d.max() - d.min())

    drivers = np.nan_to_num(drivers, nan=0.0, posinf=1.0, neginf=0.0)

    in_ch = cfg.n_classes + drivers.shape[0]
    model = UNetConvLSTM(in_ch).to(cfg.device)

    weights = torch.tensor([1.0, 1.5, 2.0, 1.5, 1.5], device=cfg.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds = TemporalDataset(t0, t1, t2, drivers, mask)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    loss_hist, acc_hist, miou_hist = [], [], []

    print("Training U-Net + ConvLSTM...")
    print(f"Input channels: {in_ch}")
    print(f"Device: {cfg.device}")

    for ep in range(cfg.epochs):
        model.train()
        tl = ta = ti = 0

        for x, y in dl:
            x, y = x.to(cfg.device), y.to(cfg.device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            acc, miou = compute_metrics(out, y)

            tl += loss.item()
            ta += acc
            ti += miou

        epoch_loss = tl / len(dl)
        epoch_acc  = ta / len(dl)
        epoch_miou = ti / len(dl)

        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)
        miou_hist.append(epoch_miou)

        print(
            f"Epoch {ep+1:02d} | "
            f"Loss {epoch_loss:.4f} | "
            f"Acc {epoch_acc*100:.2f}% | "
            f"mIoU {epoch_miou:.3f}"
        )

    torch.save(
        model.state_dict(),
        os.path.join(cfg.out_dir, "UNET_CONVLSTM.pth")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label="Loss")
    plt.plot(acc_hist, label="Accuracy")
    plt.plot(miou_hist, label="mIoU")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_curves.png"), dpi=300)
    plt.close()

    with open(os.path.join(cfg.out_dir, "final_metrics.txt"), "w") as f:
        f.write("Backbone: U-Net + ConvLSTM\n")
        f.write(f"Input channels: {in_ch}\n")
        f.write(f"Epochs: {cfg.epochs}\n")
        f.write(f"Final Loss: {loss_hist[-1]:.6f}\n")
        f.write(f"Final Accuracy: {acc_hist[-1]:.6f}\n")
        f.write(f"Final mIoU: {miou_hist[-1]:.6f}\n")

    print("Training complete.")
    print(f"Model saved to: {os.path.join(cfg.out_dir, 'UNET_CONVLSTM.pth')}")

# ================================================================
if __name__ == "__main__":
    main()