# ================================================================
# RESNET_CONVLSTM_FINAL.py
# ResNet + ConvLSTM baseline (same data pipeline as CNN baseline)
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
import torchvision.models as models

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

    # choose backbone: "resnet18", "resnet34", "resnet50"
    backbone    = "resnet18"
    pretrained  = False

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


class ResNetEncoder(nn.Module):
    def __init__(self, in_ch, feat_dim=256, backbone="resnet18", pretrained=False):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            last_ch = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            last_ch = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            last_ch = 2048
        else:
            raise ValueError("Unsupported backbone. Choose from: resnet18, resnet34, resnet50")

        # replace first conv to accept your custom number of input channels
        net.conv1 = nn.Conv2d(
            in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.proj = nn.Conv2d(last_ch, feat_dim, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)    # /4
        x = self.layer1(x)  # /4
        x = self.layer2(x)  # /8
        x = self.layer3(x)  # /16
        x = self.layer4(x)  # /32
        x = self.proj(x)    # -> feat_dim
        return x


class ResNetConvLSTM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = ResNetEncoder(
            in_ch=in_ch,
            feat_dim=cfg.feat_dim,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained
        )
        self.temporal = ConvLSTM(cfg.feat_dim)
        self.head = nn.Conv2d(cfg.feat_dim, cfg.n_classes, kernel_size=1)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        feats = []

        for t in range(x.size(1)):
            ft = self.encoder(x[:, t])   # [B, feat_dim, H/32, W/32]
            feats.append(ft)

        feats = torch.stack(feats, dim=1)  # [B, T, feat_dim, h, w]
        h = self.temporal(feats)
        out = self.head(h)

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
    model = ResNetConvLSTM(in_ch).to(cfg.device)

    weights = torch.tensor([1.0, 1.5, 2.0, 1.5, 1.5], device=cfg.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds = TemporalDataset(t0, t1, t2, drivers, mask)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    loss_hist, acc_hist, miou_hist = [], [], []

    print(f"Training {cfg.backbone} + ConvLSTM...")
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

    model_name = f"{cfg.backbone.upper()}_CONVLSTM.pth"
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, model_name))

    # save curves
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

    # save metrics to txt
    with open(os.path.join(cfg.out_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Backbone: {cfg.backbone}\n")
        f.write(f"Input channels: {in_ch}\n")
        f.write(f"Epochs: {cfg.epochs}\n")
        f.write(f"Final Loss: {loss_hist[-1]:.6f}\n")
        f.write(f"Final Accuracy: {acc_hist[-1]:.6f}\n")
        f.write(f"Final mIoU: {miou_hist[-1]:.6f}\n")

    print("Training complete.")
    print(f"Model saved to: {os.path.join(cfg.out_dir, model_name)}")

# ================================================================
if __name__ == "__main__":
    main()