# ================================================================
# SWIN_CONVLSTM_CNN_FORWARD_2024.py
# Manuscript-aligned forward validation:
# Inputs: 2010, 2015, 2020 + static drivers
# Target: 2024
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
    n_classes = 5
    class_names = ["Water", "Built-up", "Bare land", "Agriculture", "Forest"]

    patch_size = 256
    batch_size = 2
    epochs = 25
    lr = 1e-4
    weight_decay = 1e-4
    max_patches = 20000

    # Swin
    swin_model_name = "swin_tiny_patch4_window7_224"
    pretrained_swin = False

    # Feature dims
    feat_dim = 256
    hidden_dim = 256
    lstm_layers = 2

    num_workers = 0
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# ================================================================
# UTILS
# ================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_raster(path: str):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def align(path: str, ref_prof: dict, resampling: Resampling):
    with rasterio.open(path) as src:
        dst = np.zeros((ref_prof["height"], ref_prof["width"]), np.float32)
        rasterio.warp.reproject(
            rasterio.band(src, 1),
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_prof["transform"],
            dst_crs=ref_prof["crs"],
            resampling=resampling,
        )
        return dst


def rasterize_mask(shp: str, prof: dict):
    gdf = gpd.read_file(shp)
    geom = gdf.geometry.unary_union
    return features.rasterize(
        [(mapping(geom), 1)],
        out_shape=(prof["height"], prof["width"]),
        transform=prof["transform"],
        fill=0,
    ).astype(bool)


def sanitize_lulc(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.int64)
    arr[arr < 0] = 0
    arr[arr >= cfg.n_classes] = 0
    return arr


def one_hot(lbl: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((n_classes, *lbl.shape), np.float32)
    for c in range(n_classes):
        out[c] = (lbl == c)
    return out


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    stack = stack.copy().astype(np.float32)
    for i in range(stack.shape[0]):
        d = np.nan_to_num(stack[i], nan=0.0, posinf=0.0, neginf=0.0)
        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            stack[i] = (d - dmin) / (dmax - dmin)
        else:
            stack[i] = 0.0
    return np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)


# ================================================================
# DATASET
# ================================================================
class Forward2024Dataset(Dataset):
    """
    Input sequence: 2010, 2015, 2020
    Target:         2024
    Each timestep input = one-hot LULC + static drivers
    """
    def __init__(self, t2010, t2015, t2020, t2024, drivers, mask):
        h, w = t2010.shape
        ps = cfg.patch_size

        ys, xs = np.where(mask)
        valid = (ys + ps < h) & (xs + ps < w)
        ys, xs = ys[valid], xs[valid]

        n_samples = min(cfg.max_patches, len(ys))
        idx = np.random.choice(len(ys), n_samples, replace=False)
        self.coords = list(zip(ys[idx], xs[idx]))

        self.t2010 = t2010
        self.t2015 = t2015
        self.t2020 = t2020
        self.t2024 = t2024
        self.drivers = drivers

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        ps = cfg.patch_size

        def pack(lbl):
            oh = one_hot(lbl[y:y+ps, x:x+ps], cfg.n_classes)
            drv = self.drivers[:, y:y+ps, x:x+ps]
            return np.concatenate([oh, drv], axis=0)

        x_seq = np.stack([
            pack(self.t2010),
            pack(self.t2015),
            pack(self.t2020),
        ], axis=0)

        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1.0, neginf=0.0)
        y_out = self.t2024[y:y+ps, x:x+ps]

        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_out).long()


# ================================================================
# MODEL
# ================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size: int, spatial_size: tuple[int, int], device: torch.device):
        h, w = spatial_size
        h_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        c_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        return h_state, c_state


class StackedConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, hidden_dim, H, W]
        """
        batch_size, seq_len, _, h, w = x.shape
        device = x.device

        hidden_states = []
        cell_states = []
        for layer in self.layers:
            h_state, c_state = layer.init_hidden(batch_size, (h, w), device)
            hidden_states.append(h_state)
            cell_states.append(c_state)

        for t in range(seq_len):
            input_t = x[:, t]
            for i, layer in enumerate(self.layers):
                h_state, c_state = layer(input_t, hidden_states[i], cell_states[i])
                hidden_states[i], cell_states[i] = h_state, c_state
                input_t = h_state

        return hidden_states[-1]


class SwinEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.swin_model_name,
            pretrained=cfg.pretrained_swin,
            in_chans=in_ch,
            img_size=cfg.patch_size,
            features_only=True,
            out_indices=(3,),
        )

        self.backbone_out_ch = self.backbone.feature_info.channels()[-1]

        self.proj = nn.Sequential(
            nn.Conv2d(self.backbone_out_ch, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.backbone(x)[0]

        # Swin output may be NHWC
        if feats.dim() == 4 and feats.shape[-1] == self.backbone_out_ch:
            feats = feats.permute(0, 3, 1, 2).contiguous()

        feats = self.proj(feats)
        return feats


class CNNPredictionHead(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, n_classes, kernel_size=1),
        )

    def forward(self, x, out_size):
        x = self.head(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class SwinConvLSTMCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.encoder = SwinEncoder(in_ch=in_ch, feat_dim=cfg.feat_dim)
        self.temporal = StackedConvLSTM(
            input_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            kernel_size=3,
        )
        self.decoder = CNNPredictionHead(cfg.hidden_dim, cfg.n_classes)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        _, t, _, h, w = x.shape

        feat_seq = []
        for i in range(t):
            ft = self.encoder(x[:, i])
            feat_seq.append(ft)

        feat_seq = torch.stack(feat_seq, dim=1)
        h_t = self.temporal(feat_seq)
        out = self.decoder(h_t, out_size=(h, w))
        return out


# ================================================================
# METRICS
# ================================================================
def confusion_matrix_np(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, g in zip(pred.ravel(), gt.ravel()):
        if 0 <= g < n_classes and 0 <= p < n_classes:
            cm[g, p] += 1
    return cm


def compute_metrics_from_cm(cm: np.ndarray):
    oa = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    class_iou = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        class_iou.append(iou)

    miou = float(np.mean(class_iou))
    return oa, miou, class_iou


def plot_normalized_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Reference class",
        xlabel="Predicted class",
        title="Normalized Confusion Matrix (%)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = 100.0 * cm_norm[i, j] if row_sums[i] > 0 else 0.0
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ================================================================
# TRAIN / EVAL
# ================================================================
def main():
    set_seed(cfg.seed)
    print(f"Using device: {cfg.device}")
    print("Loading data...")

    t2010, prof = read_raster(cfg.lulc_2010)
    t2015 = align(cfg.lulc_2015, prof, Resampling.nearest)
    t2020 = align(cfg.lulc_2020, prof, Resampling.nearest)
    t2024 = align(cfg.lulc_2024, prof, Resampling.nearest)

    t2010 = sanitize_lulc(np.nan_to_num(t2010))
    t2015 = sanitize_lulc(np.nan_to_num(t2015))
    t2020 = sanitize_lulc(np.nan_to_num(t2020))
    t2024 = sanitize_lulc(np.nan_to_num(t2024))

    mask = rasterize_mask(cfg.malawi_shp, prof)

    # drivers
    driver_list = []
    for f in sorted(os.listdir(cfg.drivers_dir)):
        if f.lower().endswith(".tif"):
            driver_list.append(align(os.path.join(cfg.drivers_dir, f), prof, Resampling.bilinear))

    if len(driver_list) == 0:
        raise RuntimeError("No driver .tif files found in drivers_dir.")

    drivers = normalize_stack(np.stack(driver_list, axis=0))

    # dataset
    dataset = Forward2024Dataset(t2010, t2015, t2020, t2024, drivers, mask)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # model
    in_ch = cfg.n_classes + drivers.shape[0]
    model = SwinConvLSTMCNN(in_ch=in_ch).to(cfg.device)

    # loss / optimizer
    weights = torch.tensor([1.0, 1.5, 2.0, 1.5, 1.5], device=cfg.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loss_hist, acc_hist, miou_hist = [], [], []

    print("Training Swin-ConvLSTM-CNN for forward validation toward 2024...")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_miou = 0.0

        for x, y in loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = out.argmax(1).detach().cpu().numpy()
            gt = y.detach().cpu().numpy()

            cm = confusion_matrix_np(pred, gt, cfg.n_classes)
            acc, miou, _ = compute_metrics_from_cm(cm)

            total_loss += loss.item()
            total_acc += acc
            total_miou += miou

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_acc / len(loader)
        epoch_miou = total_miou / len(loader)

        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)
        miou_hist.append(epoch_miou)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss {epoch_loss:.4f} | "
            f"OA {epoch_acc*100:.2f}% | "
            f"mIoU {epoch_miou:.3f}"
        )

    # ============================================================
    # Final evaluation on same 2024 forward-validation set
    # ============================================================
    model.eval()
    cm_total = np.zeros((cfg.n_classes, cfg.n_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device, non_blocking=True)
            out = model(x)
            pred = out.argmax(1).cpu().numpy()
            gt = y.numpy()
            cm_total += confusion_matrix_np(pred, gt, cfg.n_classes)

    oa, miou, class_iou = compute_metrics_from_cm(cm_total)

    print("\nFinal Forward Validation Results (Target: 2024)")
    print(f"OA   : {oa*100:.2f}%")
    print(f"mIoU : {miou:.3f}")
    for name, iou in zip(cfg.class_names, class_iou):
        print(f"{name:12s}: {iou:.3f}")

    # ============================================================
    # Save outputs
    # ============================================================
    model_path = os.path.join(cfg.out_dir, "SWIN_CONVLSTM_CNN_FORWARD_2024.pth")
    torch.save(model.state_dict(), model_path)

    # training curves
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label="Loss")
    plt.plot(acc_hist, label="Accuracy")
    plt.plot(miou_hist, label="mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_curves.png"), dpi=300)
    plt.close()

    # confusion matrix
    plot_normalized_confusion_matrix(
        cm_total,
        cfg.class_names,
        os.path.join(cfg.out_dir, "confusion_matrix_normalized.png")
    )

    # metrics text
    metrics_path = os.path.join(cfg.out_dir, "metrics_2024.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Forward Validation Toward 2024\n")
        f.write(f"Overall Accuracy: {oa*100:.2f}%\n")
        f.write(f"Mean IoU: {miou:.3f}\n")
        f.write("\nClass-wise IoU:\n")
        for name, iou in zip(cfg.class_names, class_iou):
            f.write(f"{name}: {iou:.3f}\n")

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {os.path.join(cfg.out_dir, 'confusion_matrix_normalized.png')}")
    print(f"Saved curves to: {os.path.join(cfg.out_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()