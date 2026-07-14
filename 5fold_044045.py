# Optimized 5fold U-Net training script with Mac (MPS) and Linux (CUDA) support
# ---
import os
# Force PyTorch to only see and use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import gc
import csv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------- PyTorch Optimization Settings --------------------
torch.backends.cudnn.benchmark = True

# -------------------- Config & Path Detection --------------------
# Check if local workspace dataset path exists, else fallback to server paths
local_base = Path("./_4reatedGroup")
if local_base.exists() and (local_base / "Train").exists():
    print(f"👉 Local dataset directory found: '{local_base}'. Running local configuration.")
    BASE = str(local_base.resolve())
    GROUPS = [""]  # Flat local dataset layout (no groups)
    FOLDS = [""]   # Flat local dataset layout (no folds)
    SAVE = "./BestUnetModel"
else:
    print("👉 Local dataset not found. Using Server paths configuration.")
    BASE = r"/home/dl-box/users/students/phasathorn-jewrasumnuay/ImpPic/040SplitGroup"
    SAVE = r"/home/dl-box/users/students/phasathorn-jewrasumnuay/ImpProj-program/BestUnetModel"
    GROUPS = ["044HATimgNetpEnhAugGroup",
              "045HATganSharpEnhAugGroup"]
    FOLDS = [f"Fold{i}" for i in range(1, 6)]

os.makedirs(SAVE, exist_ok=True)

# -------------------- Device Detection --------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Selected Device: {DEVICE}")

# Hyperparameters
LR = 1.25e-4   # 0.000125
BS = 4         # Increased from 2 for better GPU/MPS utilization
EPOCHS = 100
PATIENCE = 10
THR = 0.5

# DataLoader optimizations
NUM_WORKERS = 48 if DEVICE == "cuda" else 0
PIN_MEMORY = True if DEVICE == "cuda" else False
USE_AMP = True if DEVICE == "cuda" else False # Mixed precision only for CUDA

# -------------------- Dataset with Memory Caching --------------------
class ImplantDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, tfm=None, cache=False):
        self.imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            self.imgs.extend(Path(img_dir).glob(ext))
        self.imgs = sorted(self.imgs)
        self.tfm = tfm
        self.cache = cache
        self.cached_data = {}

        # สร้าง Dictionary เก็บตำแหน่ง Mask ทั้งหมดไว้ล่วงหน้า
        self.mask_dict = {}
        if Path(mask_dir).exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
                for mask_path in Path(mask_dir).glob(ext):
                    # เก็บคู่ key เป็น stem ของไฟล์ (เช่น "image1") 
                    # โดยตัดคำว่า "_mask" ออกหากชื่อไฟล์ลงท้ายด้วย _mask
                    clean_stem = mask_path.stem.replace("_mask", "")
                    self.mask_dict[clean_stem] = mask_path

        if len(self.imgs) == 0:
            print(f"⚠️ Warning: No images found in {img_dir}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        if self.cache and i in self.cached_data:
            img, mask = self.cached_data[i]
        else:
            img_path = self.imgs[i]
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ค้นหาใน Dictionary แทนการเช็คไฟล์จริงบน Disk
            mask_path = self.mask_dict.get(img_path.stem)
            if mask_path is None:
                raise FileNotFoundError(f"Could not find mask for image: {img_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load mask: {mask_path}")
            mask = (mask > 0).astype(np.float32)

            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.float32)

            if self.cache:
                self.cached_data[i] = (img, mask)

        if self.tfm:
            t = self.tfm(image=img, mask=mask)
            img, mask = t["image"], t["mask"]
        return img, mask


# -------------------- Model --------------------
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(a, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, feats=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downs, ch = nn.ModuleList(), in_ch
        for f in feats:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.btn = DoubleConv(feats[-1], feats[-1]*2)
        self.ups = nn.ModuleList()
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.out = nn.Conv2d(feats[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.btn(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i//2+1)]
            if x.shape[2:] != s.shape[2:]:
                x = TF.resize(x, s.shape[2:])
            x = self.ups[i+1](torch.cat((s, x), dim=1))
        return self.out(x)

# -------------------- Loss --------------------
class DiceLoss(nn.Module):
    def forward(self, p, t):
        p, t = torch.sigmoid(p).view(-1), t.view(-1)
        return 1 - (2 * (p * t).sum() + 1e-5) / (p.sum() + t.sum() + 1e-5)

# -------------------- Metrics --------------------
def check_accuracy(loader, model, device, thr=THR):
    model.eval()
    loss_sum, pre, sen, f1s, mious, dices, n = 0, 0, 0, 0, 0, 0, 0
    eps = 1e-8
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 3:
                y = y.unsqueeze(1)
            gt = (y > 0.5).float()
            
            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    lg = model(x)
            else:
                lg = model(x)
                
            loss_sum += nn.BCEWithLogitsLoss()(lg, gt).item()
            p = (torch.sigmoid(lg) > thr).float()
            tp = (p * gt).sum(dim=(1, 2, 3))
            fp = (p * (1 - gt)).sum(dim=(1, 2, 3))
            fn = ((1 - p) * gt).sum(dim=(1, 2, 3))
            tn = ((1 - p) * (1 - gt)).sum(dim=(1, 2, 3))
            
            pr = (tp + eps) / (tp + fp + eps)
            se = (tp + eps) / (tp + fn + eps)
            f1 = 2 * pr * se / (pr + se + eps)
            di = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            miou = ((tp + eps) / (tp + fp + fn + eps) + (tn + eps) / (tn + fp + fn + eps)) / 2
            
            pre += pr.mean().item()
            sen += se.mean().item()
            f1s += f1.mean().item()
            mious += miou.mean().item()
            dices += di.mean().item()
            n += 1
            
    for k, v in [("Threshold", thr), ("Loss", loss_sum / max(n, 1)),
                ("Precision", pre / max(n, 1)), ("Sensitivity", sen / max(n, 1)),
                ("F1", f1s / max(n, 1)), ("mIoU", mious / max(n, 1)), ("Dice", dices / max(n, 1))]:
        print(f"  {k:12}: {v:.4f}" if k != "Threshold" else f"  {k:12}: {v}")
        
    model.train()
    return dices / max(n, 1)

# -------------------- Transform --------------------
tfm = A.Compose([A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0), ToTensorV2()])

# -------------------- Train Loops --------------------
results = []
scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

for group in GROUPS:
    for fold in FOLDS:
        group_label = group if group else "LocalGroup"
        fold_label = fold if fold else "LocalFold"
        print(f"\n{'='*60}\n  Group: {group_label} | {fold_label}\n{'='*60}")
        
        # Resolve folders based on run-mode
        if group == "" and fold == "":
            train_img_dir = f"{BASE}/Train"
            train_mask_dir = f"{BASE}/Train_mask"
            val_img_dir = f"{BASE}/Val"
            val_mask_dir = f"{BASE}/Val_mask"
        else:
            train_img_dir = f"{BASE}/{group}/{fold}/Train"
            train_mask_dir = f"{BASE}/{group}/{fold}/Train_mask"
            val_img_dir = f"{BASE}/{group}/{fold}/Val"
            val_mask_dir = f"{BASE}/{group}/{fold}/Val_mask"
            
        train_loader = DataLoader(
            ImplantDataset(train_img_dir, train_mask_dir, tfm, cache=False),
            batch_size=BS, 
            shuffle=True, 
            num_workers=NUM_WORKERS,              
            pin_memory=PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            ImplantDataset(val_img_dir, val_mask_dir, tfm, cache=False), # แนะนำให้ปิด cache ถ้า VRAM/RAM ตึง
            batch_size=BS, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=2
        )
        print(f"  Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
        
        if len(train_loader.dataset) == 0:
            print("  ⚠️ Skip training due to empty dataset.")
            continue

        model = UNet().to(DEVICE)
        
        # Use DataParallel only for CUDA if multiple devices are available
        if DEVICE == "cuda" and torch.cuda.device_count() > 1:
            print(f"  🔧 Using {torch.cuda.device_count()} GPUs: DataParallel enabled")
            model = nn.DataParallel(model)

        # Compile model to optimize CUDA graphs for RTX 6000 Ada
        if DEVICE == "cuda":
            model = torch.compile(model)

        optimizer = optim.Adam(model.parameters(), lr=LR, fused=True if DEVICE == "cuda" else False)
        bce, dice = nn.BCEWithLogitsLoss(), DiceLoss()
        best, no_imp = -1.0, 0
        ckpt = f"{SAVE}/best_{group_label}_{fold_label}.pth.tar"

        import time
        for epoch in range(EPOCHS):
            print(f"\n  ----- Epoch {epoch+1}/{EPOCHS} -----")
            model.train()
            total = 0
            batch_start = time.time()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE, non_blocking=True), y.float().unsqueeze(1).to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True) # Optimized zero_grad
                
                # Forward pass with mixed precision if enabled
                if USE_AMP:
                    with torch.amp.autocast("cuda"):
                        pred = model(x)
                        loss = 0.5 * bce(pred, y) + 0.5 * dice(pred, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x)
                    loss = 0.5 * bce(pred, y) + 0.5 * dice(pred, y)
                    loss.backward()
                    optimizer.step()
                    
                total += loss.item()
                if i % 10 == 0:
                    elasped_time = time.time() - batch_start
                    print(f"    Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
                    print(f"    Time used : {elasped_time:.4f} seconds")
                    batch_start = time.time()

            print(f"  Avg Train Loss: {total/len(train_loader):.4f}\n  [Val Metrics]")
            val_dice = check_accuracy(val_loader, model, DEVICE)

            if val_dice > best + 1e-6:
                best, no_imp = val_dice, 0
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({"state_dict": state_dict}, ckpt)
                print(f"  ✅ Saved BEST → {ckpt} (Dice={best:.4f})")
            else:
                no_imp += 1
                print(f"  (no improve: {no_imp}/{PATIENCE})")
                if no_imp >= PATIENCE:
                    print("  ⏹ Early stopping.")
                    break

        print(f"\n  {group_label} | {fold_label} done! Best Dice={best:.4f}")
        results.append({"group": group_label, "fold": fold_label, "best_dice": best})

        del model, optimizer, train_loader, val_loader
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# -------------------- Summary --------------------
print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
summary = {}
for r in results:
    summary.setdefault(r["group"], []).append(r["best_dice"])

for group, dices in summary.items():
    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    print(f"  {group:35} → Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}  (folds: {[f'{d:.4f}' for d in dices]})")

csv_path = f"{SAVE}/cv_results_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["group", "fold", "best_dice"])
    for r in results:
        writer.writerow([r["group"], r["fold"], r["best_dice"]])
print(f"\n📄 Saved results → {csv_path}")
print(f"\n{'='*60}\nAll done! Models → {SAVE}\n{'='*60}")
