#5fold U-Net (Multi-GPU: GPU 0 + GPU 2, เลี่ยง GPU 1 ที่มี ollama)
#---
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import cv2, gc, csv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------- Config --------------------
BASE   = r"/home/dl-box/users/students/phasathorn-jewrasumnuay/ImpPic/040SplitGroup"
SAVE   = r"/home/dl-box/users/students/phasathorn-jewrasumnuay/ImpProj-program/BestUnetModel"
GROUPS = ["041HATsEnhAugGroup", "042HATganEnhAugGroup",
          "043HAT-l-imgNetpEnhAugGroup", "044HATimgNetpEnhAugGroup",
          "045HATganSharpEnhAugGroup"]
FOLDS  = [f"Fold{i}" for i in range(1, 6)]
DEVICE = "cuda"
LR, BS, EPOCHS, PATIENCE, THR = 1.25e-4, 2, 100, 10, 0.5
os.makedirs(SAVE, exist_ok=True)

# -------------------- Dataset --------------------
class ImplantDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, tfm=None):
        self.imgs  = sorted(Path(img_dir).glob("*.png"))
        self.masks = Path(mask_dir)
        self.tfm   = tfm
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(self.masks/f"{self.imgs[i].stem}_mask.png"),
                cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)

        # -------------------- แก้ใหม่: ขยาย mask ให้เท่าภาพ (ไม่ลดภาพลง) --------------------
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.float32)
        # ------------------------------------------------------------------------------------

        if self.tfm:
            t = self.tfm(image=img, mask=mask)
            img, mask = t["image"], t["mask"]
        return img, mask

# -------------------- Model --------------------
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(a,b,3,1,1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b,b,3,1,1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, feats=(64,128,256,512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.downs, ch = nn.ModuleList(), in_ch
        for f in feats: self.downs.append(DoubleConv(ch,f)); ch=f
        self.btn = DoubleConv(feats[-1], feats[-1]*2)
        self.ups = nn.ModuleList()
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f*2,f,2,2))
            self.ups.append(DoubleConv(f*2,f))
        self.out = nn.Conv2d(feats[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for d in self.downs: x=d(x); skips.append(x); x=self.pool(x)
        x = self.btn(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i//2+1)]
            if x.shape[2:] != s.shape[2:]: x = TF.resize(x, s.shape[2:])
            x = self.ups[i+1](torch.cat((s,x), dim=1))
        return self.out(x)

# -------------------- Loss --------------------
class DiceLoss(nn.Module):
    def forward(self, p, t):
        p,t = torch.sigmoid(p).view(-1), t.view(-1)
        return 1-(2*(p*t).sum()+1e-5)/(p.sum()+t.sum()+1e-5)

# -------------------- Metrics --------------------
def check_accuracy(loader, model, device, thr=THR):
    model.eval()
    loss_sum,pre,sen,f1s,mious,dices,n = 0,0,0,0,0,0,0
    eps = 1e-8
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            if y.dim()==3: y=y.unsqueeze(1)
            gt = (y>0.5).float()
            lg = model(x)
            loss_sum += nn.BCEWithLogitsLoss()(lg,gt).item()
            p  = (torch.sigmoid(lg)>thr).float()
            tp = (p*gt).sum(dim=(1,2,3));  fp=(p*(1-gt)).sum(dim=(1,2,3))
            fn = ((1-p)*gt).sum(dim=(1,2,3)); tn=((1-p)*(1-gt)).sum(dim=(1,2,3))
            pr = (tp+eps)/(tp+fp+eps);     se=(tp+eps)/(tp+fn+eps)
            f1 = 2*pr*se/(pr+se+eps);      di=(2*tp+eps)/(2*tp+fp+fn+eps)
            miou = ((tp+eps)/(tp+fp+fn+eps) + (tn+eps)/(tn+fp+fn+eps)) / 2
            pre+=pr.mean().item(); sen+=se.mean().item(); f1s+=f1.mean().item()
            mious+=miou.mean().item(); dices+=di.mean().item(); n+=1
    for k,v in [("Threshold",thr),("Loss",loss_sum/max(n,1)),
                ("Precision",pre/max(n,1)),("Sensitivity",sen/max(n,1)),
                ("F1",f1s/max(n,1)),("mIoU",mious/max(n,1)),("Dice",dices/max(n,1))]:
        print(f"  {k:12}: {v:.4f}" if k!="Threshold" else f"  {k:12}: {v}")
    model.train()
    return dices/max(n,1)

# -------------------- Transform --------------------
tfm = A.Compose([A.Normalize(mean=[0,0,0],std=[1,1,1],max_pixel_value=255.0), ToTensorV2()])

# -------------------- Train 5 Groups x 5 Folds --------------------
results = []

print(f"  Visible GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"    [{i}] {torch.cuda.get_device_name(i)}")

for group in GROUPS:
    for fold in FOLDS:
        print(f"\n{'='*60}\n  Group: {group} | {fold}\n{'='*60}")
        base = f"{BASE}/{group}/{fold}"
        train_loader = DataLoader(ImplantDataset(f"{base}/Train", f"{base}/Train_mask", tfm),
                                  batch_size=BS, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(ImplantDataset(f"{base}/Val",   f"{base}/Val_mask",   tfm),
                                  batch_size=BS, shuffle=False, num_workers=0)
        print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

        model = UNet().to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f"  🔧 Using {torch.cuda.device_count()} GPUs: DataParallel enabled")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        bce, dice = nn.BCEWithLogitsLoss(), DiceLoss()
        best, no_imp = -1.0, 0
        ckpt = f"{SAVE}/best_{group}_{fold}.pth.tar"

        for epoch in range(EPOCHS):
            print(f"\n  ----- Epoch {epoch+1}/{EPOCHS} -----")
            model.train(); total=0
            for i,(x,y) in enumerate(train_loader):
                x,y  = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
                pred = model(x)
                loss = 0.5*bce(pred,y) + 0.5*dice(pred,y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()
                if i%10==0: print(f"    Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

            print(f"  Avg Train Loss: {total/len(train_loader):.4f}\n  [Val Metrics]")
            val_dice = check_accuracy(val_loader, model, DEVICE)

            if val_dice > best+1e-6:
                best, no_imp = val_dice, 0
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({"state_dict": state_dict}, ckpt)
                print(f"  ✅ Saved BEST → {ckpt} (Dice={best:.4f})")
            else:
                no_imp += 1
                print(f"  (no improve: {no_imp}/{PATIENCE})")
                if no_imp >= PATIENCE: print("  ⏹ Early stopping."); break

        print(f"\n  {group} | {fold} done! Best Dice={best:.4f}")
        results.append({"group": group, "fold": fold, "best_dice": best})

        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

# -------------------- สรุปผลลัพธ์ --------------------
print(f"\n{'='*60}\n  SUMMARY (5-Fold CV)\n{'='*60}")
summary = {}
for r in results:
    summary.setdefault(r["group"], []).append(r["best_dice"])

for group, dices in summary.items():
    mean_dice = np.mean(dices)
    std_dice  = np.std(dices)
    print(f"  {group:35} → Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}  (folds: {[f'{d:.4f}' for d in dices]})")

csv_path = f"{SAVE}/cv_results_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["group", "fold", "best_dice"])
    for r in results:
        writer.writerow([r["group"], r["fold"], r["best_dice"]])
print(f"\n📄 Saved results → {csv_path}")

print(f"\n{'='*60}\nAll done! Models → {SAVE}\n{'='*60}")

=======================================================================================