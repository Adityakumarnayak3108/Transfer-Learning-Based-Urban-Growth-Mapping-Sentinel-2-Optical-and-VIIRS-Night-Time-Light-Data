import torch
from torch.utils.data import DataLoader
from config import *
from dataset import SegDataset, get_val_transform
from predict import load_model
from metrics import iou_score, dice_score

def main():
    test_ds = SegDataset(TEST_IMG_DIR, TEST_MASK_DIR, get_val_transform(IMG_SIZE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = load_model(f"{CHECKPOINT_DIR}/best_model.pth")

    ious, dices = [], []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            ious.append(iou_score(logits, masks).item())
            dices.append(dice_score(logits, masks).item())

    print(f"Test IoU: {sum(ious)/len(ious):.4f}")
    print(f"Test Dice: {sum(dices)/len(dices):.4f}")

if __name__ == "__main__":
    main()