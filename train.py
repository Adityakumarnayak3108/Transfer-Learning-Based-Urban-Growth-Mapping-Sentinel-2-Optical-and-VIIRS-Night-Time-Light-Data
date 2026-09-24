import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import SegDataset, get_train_transform, get_val_transform
from model import get_model
from losses import ComboLoss
from metrics import iou_score, dice_score, pixel_accuracy
from utils import EarlyStopping, plot_history


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = SegDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        get_train_transform(IMG_SIZE)
    )

    val_ds = SegDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        get_val_transform(IMG_SIZE)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = get_model(
        ENCODER,
        ENCODER_WEIGHTS,
        NUM_CLASSES
    ).to(DEVICE)

    # Address class imbalance
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = ComboLoss(
        pos_weight=pos_weight,
        dice_weight=0.5
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    # Training history
    history = {
        "train_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_acc": []
    }

    # Early stopping
    early_stopping = EarlyStopping(patience=10)

    best_iou = 0.0

    for epoch in range(EPOCHS):

        # ---------------- Training ----------------
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [train]"
        ):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(imgs)

            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- Validation ----------------
        model.eval()

        val_iou = 0.0
        val_dice = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS} [val]"
            ):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)

                val_iou += iou_score(logits, masks).item()
                val_dice += dice_score(logits, masks).item()
                val_acc += pixel_accuracy(logits, masks).item()

        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step(val_iou)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_iou={val_iou:.4f} "
            f"val_dice={val_dice:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )

            print(f"  -> Saved new best model (IoU={best_iou:.4f})")

        # ---------------- Early Stopping ----------------
        early_stopping(val_iou)

        if early_stopping.early_stop:
            print("\nEarly stopping triggered.")
            break

    # ---------------- Save Graphs ----------------
    plot_history(history)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()