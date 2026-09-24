import torch
import cv2
import numpy as np
from config import *
from model import get_model
from dataset import get_val_transform


def load_model(ckpt_path):
    model = get_model(ENCODER, ENCODER_WEIGHTS, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def predict_tta(model, img_path, threshold=0.5):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transform = get_val_transform(IMG_SIZE)
    base = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    augments = [
        base,
        torch.flip(base, [3]),
        torch.flip(base, [2]),
        torch.rot90(base, 1, [2, 3])
    ]

    preds = []

    with torch.no_grad():
        for i, x in enumerate(augments):
            out = torch.sigmoid(model(x))

            if i == 1:
                out = torch.flip(out, [3])
            elif i == 2:
                out = torch.flip(out, [2])
            elif i == 3:
                out = torch.rot90(out, -1, [2, 3])

            preds.append(out)

    avg_pred = torch.stack(preds).mean(0).squeeze().cpu().numpy()

    mask = (avg_pred > threshold).astype(np.uint8) * 255

    # Spatial smoothing to remove speckle noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


if __name__ == "__main__":
    import os
    from glob import glob

    # Create output folder
    os.makedirs("predictions", exist_ok=True)

    # Load trained model
    model = load_model(f"{CHECKPOINT_DIR}/best_model.pth")

    # Predict masks for all test images
    for img_path in glob("data/test/images/*.png"):
        mask = predict_tta(model, img_path)

        name = os.path.basename(img_path)

        cv2.imwrite(
            os.path.join("predictions", name),
            mask
        )

    print("Finished.")