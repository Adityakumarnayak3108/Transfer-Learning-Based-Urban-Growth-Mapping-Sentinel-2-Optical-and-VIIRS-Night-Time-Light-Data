import os, shutil, random
from glob import glob

random.seed(42)

def collect_pairs(img_root="data/processed_images", mask_root="data/processed_masks"):
    pairs = []
    for aoi in os.listdir(img_root):
        img_dir = os.path.join(img_root, aoi)
        mask_dir = os.path.join(mask_root, aoi)
        if not os.path.isdir(img_dir):
            continue
        for img_file in glob(os.path.join(img_dir, "*.png")):
            base = os.path.basename(img_file)
            mask_file = os.path.join(mask_dir, base)
            if os.path.exists(mask_file):
                pairs.append((img_file, mask_file))
    return pairs

def split_and_copy(pairs, ratios=(0.7, 0.15, 0.15)):
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }
    for split, items in splits.items():
        img_out = f"data/{split}/images"
        mask_out = f"data/{split}/masks"
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        for i, (img, mask) in enumerate(items):
            fname = os.path.basename(img)
            shutil.copy(img, os.path.join(img_out, fname))
            shutil.copy(mask, os.path.join(mask_out, fname))
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    print("Processed Images Exists:", os.path.exists("data/processed_images"))
    print("Processed Masks Exists :", os.path.exists("data/processed_masks"))
    pairs = collect_pairs()
    split_and_copy(pairs)