import matplotlib.pyplot as plt
import os

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score

        elif score <= self.best_score:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0


def plot_history(history, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Loss ----------------
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,"loss.png"))
    plt.close()

    # ---------------- IoU ----------------
    plt.figure(figsize=(8,5))
    plt.plot(history["val_iou"], label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,"iou.png"))
    plt.close()

    # ---------------- Dice ----------------
    plt.figure(figsize=(8,5))
    plt.plot(history["val_dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,"dice.png"))
    plt.close()

    # ---------------- Accuracy ----------------
    plt.figure(figsize=(8,5))
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,"accuracy.png"))
    plt.close()