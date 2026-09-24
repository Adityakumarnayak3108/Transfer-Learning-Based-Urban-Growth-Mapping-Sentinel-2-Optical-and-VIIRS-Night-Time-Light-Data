import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("data/test/images/test_00000.png"), cv2.COLOR_BGR2RGB)
gt = cv2.imread("data/test/masks/test_00000.png", 0)
pred = cv2.imread("prediction.png", 0)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Satellite Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")
plt.axis("off")

plt.show()