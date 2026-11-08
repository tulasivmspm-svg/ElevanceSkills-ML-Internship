
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

train_path = "dataset/train"

emotions = sorted(os.listdir(train_path))

plt.figure(figsize=(12, 8))

for i, emotion in enumerate(emotions):
    folder = os.path.join(train_path, emotion)
    image_name = random.choice(os.listdir(folder))
    image_path = os.path.join(folder, image_name)

    image = Image.open(image_path)

    if i == 0:
        print("Image dimensions:", image.size)
        print("Image mode:", image.mode)

    plt.subplot(2, 4, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(emotion)
    plt.axis("off")

plt.tight_layout()
plt.show()
