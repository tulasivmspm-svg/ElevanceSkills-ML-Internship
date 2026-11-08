
import os
import numpy as np
from PIL import Image

train_path = "dataset/train"
test_path = "dataset/test"

emotions = sorted(os.listdir(train_path))

X_train = []
y_train = []

X_test = []
y_test = []

for label, emotion in enumerate(emotions):

    # Training images
    folder = os.path.join(train_path, emotion)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        image = Image.open(path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        X_train.append(image)
        y_train.append(label)

    # Testing images
    folder = os.path.join(test_path, emotion)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        image = Image.open(path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        X_test.append(image)
        y_test.append(label)

X_train = np.array(X_train)[..., np.newaxis]
X_test = np.array(X_test)[..., np.newaxis]

y_train = np.array(y_train)
y_test = np.array(y_test)

print("PREPROCESSING COMPLETED")
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print("Pixel range:", X_train.min(), "to", X_train.max())

np.savez_compressed(
    "dataset/fer2013_processed.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    emotions=np.array(emotions)
)
