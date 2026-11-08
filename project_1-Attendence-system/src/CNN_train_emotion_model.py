
import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

data = np.load("dataset/fer2013_processed.npz")

X = torch.tensor(data["X_train"], dtype=torch.float32)
X = X.permute(0, 3, 1, 2)

y = torch.tensor(data["y_train"], dtype=torch.long)

emotions = data["emotions"].tolist()

print("Training images:", len(X))
print("Emotion classes:", emotions)


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.95, 1.05)
    )
])


# ==========================================
# 4. CUSTOM AUGMENTED DATASET
# ==========================================

class AugmentedDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# 5. TRAIN AND VALIDATION SPLIT
# ==========================================

indices = np.arange(len(y))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=SEED,
    stratify=y.numpy()
)

train_dataset = AugmentedDataset(
    X[train_idx],
    y[train_idx],
    transform=augmentation
)

val_dataset = TensorDataset(
    X[val_idx],
    y[val_idx]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 6. CNN MODEL ARCHITECTURE
# ==========================================

class EmotionCNN(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(

            # Convolution block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


model = EmotionCNN(
    num_classes=len(emotions)
).to(device)

print("\nCNN model created successfully!")


# ==========================================
# 7. HANDLE CLASS IMBALANCE
# ==========================================

counts = torch.bincount(
    y[train_idx],
    minlength=len(emotions)
).float()

class_weights = len(train_idx) / (
    len(emotions) * counts
)

class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# 8. OPTIMIZER AND LEARNING RATE
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==========================================
# 9. MODEL TRAINING SETTINGS
# ==========================================

os.makedirs("models", exist_ok=True)

best_val_f1 = 0.0

epochs = 35
patience = 7

epochs_without_improvement = 0

MODEL_PATH = "models/emotion_cnn_augmented.pth"


# ==========================================
# 10. TRAINING LOOP
# ==========================================

for epoch in range(epochs):

    model.train()

    train_losses = []

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_predictions = []
    val_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                labels.numpy()
            )


    val_acc = accuracy_score(
        val_targets,
        val_predictions
    )

    val_f1 = f1_score(
        val_targets,
        val_predictions,
        average="macro",
        zero_division=0
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {np.mean(train_losses):.4f} | "
        f"Val Accuracy: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save({

            "model_state_dict": model.state_dict(),

            "emotions": emotions,

            "best_val_f1": best_val_f1,

            "epoch": epoch + 1

        }, MODEL_PATH)

        print("Improved model saved!")

    else:

        epochs_without_improvement += 1


    # ======================================
    # EARLY STOPPING
    # ======================================

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered!")

        break


# ==========================================
# 11. TRAINING COMPLETED
# ==========================================

print("\n" + "=" * 45)

print("CNN VERSION 2 TRAINING COMPLETED!")

print("Best Validation Macro F1:", round(best_val_f1, 4))

print("Saved Model:", MODEL_PATH)

print("=" * 45)
