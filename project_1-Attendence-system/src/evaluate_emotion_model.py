
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Testing device:", device)

# Load test dataset
data = np.load("dataset/fer2013_processed.npz")

X_test = torch.tensor(data["X_test"]).permute(0, 3, 1, 2)
y_test = torch.tensor(data["y_test"], dtype=torch.long)
emotions = data["emotions"].tolist()

print("Test images:", len(X_test))
print("Emotion classes:", emotions)

# Recreate CNN architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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
        return self.classifier(self.features(x))


# Prepare test DataLoader
test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# Models to compare
model_paths = {
    "Version 1": "models/emotion_cnn.pth",
    "Version 2": "models/emotion_cnn_augmented.pth"
}

results = []

os.makedirs("models", exist_ok=True)

# Evaluate both models
for version, model_path in model_paths.items():

    print("\n" + "=" * 50)
    print("EVALUATING", version)
    print("=" * 50)

    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        continue

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = EmotionCNN(len(emotions)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(actual, predictions)

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Version": version,
        "Accuracy": accuracy * 100,
        "Macro F1": macro_f1
    })

    print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
    print("Test Macro F1:", round(macro_f1, 4))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        actual,
        predictions,
        labels=list(range(len(emotions))),
        target_names=emotions,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        actual,
        predictions,
        labels=list(range(len(emotions)))
    )

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=emotions
    )

    matrix_display.plot(
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title(version + " - Emotion Detection Confusion Matrix")
    plt.tight_layout()

    filename = "models/confusion_matrix_" + version.replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.show()
    plt.close()

# Final comparison
print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

for result in results:
    print(
        f'{result["Version"]}: '
        f'Accuracy = {result["Accuracy"]:.2f}% | '
        f'Macro F1 = {result["Macro F1"]:.4f}'
    )

if len(results) == 2:
    best_model = max(results, key=lambda x: x["Macro F1"])

    print("\nBetter model based on test Macro F1:", best_model["Version"])
    print("Best Test Macro F1:", round(best_model["Macro F1"], 4))
