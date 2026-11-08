
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load emotion labels
data = np.load("dataset/fer2013_processed.npz")
emotions = data["emotions"].tolist()

# CNN architecture (Version 1)
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
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load Version 1 model
checkpoint = torch.load(
    "models/emotion_cnn.pth",
    map_location=device,
    weights_only=False
)

model = EmotionCNN(len(emotions)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Select one test image
image_folder = "dataset/test/happy"

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

if not image_files:
    raise FileNotFoundError("No test images found in " + image_folder)

image_path = os.path.join(image_folder, image_files[0])

# Preprocess image
image = Image.open(image_path).convert("L")
image = image.resize((48, 48))

image_array = np.array(image, dtype=np.float32) / 255.0

image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)
image_tensor = image_tensor.to(device)

# Predict emotion
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

predicted_emotion = emotions[predicted_class.item()]

print("\nIMAGE:", image_path)
print("PREDICTED EXPRESSION:", predicted_emotion)
print("MODEL CONFIDENCE:", round(confidence.item() * 100, 2), "%")

