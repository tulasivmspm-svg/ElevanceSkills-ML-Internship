
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "outputs" /
    "animal_detection_fixed" / "weights" / "best.pt"
)

# Load trained model
model = YOLO(str(MODEL_PATH))

# Select an image from the test dataset
TEST_IMAGES = (
    PROJECT_DIR / "dataset" /
    "species detection" / "test" / "images"
)

images = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    images.extend(TEST_IMAGES.glob(extension))

if not images:
    raise FileNotFoundError("No test images found!")

image_path = images[0]

print("Testing image:", image_path.name)

# Run prediction
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project=str(PROJECT_DIR / "outputs"),
    name="prediction_preview",
    exist_ok=True
)

print("\nDetection completed!")
print("Number of detected animals:", len(results[0].boxes))
print("Preview saved in outputs/prediction_preview/")
