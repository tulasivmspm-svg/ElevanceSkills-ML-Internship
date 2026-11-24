
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "outputs" /
    "animal_detection_fixed" / "weights" / "best.pt"
)

DATA_YAML = (
    PROJECT_DIR / "dataset" /
    "species detection" / "data.yaml"
)

print("Loading trained animal detection model...")

model = YOLO(str(MODEL_PATH))

print("\nEvaluating on unseen test images...")

results = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    project=str(PROJECT_DIR / "outputs"),
    name="test_evaluation"
)

print("\nTEST EVALUATION COMPLETED!")

print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
