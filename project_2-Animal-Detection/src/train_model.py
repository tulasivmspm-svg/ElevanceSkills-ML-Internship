

from pathlib import Path
import torch
from ultralytics import YOLO


def main():

    # Project directories
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_DIR / "dataset" / "species detection"
    DATA_YAML = DATASET_DIR / "data.yaml"

    # Check GPU
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        DEVICE = 0
    else:
        print("GPU not available. Using CPU.")
        DEVICE = "cpu"

    # Check dataset
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    # Load YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    print("\nStarting Animal Detection Training...")

    # Train model
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=DEVICE,
        workers=0,
        project=str(PROJECT_DIR / "outputs"),
        name="animal_detection_fixed",
        patience=10,
        save=True,
        plots=True
    )

    print("\nTraining completed successfully!")

    print(
        "Best model:",
        PROJECT_DIR / "outputs" /
        "animal_detection_fixed" / "weights" / "best.pt"
    )

    print(
        "Last model:",
        PROJECT_DIR / "outputs" /
        "animal_detection_fixed" / "weights" / "last.pt"
    )


if __name__ == "__main__":
    main()
