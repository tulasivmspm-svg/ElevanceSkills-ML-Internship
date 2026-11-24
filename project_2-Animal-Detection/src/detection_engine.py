
from pathlib import Path
import cv2
from ultralytics import YOLO
from animal_categories import is_carnivore

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "outputs" /
    "animal_detection_fixed" / "weights" / "best.pt"
)

# Load trained model
model = YOLO(str(MODEL_PATH))


def detect_animals(image, confidence=0.25):
    """
    Detect animals in an image.

    Carnivores: Red bounding boxes
    Other animals: Green bounding boxes

    Returns:
        annotated_image: Image with bounding boxes
        carnivore_count: Number of detected carnivores
        detections: Details of detected animals
    """

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    annotated_image = image.copy()
    carnivore_count = 0
    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            animal_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            carnivore = is_carnivore(animal_name)

            if carnivore:
                color = (0, 0, 255)  # Red
                carnivore_count += 1
            else:
                color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{animal_name} {confidence:.2f}"

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            detections.append({
                "animal": animal_name,
                "confidence": round(confidence, 3),
                "carnivore": carnivore
            })

    return annotated_image, carnivore_count, detections
