
from pathlib import Path
import cv2

from hybrid_detection_engine import detect_animals_hybrid

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_DIR / "test_inputs" / "multiple_image" / "mul1.jpg"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "hybrid_mul2_test.jpg"

annotated, carnivore_count, detections = detect_animals_hybrid(
    IMAGE_PATH
)

cv2.imwrite(str(OUTPUT_PATH), annotated)

print("\n========== HYBRID RESULTS ==========")

for index, detection in enumerate(detections, start=1):
    print(f"\nAnimal {index}")
    print("Species:", detection["animal"])
    print("Diet:", detection["diet_category"])
    print("Source:", detection["classification_source"])
    print("Confidence:", detection["confidence"])

print("\nTotal detections:", len(detections))
print("Carnivore count:", carnivore_count)
print("Saved image:", OUTPUT_PATH)
