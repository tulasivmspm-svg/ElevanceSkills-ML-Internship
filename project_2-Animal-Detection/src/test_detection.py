
from pathlib import Path
import cv2
from detection_engine import detect_animals

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_DIR / "dataset" /
    "species detection" / "test" / "images" /
    "banded_civet_104_JPG.rf.2d408be7ff5750bbde4157235a8c941a.jpg"
)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError("Test image could not be loaded!")

annotated_image, carnivore_count, detections = detect_animals(image)

print("\nDetected animals:")

for item in detections:
    print(item)

print("\nCarnivore count:", carnivore_count)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "carnivore_test.jpg"

cv2.imwrite(str(output_path), annotated_image)

print("\nAnnotated image saved:", output_path)
