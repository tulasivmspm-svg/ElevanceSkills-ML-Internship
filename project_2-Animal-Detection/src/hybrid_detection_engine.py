
from pathlib import Path

import cv2
from ultralytics import YOLO
from speciesnet import SpeciesNet

from animal_categories import get_diet_category as classify_diet


# ---------------- PATHS AND SETTINGS ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = (
    PROJECT_DIR / "outputs" / "animal_detection_fixed"
    / "weights" / "best.pt"
)

# Official SpeciesNet model identifier.
# The model downloads automatically if it is not cached.
SPECIESNET_MODEL_NAME = (
    "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
)

YOLO_CONFIDENCE = 0.25
SPECIES_CONFIDENCE = 0.65

# Duplicate-removal settings
DUPLICATE_IOU = 0.25
CONTAINMENT_THRESHOLD = 0.70


# ---------------- LOAD MODELS ----------------

_yolo_model = None
_speciesnet_model = None


def load_models():
    global _yolo_model, _speciesnet_model

    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}"
            )

        _yolo_model = YOLO(str(YOLO_MODEL_PATH))

    if _speciesnet_model is None:
        _speciesnet_model = SpeciesNet(
            model_name=SPECIESNET_MODEL_NAME,
            components="all",
            multiprocessing=False
        )

    return _yolo_model, _speciesnet_model


# ---------------- BOX UTILITIES ----------------

def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def calculate_containment(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    smaller_area = min(area_a, area_b)

    return intersection / smaller_area if smaller_area > 0 else 0.0


def species_bbox_to_pixels(bbox, image_width, image_height):
    x, y, width, height = bbox

    x1 = int(x * image_width)
    y1 = int(y * image_height)
    x2 = int((x + width) * image_width)
    y2 = int((y + height) * image_height)

    return (
        max(0, x1),
        max(0, y1),
        min(image_width, x2),
        min(image_height, y2)
    )


# ---------------- SPECIES CLASSIFICATION ----------------

BROAD_LABELS = {
    "animal", "mammal", "mammalia",
    "bird", "aves", "reptile",
    "reptilia", "unknown", "bovidae"
}


def get_reliable_species(processed):
    if not processed:
        return None, 0.0

    taxonomy = processed.get("class", "")
    score = float(processed.get("score", 0.0))

    if processed.get("source") != "classifier":
        return None, score

    parts = taxonomy.split(";")
    species_name = parts[-1].strip().lower() if parts else ""

    if (
        not species_name
        or species_name in BROAD_LABELS
        or species_name.endswith(" species")
        or score < SPECIES_CONFIDENCE
    ):
        return None, score

    return species_name, score


# ---------------- REMOVE DUPLICATES ----------------

def remove_duplicate_detections(detections):
    """
    Keep the strongest detection when two boxes appear
    to represent the same animal.
    """

    detections = sorted(
        detections,
        key=lambda item: (
            item["classification_score"],
            item["confidence"]
        ),
        reverse=True
    )

    kept = []

    for candidate in detections:
        duplicate = False

        for existing in kept:
            iou = calculate_iou(
                candidate["bbox"],
                existing["bbox"]
            )

            containment = calculate_containment(
                candidate["bbox"],
                existing["bbox"]
            )

            if (
                iou >= DUPLICATE_IOU
                or containment >= CONTAINMENT_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ---------------- DIET CATEGORY ----------------

def get_diet_category(animal_name):
    return classify_diet(animal_name)


# ---------------- HYBRID DETECTION ----------------

def detect_animals_hybrid(image_path):

    image_path = str(image_path)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]

    yolo_model, speciesnet_model = load_models()

    # -------- YOLO DETECTION --------

    yolo_results = yolo_model.predict(
        source=image,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    yolo_detections = []

    for result in yolo_results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            name = str(yolo_model.names[class_id]).lower()

            if name == "human":
                continue

            bbox = tuple(map(int, box.xyxy[0].tolist()))

            yolo_detections.append({
                "animal": name,
                "bbox": bbox,
                "confidence": float(box.conf[0]),
                "classification_score": 0.0,
                "source": "YOLOv8"
            })

    # -------- SPECIESNET DETECTION --------

    species_result = speciesnet_model.predict(
        filepaths=[image_path],
        run_mode="single_thread",
        progress_bars=False
    )

    species_detections = []

    predictions = species_result.get("predictions", [])

    if predictions:
        for detection in predictions[0].get("detections", []):

            if detection.get("category") != "1":
                continue

            bbox = species_bbox_to_pixels(
                detection["bbox"],
                image_width,
                image_height
            )

            processed = detection.get(
                "processed_classification", {}
            )

            species_name, class_score = get_reliable_species(
                processed
            )

            species_detections.append({
                "animal": species_name,
                "bbox": bbox,
                "confidence": float(detection.get("conf", 0)),
                "classification_score": class_score,
                "source": "SpeciesNet"
            })

    # -------- COMBINE BOTH MODELS --------

    final_detections = []
    matched_yolo = set()

    for species_detection in species_detections:

        best_index = None
        best_iou = 0.0

        for index, yolo_detection in enumerate(yolo_detections):

            if index in matched_yolo:
                continue

            iou = calculate_iou(
                species_detection["bbox"],
                yolo_detection["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        yolo_name = None
        yolo_confidence = 0.0

        if best_index is not None and best_iou >= DUPLICATE_IOU:
            matched_yolo.add(best_index)
            yolo_name = yolo_detections[best_index]["animal"]
            yolo_confidence = yolo_detections[best_index]["confidence"]

        if species_detection["animal"]:
            final_name = species_detection["animal"]
            source = "SpeciesNet"

        elif yolo_name:
            final_name = yolo_name
            source = "YOLOv8 fallback"

        else:
            final_name = "unidentified animal"
            source = "Unidentified"

        final_detections.append({
            "animal": final_name,
            "bbox": species_detection["bbox"],
            "confidence": species_detection["confidence"],
            "classification_score": species_detection[
                "classification_score"
            ],
            "yolo_confidence": yolo_confidence,
            "classification_source": source
        })

    # Add unmatched YOLO detections
    for index, detection in enumerate(yolo_detections):

        if index in matched_yolo:
            continue

        final_detections.append({
            "animal": detection["animal"],
            "bbox": detection["bbox"],
            "confidence": detection["confidence"],
            "classification_score": 0.0,
            "yolo_confidence": detection["confidence"],
            "classification_source": "YOLOv8 only"
        })

    # Final duplicate removal
    final_detections = remove_duplicate_detections(
        final_detections
    )

    # -------- DRAW RESULTS --------

    annotated_image = image.copy()
    carnivore_count = 0

    for detection in final_detections:

        name = detection["animal"]
        x1, y1, x2, y2 = detection["bbox"]

        diet = get_diet_category(name)

        if diet == "carnivore":
            color = (0, 0, 255)
            carnivore_count += 1

        elif diet == "herbivore":
            color = (0, 255, 0)

        else:
            color = (0, 255, 255)

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        display_confidence = (
            detection["classification_score"]
            if detection["classification_score"] > 0
            else detection["confidence"]
        )

        label = f"{name} | {display_confidence:.2f}"

        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        detection["diet_category"] = diet
        detection["carnivore"] = diet == "carnivore"

    return annotated_image, carnivore_count, final_detections
