
import json
import os
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    PROJECT_DIR, "outputs", "multiple_animals.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "outputs", "multiple_animal_previews"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DIET_CATEGORY = {
    "lion": "carnivore",
    "leopard": "carnivore",
    "giraffe": "herbivore",
    "plains zebra": "herbivore",
    "african elephant": "herbivore",
}

COLORS = {
    "carnivore": (0, 0, 255),
    "herbivore": (0, 255, 0),
    "omnivore": (0, 255, 255),
    "unknown": (0, 255, 255)
}


def get_species(detection):
    processed = detection.get("processed_classification", {})
    taxonomy = processed.get("class", "")
    parts = taxonomy.split(";")

    species = parts[-1].strip() if parts else ""

    if not species or species in ("animal", "mammal"):
        return "unknown animal"

    return species


def get_diet(species):
    return DIET_CATEGORY.get(species.lower(), "unknown")


def draw_predictions(json_path):

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    grand_total = 0
    grand_carnivores = 0

    for prediction in data.get("predictions", []):

        image_path = prediction.get("filepath", "")
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", image_path)
            continue

        height, width = image.shape[:2]

        counts = {
            "carnivore": 0,
            "herbivore": 0,
            "omnivore": 0,
            "unknown": 0
        }

        detections = prediction.get("detections", [])

        for detection in detections:

            species = get_species(detection)
            diet = get_diet(species)

            confidence = detection.get("conf", 0.0)
            bbox = detection.get("bbox", [])

            counts[diet] += 1

            if len(bbox) != 4:
                continue

            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int((bbox[0] + bbox[2]) * width)
            y2 = int((bbox[1] + bbox[3]) * height)

            color = COLORS[diet]

            label = f"{species} | {diet} | {confidence * 100:.1f}%"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        total = sum(counts.values())
        carnivores = counts["carnivore"]

        grand_total += total
        grand_carnivores += carnivores

        summary = [
            f"Total animals: {total}",
            f"Carnivores: {carnivores}",
            f"Herbivores: {counts['herbivore']}",
            f"Omnivores: {counts['omnivore']}",
            f"Unknown: {counts['unknown']}"
        ]

        for index, line in enumerate(summary):
            cv2.putText(
                image,
                line,
                (15, 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        if carnivores > 0:
            alert = f"ALERT: {carnivores} carnivore(s) detected!"
            print(alert)

            cv2.putText(
                image,
                alert,
                (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if cv2.imwrite(output_path, image):
            print("\nImage:", filename)
            print("Total animals:", total)
            print("Carnivores:", carnivores)
            print("Herbivores:", counts["herbivore"])
            print("Omnivores:", counts["omnivore"])
            print("Unknown:", counts["unknown"])
            print("Saved:", output_path)

    print("\n========== OVERALL RESULTS ==========")
    print("Total animal detections:", grand_total)
    print("Total carnivores:", grand_carnivores)


if __name__ == "__main__":

    if not os.path.exists(JSON_PATH):
        print("SpeciesNet JSON file not found!")
    else:
        draw_predictions(JSON_PATH)
        print("\nVisualization and counting completed!")
