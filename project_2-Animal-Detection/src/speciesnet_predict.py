
import json
import os

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "speciesnet_animal_tests.json"
)


def extract_speciesnet_predictions(json_path):

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []

    for prediction in data.get("predictions", []):

        filepath = prediction.get("filepath", "")
        filename = os.path.basename(filepath)

        for detection in prediction.get("detections", []):

            processed = detection.get("processed_classification", {})
            raw = detection.get("raw_classifications", {})

            taxonomy = processed.get("class", "")
            parts = taxonomy.split(";")

            # Extract the most specific available label
            species_name = parts[-1].strip() if parts else "unknown"

            if not species_name or species_name == "animal":
                species_name = "unknown / unidentified animal"

            detection_confidence = detection.get("conf", 0.0)
            classification_score = processed.get("score", 0.0)
            bbox = detection.get("bbox", [])

            result = {
                "image": filename,
                "animal": species_name,
                "detection_confidence": detection_confidence,
                "classification_score": classification_score,
                "bounding_box": bbox,
                "classification_source": processed.get("source", "unknown"),
                "top_candidates": raw.get("classes", [])[:3],
                "candidate_scores": raw.get("scores", [])[:3]
            }

            results.append(result)

    return results


if __name__ == "__main__":

    if not os.path.exists(JSON_PATH):
        print("ERROR: SpeciesNet JSON file not found!")
    else:
        predictions = extract_speciesnet_predictions(JSON_PATH)

        print("\n========== SPECIESNET RESULTS ==========\n")

        for index, result in enumerate(predictions, start=1):

            print(f"Animal {index}")
            print("Image:", result["image"])
            print("Species:", result["animal"])
            print(
                "Detection confidence:",
                f'{result["detection_confidence"] * 100:.2f}%'
            )
            print(
                "Classification score:",
                f'{result["classification_score"] * 100:.2f}%'
            )
            print("Bounding box:", result["bounding_box"])
            print("Classification source:", result["classification_source"])
            print("----------------------------------------")

        print("\nTotal animal detections:", len(predictions))
