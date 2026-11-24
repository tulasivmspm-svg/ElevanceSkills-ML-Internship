
import json
from collections import Counter

PROJECT_DIR = r"D:\ml internship\ElevanceSkills-ML-Internship\project_2-Animal-Detection"

JSON_PATH = PROJECT_DIR + r"\outputs\video_predictions.json"

with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

species_counts = Counter()
total_detections = 0
total_frames = 0
carnivore_count = 0

CARNIVORES = {
    "lion",
    "leopard",
    "cheetah",
    "tiger",
    "wolf",
    "fox",
    "hyena"
}

for prediction in data.get("predictions", []):

    total_frames += 1
    detections = prediction.get("detections", [])

    for detection in detections:

        processed = detection.get(
            "processed_classification", {}
        )

        taxonomy = processed.get("class", "")
        parts = taxonomy.split(";")
        species = parts[-1].strip() if parts else ""

        if not species or species in ("animal", "mammal"):
            species = "unknown animal"

        total_detections += 1
        species_counts[species] += 1

        if species.lower() in CARNIVORES:
            carnivore_count += 1

print("\n========== VIDEO ANALYSIS ==========")

print("Frames processed:", total_frames)
print("Total animal detections:", total_detections)
print("Total carnivore detections:", carnivore_count)

print("\n========== SPECIES COUNTS ==========")

for species, count in species_counts.most_common():
    print(f"{species}: {count}")

print("\nVideo analysis completed!")
