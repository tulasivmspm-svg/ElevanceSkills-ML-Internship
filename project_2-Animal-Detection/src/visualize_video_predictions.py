
import json
import os
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(
    PROJECT_DIR, "test_inputs", "video_tests", "antelope.mp4"
)

JSON_PATH = os.path.join(
    PROJECT_DIR, "outputs", "video_predictions.json"
)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR, "outputs", "antelope_fast.mp4"
)

CARNIVORES = {
    "lion", "leopard", "cheetah", "tiger",
    "wolf", "fox", "hyena"
}

HERBIVORES = {
    "hartebeest", "giraffe", "plains zebra",
    "african elephant", "impala", "deer",
    "aepyceros species"
}

COLORS = {
    "carnivore": (0, 0, 255),
    "herbivore": (0, 255, 0),
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


def get_category(species):
    species = species.lower()

    if species in CARNIVORES:
        return "carnivore"

    if species in HERBIVORES:
        return "herbivore"

    return "unknown"


def draw_detection(frame, detection, width, height):

    species = get_species(detection)
    category = get_category(species)

    bbox = detection.get("bbox", [])
    confidence = detection.get("conf", 0)

    if len(bbox) != 4:
        return False

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int((bbox[0] + bbox[2]) * width)
    y2 = int((bbox[1] + bbox[3]) * height)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    color = COLORS[category]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"{species.upper()} | {confidence * 100:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    label_y = max(y1 - 12, text_height + 18)

    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 12),
        (x1 + text_width + 16, label_y + baseline + 5),
        (0, 0, 0),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 12),
        (x1 + text_width + 16, label_y + baseline + 5),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x1 + 8, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return category == "carnivore"


def main():

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    predictions = data.get("predictions", [])

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Could not open video!")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not output.isOpened():
        print("Could not create output video!")
        video.release()
        return

    frame_interval = max(1, round(fps))

    frame_number = 0
    prediction_index = 0

    total_detections = 0
    total_carnivores = 0

    active_detections = []

    while True:

        success, frame = video.read()

        if not success:
            break

        # Update detections approximately once per second
        if frame_number % frame_interval == 0:

            if prediction_index < len(predictions):

                active_detections = predictions[
                    prediction_index
                ].get("detections", [])

                prediction_index += 1

                total_detections += len(active_detections)

                total_carnivores += sum(
                    1 for detection in active_detections
                    if get_category(get_species(detection)) == "carnivore"
                )

        carnivore_count = 0

        # Draw the latest detections on each frame
        for detection in active_detections:

            if draw_detection(frame, detection, width, height):
                carnivore_count += 1

        cv2.putText(
            frame,
            f"Animals detected: {len(active_detections)}",
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        if carnivore_count > 0:
            cv2.putText(
                frame,
                f"ALERT: {carnivore_count} CARNIVORE(S)!",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        output.write(frame)
        frame_number += 1

        if frame_number % 300 == 0:
            print(f"Processed {frame_number} video frames...")

    video.release()
    output.release()

    print("\n========== FAST VIDEO RESULTS ==========")
    print("Video frames:", frame_number)
    print("Prediction frames used:", prediction_index)
    print("Total animal detections:", total_detections)
    print("Total carnivore detections:", total_carnivores)
    print("Saved video:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
