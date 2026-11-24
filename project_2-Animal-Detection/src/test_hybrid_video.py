
from pathlib import Path
import cv2

from hybrid_detection_engine import detect_animals_hybrid


# ---------------- PATHS ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "test_inputs"
    / "video_tests"
    / "antelope.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "hybrid_antelope.mp4"
TEMP_IMAGE_PATH = OUTPUT_DIR / "hybrid_temp_frame.jpg"


# ---------------- DISPLAY SETTINGS ----------------

FONT = cv2.FONT_HERSHEY_SIMPLEX

ANIMAL_FONT_SCALE = 1.0
ANIMAL_TEXT_THICKNESS = 3
BOX_THICKNESS = 4

COUNTER_FONT_SCALE = 1.3
COUNTER_TEXT_THICKNESS = 4


# ---------------- DRAW LABEL ----------------

def draw_label(frame, text, x, y, color):

    text_size, baseline = cv2.getTextSize(
        text,
        FONT,
        ANIMAL_FONT_SCALE,
        ANIMAL_TEXT_THICKNESS
    )

    text_width, text_height = text_size

    # Keep label inside the frame.
    y = max(text_height + 15, y)

    # Filled background behind text.
    cv2.rectangle(
        frame,
        (x, y - text_height - 12),
        (x + text_width + 12, y + baseline),
        color,
        -1
    )

    # White text.
    cv2.putText(
        frame,
        text,
        (x + 6, y - 5),
        FONT,
        ANIMAL_FONT_SCALE,
        (255, 255, 255),
        ANIMAL_TEXT_THICKNESS,
        cv2.LINE_AA
    )


# ---------------- DRAW CARNIVORE COUNTER ----------------

def draw_counter(frame, carnivore_count):

    text = f"CARNIVORES: {carnivore_count}"

    cv2.rectangle(
        frame,
        (15, 15),
        (480, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (30, 65),
        FONT,
        COUNTER_FONT_SCALE,
        (0, 0, 255),
        COUNTER_TEXT_THICKNESS,
        cv2.LINE_AA
    )


# ---------------- VIDEO PROCESSING ----------------

def process_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise ValueError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    # Analyse one frame every second.
    frame_interval = max(1, round(fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video.")

    frame_index = 0
    analysed_frames = 0

    latest_detections = []
    latest_carnivore_count = 0

    print("\nStarting hybrid video detection...")
    print("Analysing one frame per second.\n")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # -------- HYBRID ANALYSIS --------

            if frame_index % frame_interval == 0:

                cv2.imwrite(
                    str(TEMP_IMAGE_PATH),
                    frame
                )

                (
                    annotated_image,
                    latest_carnivore_count,
                    latest_detections
                ) = detect_animals_hybrid(
                    TEMP_IMAGE_PATH
                )

                analysed_frames += 1

                current_time = frame_index / fps

                print(
                    f"Analysed frame at {current_time:.1f}s | "
                    f"Animals: {len(latest_detections)} | "
                    f"Carnivores: {latest_carnivore_count}"
                )

            # -------- DRAW DETECTIONS --------

            output_frame = frame.copy()

            for detection in latest_detections:

                animal = detection["animal"]
                diet = detection["diet_category"]

                x1, y1, x2, y2 = detection["bbox"]

                if diet == "carnivore":
                    color = (0, 0, 255)

                elif diet == "herbivore":
                    color = (0, 255, 0)

                else:
                    color = (0, 255, 255)

                # Bounding box.
                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    BOX_THICKNESS
                )

                # Animal name and confidence.
                label = (
                    f"{animal} | "
                    f"{detection['confidence']:.2f}"
                )

                draw_label(
                    output_frame,
                    label,
                    x1,
                    y1,
                    color
                )

            # -------- CARNIVORE COUNTER --------

            draw_counter(
                output_frame,
                latest_carnivore_count
            )

            writer.write(output_frame)

            frame_index += 1

    finally:

        cap.release()
        writer.release()

        if TEMP_IMAGE_PATH.exists():
            TEMP_IMAGE_PATH.unlink()

    print("\n========== HYBRID VIDEO RESULTS ==========")
    print(f"Video frames: {total_frames}")
    print(f"Analysed frames: {analysed_frames}")
    print(f"Video duration: {total_frames / fps:.2f} seconds")
    print(f"Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_video()
