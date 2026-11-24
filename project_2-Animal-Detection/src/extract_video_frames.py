
import cv2
import os

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VIDEO_PATH = os.path.join(
    PROJECT_DIR, "test_inputs", "video_tests", "antelope.mp4"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "test_inputs", "video_frames"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Error: Could not open video!")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Total video frames:", total_frames)

interval = max(1, int(fps))  # Extract one frame per second

frame_number = 0
saved_count = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_number % interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(output_path, frame)
        saved_count += 1

    frame_number += 1

video.release()

print("\nFrame extraction completed!")
print("Total frames saved:", saved_count)
print("Saved location:", OUTPUT_DIR)
