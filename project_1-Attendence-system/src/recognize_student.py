
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# =====================================
# 1. PROJECT SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_DIR = BASE_DIR / "student_data"

IMAGE_COUNT = 30

# =====================================
# 2. LOAD INSIGHTFACE
# =====================================

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
    det_thresh=0.25
)

print("InsightFace loaded successfully!")

# =====================================
# 3. STUDENT REGISTRATION
# =====================================

student_id = input(
    "\nEnter Student ID (example: STU001): "
).strip()

if not student_id:
    raise ValueError("Student ID cannot be empty.")

student_folder = STUDENT_DIR / student_id
student_folder.mkdir(parents=True, exist_ok=True)

print(f"\nRegistering student: {student_id}")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

saved_count = 0
frame_count = 0

print("\nRegistration instructions:")
print("1. Look directly at the camera.")
print("2. Turn your face slightly left and right.")
print("3. Change your head angle slightly.")
print("4. Press C to capture an image.")
print("5. Press Q to finish registration.")
print(f"\nCapture {IMAGE_COUNT} images.")

# =====================================
# 4. CAPTURE FACE IMAGES
# =====================================

try:

    while saved_count < IMAGE_COUNT:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame_count += 1

        faces = app.get(frame)

        if len(faces) == 1:

            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            height, width = frame.shape[:2]

            # Add padding around the detected face
            padding = 30

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            face_crop = frame[y1:y2, x1:x2]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face detected - Press C",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif len(faces) > 1:

            face_crop = None

            cv2.putText(
                frame,
                "Only one person should be visible",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        else:

            face_crop = None

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Images saved: {saved_count}/{IMAGE_COUNT}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Student Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            if face_crop is None:
                print("Cannot capture: exactly one face is required.")
                continue

            image_path = (
                student_folder
                / f"{student_id}_{saved_count + 1}.jpg"
            )

            cv2.imwrite(
                str(image_path),
                face_crop
            )

            saved_count += 1

            print(
                f"Saved image {saved_count}/{IMAGE_COUNT}"
            )

        elif key == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

# =====================================
# 5. REGISTRATION SUMMARY
# =====================================

print("\n================================")
print("REGISTRATION FINISHED")
print("================================")

print(f"Student ID: {student_id}")
print(f"Images saved: {saved_count}")
print(f"Folder: {student_folder}")

if saved_count < 10:
    print("Please capture more images for reliable testing.")
else:
    print("Student images saved successfully!")
