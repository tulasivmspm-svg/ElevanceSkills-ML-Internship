
import cv2
from pathlib import Path
from insightface.app import FaceAnalysis

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_DIR = BASE_DIR / "student_data"

IMAGE_COUNT = 10


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

if not student_id or any(c in student_id for c in r'\/:*?"<>|'):
    raise ValueError("Please enter a valid student ID.")

student_folder = STUDENT_DIR / student_id
student_folder.mkdir(parents=True, exist_ok=True)

print(f"\nRegistering student: {student_id}")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

saved_count = 0

print("\nRegistration instructions:")
print("Look at the camera and change your face angle slightly.")
print("Press C to capture an image.")
print("Press Q to finish.")
print(f"Target: {IMAGE_COUNT} images.")

# =====================================
# 4. CAPTURE IMAGES
# =====================================

try:

    while saved_count < IMAGE_COUNT:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        faces = app.get(frame)
        face_crop = None

        if len(faces) == 1:

            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)

            height, width = frame.shape[:2]

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

            message = "Face detected - Press C"

        elif len(faces) > 1:

            message = "Only one person should be visible"

        else:

            message = "No face detected"

        cv2.putText(
            frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if len(faces) == 1 else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Saved: {saved_count}/{IMAGE_COUNT}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Student Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            if face_crop is None or face_crop.size == 0:
                print("Cannot capture. Keep exactly one face visible.")
                continue

            image_path = (
                student_folder /
                f"{student_id}_{saved_count + 1}.jpg"
            )

            cv2.imwrite(str(image_path), face_crop)

            saved_count += 1

            print(f"Saved image {saved_count}/{IMAGE_COUNT}")

        elif key == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

# =====================================
# 5. REGISTRATION SUMMARY
# =====================================

print("\nRegistration completed!")
print(f"Student: {student_id}")
print(f"Images saved: {saved_count}/{IMAGE_COUNT}")
print(f"Location: {student_folder}")
