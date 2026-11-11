
import cv2
import csv
import numpy as np
import re

from pathlib import Path
from datetime import datetime, time
from insightface.app import FaceAnalysis
from database import get_connection

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
EMBEDDING_FILE = MODEL_DIR / "face_embeddings.npz"

TEST_MODE = True

if TEST_MODE:
    ATTENDANCE_FILE = BASE_DIR / "attendance_test.csv"
else:
    ATTENDANCE_FILE = BASE_DIR / "attendance.csv"

SIMILARITY_THRESHOLD = 0.40

WINDOW_START = time(9, 30)
WINDOW_END = time(10, 0)

REQUIRED_FRAMES = 5

# ==========================================
# 2. LOAD STUDENT DATABASE
# ==========================================

def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


with get_connection() as conn:
    rows = conn.execute("""
        SELECT student_id, student_name
        FROM students
    """).fetchall()

student_mapping = {}

for student_id, student_name in rows:
    student_mapping[normalize_name(student_name)] = {
        "student_id": student_id,
        "student_name": student_name
    }

print("\nStudents in database:")

for student in student_mapping.values():
    print(student["student_id"], "-", student["student_name"])

# ==========================================
# 3. CHECK ATTENDANCE TIME
# ==========================================

def is_attendance_time():
    current_time = datetime.now().time()
    return WINDOW_START <= current_time <= WINDOW_END

# ==========================================
# 4. LOAD FACE EMBEDDINGS
# ==========================================

if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(
        "Face embeddings not found. "
        "Run face_embedding_recognition.py and choose option 1."
    )

data = np.load(EMBEDDING_FILE)

known_embeddings = data["embeddings"].astype(np.float32)
known_labels = data["labels"].astype(str)

print("\nRegistered face labels:")

for label in sorted(set(known_labels)):
    print("-", label)

# ==========================================
# 5. LOAD INSIGHTFACE
# ==========================================

print("\nLoading InsightFace...")

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

# ==========================================
# 6. PREPARE ATTENDANCE CSV
# ==========================================

ATTENDANCE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

if not ATTENDANCE_FILE.exists():

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student_ID",
            "Student_Name",
            "Date",
            "Identification_Time",
            "Status"
        ])

today = datetime.now().strftime("%Y-%m-%d")

marked_today = set()

with open(
    ATTENDANCE_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        if row["Date"] == today:
            marked_today.add(row["Student_ID"])

# ==========================================
# 7. MARK ATTENDANCE
# ==========================================

def mark_attendance(face_label):

    global marked_today

    key = normalize_name(face_label)

    student = student_mapping.get(key)

    if student is None:
        print(
            f"No database record found for face label: {face_label}"
        )
        return

    student_id = student["student_id"]
    student_name = student["student_name"]

    if student_id in marked_today:
        return

    now = datetime.now()

    if not TEST_MODE and not is_attendance_time():
        print("Attendance marking is currently closed.")
        return

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            student_id,
            student_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    marked_today.add(student_id)

    print(
        f"Attendance marked: {student_name} "
        f"({student_id}) at {now.strftime('%H:%M:%S')}"
    )

# ==========================================
# 8. START ATTENDANCE SYSTEM
# ==========================================

if not TEST_MODE and not is_attendance_time():

    print("\nAttendance marking is currently closed.")
    print("Allowed time: 9:30 AM to 10:00 AM.")
    raise SystemExit

print("\nAttendance system started!")

if TEST_MODE:
    print("TEST MODE: Time restriction is bypassed.")
    print(f"Saving to: {ATTENDANCE_FILE}")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

recognition_counts = {}

print("\nPress Q to stop attendance.")

# ==========================================
# 9. LIVE FACE RECOGNITION
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam.")
            break

        faces = app.get(frame)

        detected_students = set()

        for face in faces:

            embedding = face.normed_embedding.astype(np.float32)

            similarities = known_embeddings @ embedding

            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])

            if best_score >= SIMILARITY_THRESHOLD:

                face_label = known_labels[best_index]
                detected_students.add(face_label)

                recognition_counts[face_label] = (
                    recognition_counts.get(face_label, 0) + 1
                )

                if recognition_counts[face_label] >= REQUIRED_FRAMES:
                    mark_attendance(face_label)

                display_name = face_label
                color = (0, 255, 0)

            else:

                display_name = "Unknown"
                color = (0, 0, 255)

            x1, y1, x2, y2 = face.bbox.astype(int)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{display_name} ({best_score:.2f})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        for student in list(recognition_counts):

            if student not in detected_students:
                recognition_counts[student] = 0

        cv2.putText(
            frame,
            "Attendance System",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Student Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nAttendance system stopped.")
    print(f"Attendance file: {ATTENDANCE_FILE}")
