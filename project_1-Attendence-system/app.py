
import streamlit as st
import pandas as pd
import secrets
import re
import cv2
import csv
import numpy as np

from io import BytesIO
from pathlib import Path
from datetime import datetime, time
from insightface.app import FaceAnalysis

from src.database import (
    add_user,
    verify_login,
    get_connection
)

# =====================================
# 1. PROJECT SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent
STUDENT_DATA = BASE_DIR / "student_data"

EMBEDDING_FILE = BASE_DIR / "models" / "face_embeddings.npz"
ATTENDANCE_FILE = BASE_DIR / "attendance_test.csv"

SIMILARITY_THRESHOLD = 0.40

WINDOW_START = time(9, 30)
WINDOW_END = time(10, 0)

st.set_page_config(
    page_title="Student Attendance System",
    page_icon="📚",
    layout="wide"
)


# =====================================
# 2. LOAD INSIGHTFACE MODEL
# =====================================

@st.cache_resource
def load_face_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=-1,
        det_size=(640, 640),
        det_thresh=0.25
    )

    return model


# =====================================
# 3. HELPER FUNCTIONS
# =====================================

def normalize_name(name):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        name.casefold()
    ).strip("_")


def is_attendance_time():

    current_time = datetime.now().time()

    return WINDOW_START <= current_time <= WINDOW_END


def recognize_student(image):

    if not EMBEDDING_FILE.exists():

        raise FileNotFoundError(
            "Face embeddings not found. "
            "Generate embeddings first."
        )

    face_model = load_face_model()

    image_bytes = np.frombuffer(
        image.getvalue(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError("Unable to read captured image.")

    faces = face_model.get(frame)

    if len(faces) != 1:

        raise ValueError(
            "Please capture an image containing exactly one face."
        )

    data = np.load(EMBEDDING_FILE)

    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)

    captured_embedding = (
        faces[0].normed_embedding.astype(np.float32)
    )

    similarities = embeddings @ captured_embedding

    best_index = int(np.argmax(similarities))

    best_score = float(similarities[best_index])

    recognized_name = labels[best_index]

    return recognized_name, best_score


# =====================================
# 4. LOGIN SYSTEM
# =====================================

if "user" not in st.session_state:

    st.session_state.user = None


st.title("📚 Student Attendance System")


if st.session_state.user is None:

    st.subheader("Login")

    with st.form("login_form"):

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button("Login")

    if submitted:

        user = verify_login(
            username.strip(),
            password
        )

        if user:

            st.session_state.user = user

            st.rerun()

        else:

            st.error("Invalid username or password.")

    st.stop()


user = st.session_state.user

st.success(f"Welcome, {user['username']}!")


if st.button("Logout"):

    st.session_state.user = None

    st.rerun()


# =====================================
# 5. ADMIN DASHBOARD
# =====================================

if user["role"] == "admin":

    st.header("Admin Dashboard")

    # ---------------------------------
    # STUDENT REGISTRATION
    # ---------------------------------

    st.subheader("Register New Student")

    with st.form("student_registration"):

        student_id = st.text_input(
            "Student ID (Example: STU003)"
        )

        student_name = st.text_input("Student Name")

        student_username = st.text_input(
            "Student Login Username"
        )

        student_password = st.text_input(
            "Student Login Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Register Student"
        )

    st.write(
        "Upload clear individual face images of the student."
    )

    uploaded_images = st.file_uploader(
        "Upload student face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    camera_image = st.camera_input(
        "Or capture a face image"
    )

    if submitted:

        student_id = student_id.strip().upper()

        student_name = student_name.strip()

        student_username = student_username.strip()

        if not all([
            student_id,
            student_name,
            student_username,
            student_password
        ]):

            st.error("Please complete all student details.")

        elif not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            student_id
        ):

            st.error("Invalid Student ID.")

        elif not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            student_username
        ):

            st.error("Invalid student username.")

        elif len(student_password) < 8:

            st.error(
                "Password must contain at least 8 characters."
            )

        elif not uploaded_images and camera_image is None:

            st.error("Please upload or capture face images.")

        else:

            try:

                with get_connection() as conn:

                    existing = conn.execute(
                        """
                        SELECT 1 FROM students
                        WHERE student_id = ?
                        """,
                        (student_id,)
                    ).fetchone()

                if existing:

                    st.error(
                        "This Student ID is already registered."
                    )

                else:

                    success = add_user(
                        student_username,
                        student_password,
                        "student",
                        student_id
                    )

                    if not success:

                        st.error(
                            "This username or Student ID already exists."
                        )

                    else:

                        with get_connection() as conn:

                            conn.execute(
                                """
                                INSERT INTO students
                                (student_id, student_name)
                                VALUES (?, ?)
                                """,
                                (student_id, student_name)
                            )

                        safe_name = re.sub(
                            r"[^A-Za-z0-9_-]+",
                            "_",
                            student_name
                        ).strip("_")

                        student_folder = (
                            STUDENT_DATA / safe_name
                        )

                        student_folder.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        saved_count = 0

                        for image_file in uploaded_images or []:

                            extension = Path(
                                image_file.name
                            ).suffix.lower()

                            image_path = (
                                student_folder /
                                f"admin_{secrets.token_hex(6)}{extension}"
                            )

                            image_path.write_bytes(
                                image_file.getvalue()
                            )

                            saved_count += 1

                        if camera_image is not None:

                            image_path = (
                                student_folder /
                                f"admin_{secrets.token_hex(6)}.jpg"
                            )

                            image_path.write_bytes(
                                camera_image.getvalue()
                            )

                            saved_count += 1

                        st.success(
                            f"Student registered successfully! "
                            f"{saved_count} image(s) saved."
                        )

                        st.info(
                            "Regenerate face embeddings before "
                            "the student uses face verification."
                        )

            except Exception as error:

                st.error(
                    f"Registration error: {error}"
                )

    # ---------------------------------
    # REGISTERED STUDENTS
    # ---------------------------------

    st.divider()

    st.subheader("Registered Students")

    with get_connection() as conn:

        students = pd.read_sql_query(
            """
            SELECT student_id, student_name
            FROM students
            """,
            conn
        )

    st.dataframe(
        students,
        use_container_width=True,
        hide_index=True
    )

    # =================================
    # 6. ADMIN ATTENDANCE REPORT
    # =================================

    st.divider()

    st.header("Attendance Management")

    selected_date = st.date_input(
        "Select Attendance Date",
        value=datetime.now().date()
    )

    date_string = selected_date.strftime("%Y-%m-%d")

    with get_connection() as conn:

        attendance = pd.read_sql_query(
            """
            SELECT
                a.student_id AS Student_ID,
                s.student_name AS Student_Name,
                a.date AS Date,
                a.identification_time AS Identification_Time,
                a.status AS Status
            FROM attendance a
            LEFT JOIN students s
            ON a.student_id = s.student_id
            WHERE a.date = ?
            ORDER BY a.identification_time
            """,
            conn,
            params=(date_string,)
        )

    # Attendance metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Attendance Records",
        len(attendance)
    )

    col2.metric(
        "Students Present",
        attendance["Student_ID"].nunique()
    )

    col3.metric(
        "Registered Students",
        len(students)
    )

    # Attendance table

    st.subheader("Daily Attendance Records")

    if attendance.empty:

        st.info(
            "No attendance records found for this date."
        )

    else:

        st.dataframe(
            attendance,
            use_container_width=True,
            hide_index=True
        )

        # CSV DOWNLOAD

        st.download_button(
            label="Download CSV Report",
            data=attendance.to_csv(index=False),
            file_name=f"attendance_{date_string}.csv",
            mime="text/csv"
        )

        # EXCEL DOWNLOAD

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            attendance.to_excel(
                writer,
                index=False,
                sheet_name="Attendance"
            )

        st.download_button(
            label="Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name=f"attendance_{date_string}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =====================================
# 7. STUDENT DASHBOARD
# =====================================

elif user["role"] == "student":

    st.header("Student Attendance Dashboard")

    student_id = user["student_id"]

    with get_connection() as conn:

        student = conn.execute(
            """
            SELECT student_name
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()

    if student is None:

        st.error("Student record not found.")

        st.stop()

    student_name = student[0]

    st.write(f"Student Name: {student_name}")

    st.write(f"Student ID: {student_id}")

    st.divider()

    st.subheader("Attendance Status")

    if is_attendance_time():

        st.success(
            "Attendance window is OPEN (9:30 AM - 10:00 AM)."
        )

    else:

        st.warning(
            "Attendance is available only between "
            "9:30 AM and 10:00 AM."
        )

    st.divider()

    st.subheader("Face Verification")

    photo = st.camera_input(
        "Capture your face to mark attendance"
    )

    if st.button(
        "Verify Face and Mark Attendance",
        type="primary"
    ):

        if photo is None:

            st.error("Please capture your face first.")

        elif not is_attendance_time():

            st.error(
                "Attendance marking is currently closed."
            )

        else:

            try:

                recognized_name, score = recognize_student(
                    photo
                )

                st.write(
                    f"Face matching score: {score:.3f}"
                )

                if score < SIMILARITY_THRESHOLD:

                    st.error("Face verification failed.")

                elif (
                    normalize_name(recognized_name)
                    != normalize_name(student_name)
                ):

                    st.error(
                        "The captured face does not match "
                        "your registered account."
                    )

                else:

                    now = datetime.now()

                    date_value = now.strftime("%Y-%m-%d")

                    time_value = now.strftime("%H:%M:%S")

                    # Save attendance in SQLite

                    with get_connection() as conn:

                        cursor = conn.execute(
                            """
                            INSERT OR IGNORE INTO attendance
                            (
                                student_id,
                                date,
                                identification_time,
                                status
                            )
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                student_id,
                                date_value,
                                time_value,
                                "Present"
                            )
                        )

                        inserted = cursor.rowcount

                    if inserted == 0:

                        st.warning(
                            "Your attendance has already "
                            "been marked today."
                        )

                    else:

                        # Save attendance in CSV

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
                                date_value,
                                time_value,
                                "Present"
                            ])

                        st.success(
                            "Attendance marked successfully!"
                        )

                        st.write(
                            f"Student: {student_name}"
                        )

                        st.write(
                            f"Student ID: {student_id}"
                        )

                        st.write(
                            f"Date: {date_value}"
                        )

                        st.write(
                            f"Time: {time_value}"
                        )

            except Exception as error:

                st.error(
                    f"Face verification error: {error}"
                )

    # =================================
    # 8. STUDENT ATTENDANCE HISTORY
    # =================================

    st.divider()

    st.subheader("My Attendance History")

    with get_connection() as conn:

        history = pd.read_sql_query(
            """
            SELECT
                student_id,
                date,
                identification_time,
                status
            FROM attendance
            WHERE student_id = ?
            ORDER BY date DESC
            """,
            conn,
            params=(student_id,)
        )

    if history.empty:

        st.info("No attendance records found yet.")

    else:

        st.dataframe(
            history,
            use_container_width=True
        )


else:

    st.error("Invalid account role.")
