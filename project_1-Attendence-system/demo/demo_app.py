
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from io import BytesIO

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Attendance System - Demo",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Attendance Management System")
st.caption("Public demonstration | Sample data only")

st.info(
    "This is a demonstration application. "
    "Face recognition is disabled. Attendance is simulated "
    "using sample accounts and sample records."
)

# ---------------- SAMPLE ACCOUNTS ----------------

ACCOUNTS = {
    "admin": {
        "password": "demo123",
        "role": "Admin",
        "name": "Demo Administrator"
    },
    "demo_student1": {
        "password": "demo123",
        "role": "Student",
        "name": "Sample Student A",
        "student_id": "STU001"
    },
    "demo_student2": {
        "password": "demo123",
        "role": "Student",
        "name": "Sample Student B",
        "student_id": "STU002"
    },
    "demo_student3": {
        "password": "demo123",
        "role": "Student",
        "name": "Sample Student C",
        "student_id": "STU003"
    }
}

# ---------------- SAMPLE ATTENDANCE DATA ----------------

def create_sample_data():
    records = []

    students = [
        ("STU001", "Sample Student A"),
        ("STU002", "Sample Student B"),
        ("STU003", "Sample Student C")
    ]

    for days_ago in range(7, 0, -1):
        record_date = date.today() - timedelta(days=days_ago)

        for index, (student_id, name) in enumerate(students):
            records.append({
                "Student ID": student_id,
                "Student Name": name,
                "Date": record_date,
                "Time": "09:45:00",
                "Status": "Present" if (days_ago + index) % 4 != 0 else "Absent"
            })

    return pd.DataFrame(records)


if "attendance" not in st.session_state:
    st.session_state.attendance = create_sample_data()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- LOGIN ----------------

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()


if not st.session_state.logged_in:

    st.subheader("🔐 Demo Login")

    with st.container(border=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):

            if username in ACCOUNTS and ACCOUNTS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()

            else:
                st.error("Invalid username or password.")

    st.markdown("### Demo Login Credentials")

    credentials = pd.DataFrame([
        ["Admin", "admin", "demo123"],
        ["Student 1", "demo_student1", "demo123"],
        ["Student 2", "demo_student2", "demo123"],
        ["Student 3", "demo_student3", "demo123"]
    ], columns=["Role", "Username", "Password"])

    st.dataframe(credentials, hide_index=True, use_container_width=True)

    st.caption(
        "These are public demonstration credentials. "
        "Never use them for a real attendance system."
    )

# ---------------- DASHBOARDS ----------------

else:

    account = ACCOUNTS[st.session_state.username]

    st.sidebar.success(f"Logged in as {account['name']}")
    st.sidebar.write(f"Role: {account['role']}")

    if st.sidebar.button("Logout"):
        logout()

    attendance = st.session_state.attendance.copy()

    # ---------------- ADMIN DASHBOARD ----------------

    if account["role"] == "Admin":

        st.header("👨‍💼 Admin Dashboard")

        total_students = len([
            user for user in ACCOUNTS.values()
            if user["role"] == "Student"
        ])

        today_records = attendance[
            attendance["Date"] == date.today()
        ]

        present_today = len(
            today_records[today_records["Status"] == "Present"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", total_students)
        col2.metric("Present Today", present_today)
        col3.metric(
            "Attendance Records",
            len(attendance)
        )

        st.divider()

        st.subheader("📋 Attendance Report")

        selected_date = st.date_input(
            "Select attendance date",
            value=date.today()
        )

        filtered = attendance[
            attendance["Date"] == selected_date
        ]

        if filtered.empty:
            st.warning("No attendance records for this date.")
        else:
            st.dataframe(
                filtered.sort_values("Student ID"),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("📊 Complete Attendance History")

        st.dataframe(
            attendance.sort_values(
                ["Date", "Student ID"],
                ascending=[False, True]
            ),
            use_container_width=True,
            hide_index=True
        )

        csv_data = attendance.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV Report",
            data=csv_data,
            file_name="demo_attendance_report.csv",
            mime="text/csv"
        )

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
            "⬇️ Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="demo_attendance_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ---------------- STUDENT DASHBOARD ----------------

    else:

        st.header("👨‍🎓 Student Dashboard")

        student_id = account["student_id"]
        student_name = account["name"]

        st.write(f"**Student Name:** {student_name}")
        st.write(f"**Student ID:** {student_id}")

        st.divider()

        st.subheader("📸 Attendance Verification")

        st.warning(
            "Face recognition is disabled in this public demo. "
            "The button below simulates successful verification."
        )

        if st.button(
            "Mark Demo Attendance",
            type="primary"
        ):

            today = date.today()

            already_marked = (
                (st.session_state.attendance["Student ID"] == student_id)
                &
                (st.session_state.attendance["Date"] == today)
                &
                (st.session_state.attendance["Status"] == "Present")
            ).any()

            if already_marked:
                st.warning("Attendance has already been marked today.")

            else:
                new_record = pd.DataFrame([{
                    "Student ID": student_id,
                    "Student Name": student_name,
                    "Date": today,
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Status": "Present"
                }])

                st.session_state.attendance = pd.concat(
                    [st.session_state.attendance, new_record],
                    ignore_index=True
                )

                st.success("Demo attendance marked successfully!")
                st.rerun()

        st.divider()

        st.subheader("📅 My Attendance History")

        my_records = st.session_state.attendance[
            st.session_state.attendance["Student ID"] == student_id
        ].copy()

        st.dataframe(
            my_records.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        present_count = len(
            my_records[my_records["Status"] == "Present"]
        )

        total_count = len(my_records)

        attendance_percentage = (
            present_count / total_count * 100
            if total_count > 0 else 0
        )

        col1, col2 = st.columns(2)

        col1.metric("Present Days", present_count)
        col2.metric(
            "Attendance Percentage",
            f"{attendance_percentage:.1f}%"
        )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Student Attendance Management System | "
    "ElevanceSkills ML Internship | Public Demo"
)
