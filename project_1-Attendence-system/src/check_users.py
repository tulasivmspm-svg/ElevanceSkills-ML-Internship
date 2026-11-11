
from database import get_connection

with get_connection() as conn:
    users = conn.execute("""
        SELECT username, role, student_id
        FROM users
    """).fetchall()

print("\nRegistered Login Accounts:")

for username, role, student_id in users:
    print(
        f"Username: {username} | "
        f"Role: {role} | "
        f"Student ID: {student_id}"
    )
