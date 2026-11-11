
import sqlite3
import hashlib
import secrets
import hmac
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "attendance.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                student_id TEXT UNIQUE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                identification_time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(student_id, date)
            )
        """)


def add_user(username, password, role, student_id=None):
    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        200000
    ).hex()

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users
                (username, password_hash, salt, role, student_id)
                VALUES (?, ?, ?, ?, ?)
            """, (username, password_hash, salt, role, student_id))

        return True

    except sqlite3.IntegrityError:
        return False


def verify_login(username, password):
    with get_connection() as conn:
        user = conn.execute("""
            SELECT username, password_hash, salt, role, student_id
            FROM users
            WHERE username = ?
        """, (username,)).fetchone()

    if user is None:
        return None

    stored_hash = user[1]
    salt = user[2]

    entered_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        200000
    ).hex()

    if hmac.compare_digest(stored_hash, entered_hash):
        return {
            "username": user[0],
            "role": user[3],
            "student_id": user[4]
        }

    return None


initialize_database()
