
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "attendance_test.csv"
EXCEL_FILE = BASE_DIR / "attendance_report.xlsx"

if not CSV_FILE.exists():
    print("Attendance CSV file not found!")
else:
    df = pd.read_csv(CSV_FILE)

    df.to_excel(EXCEL_FILE, index=False)

    print("Excel report generated successfully!")
    print(f"Saved at: {EXCEL_FILE}")
