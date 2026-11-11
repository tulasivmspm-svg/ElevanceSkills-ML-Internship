
# AI-Based Student Attendance Management System

## 1. Project Overview

The AI-Based Student Attendance Management System is a facial recognition-based attendance application designed to automate student attendance recording.

The system combines face recognition, facial emotion classification, and a web-based dashboard to provide an organized attendance management process.

## 2. Objectives

- Automate student identification using facial recognition.
- Record attendance with student ID, date, and time.
- Estimate facial expressions using a CNN-based emotion classification model.
- Provide separate admin and student dashboards.
- Generate attendance reports in CSV and Excel formats.
- Reduce manual attendance-recording work.

## 3. Technologies Used

- Python
- PyTorch
- OpenCV
- InsightFace
- ONNX Runtime
- Streamlit
- NumPy
- Pandas
- SQLite
- Matplotlib
- Scikit-learn

## 4. Main Features

### Face Recognition
- Uses InsightFace for face detection and recognition.
- Generates and compares face embeddings.
- Identifies registered students.

### Emotion Detection
- Uses a custom Convolutional Neural Network (CNN).
- Trained using the FER2013 facial expression dataset.
- Classifies facial expressions into emotion categories.

### Attendance Management
- Records student ID, name, date, and time.
- Applies an attendance time window.
- Prevents duplicate attendance entries for the same student and date.
- Stores attendance information.

### Admin Dashboard
- Admin login.
- Student registration.
- Registered student management.
- Attendance report viewing.
- CSV and Excel report downloads.

### Student Dashboard
- Student login.
- Camera-based face verification.
- Attendance marking.
- Attendance history viewing.

## 5. Project Structure

```text
project_1-Attendence-system/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── attendance_system.py
│   ├── database.py
│   ├── capture_student.py
│   ├── face_embedding_recognition.py
│   ├── recognize_student.py
│   ├── create_admin.py
│   ├── check_users.py
│   ├── convert_attendance_to_excel.py
│   ├── CNN_train_emotion_model.py
│   ├── evaluate_emotion_model.py
│   ├── explore_dataset.py
│   └── visualize_dataset.py
│
├── models/
│   └── Model files and evaluation results
│
└── student_data/
    └── Registered student images (private)
```

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/tulasivmspm-svg/ElevanceSkills-ML-Internship.git
```

Navigate to the project directory:

```bash
cd ElevanceSkills-ML-Internship/project_1-Attendence-system
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU-enabled PyTorch, install the appropriate CUDA-compatible PyTorch build separately according to the official PyTorch installation instructions.

## 7. Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 8. Attendance Timing

The application is configured with an attendance window of:

- Start time: 9:30 AM
- End time: 10:00 AM

Attendance outside the configured window is restricted.

## 9. Model Evaluation

The facial emotion classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The face recognition component uses InsightFace embeddings to compare registered student faces with camera input.

## 10. Limitations

- Face recognition accuracy can vary with lighting, camera quality, and facial angles.
- Facial expression classification does not determine a person's actual emotional state.
- Photo-based face verification can be vulnerable to spoofing.
- The current implementation requires further security and deployment improvements before real-world institutional use.

## 11. Future Improvements

- Add liveness detection to reduce photo-based spoofing.
- Improve face recognition under different lighting conditions.
- Enhance emotion classification performance.
- Add more detailed attendance analytics.
- Improve database security and deployment.
- Develop a secure cloud-based attendance platform.

## 12. Privacy Notice

Student photographs, face embeddings, login credentials, and attendance records are private information.

These files should not be uploaded to a public GitHub repository. Use appropriate consent, access controls, and secure storage when handling student information.

## 13. Author

Developed as part of the ElevanceSkills Machine Learning Internship.
