
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# ==========================================
# 1. PROJECT PATHS AND SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

STUDENT_DIR = BASE_DIR / "student_data"
MODEL_DIR = BASE_DIR / "models"

EMBEDDING_FILE = MODEL_DIR / "face_embeddings.npz"

SIMILARITY_THRESHOLD = 0.40


# ==========================================
# 2. LOAD INSIGHTFACE MODEL
# ==========================================

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
    det_thresh=0.1
)

print("InsightFace model loaded successfully!")


# ==========================================
# 3. IMAGE PREPROCESSING FUNCTION
# ==========================================

def preprocess_image(image):

    # Add border around tightly cropped face
    image = cv2.copyMakeBorder(
        image,
        100, 100, 100, 100,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Improve image contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    image = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_LAB2BGR
    )

    # Enlarge image
    image = cv2.resize(
        image,
        (800, 800),
        interpolation=cv2.INTER_CUBIC
    )

    return image


# ==========================================
# 4. BUILD FACE EMBEDDINGS
# ==========================================

def build_embeddings():

    embeddings = []
    labels = []

    if not STUDENT_DIR.exists():
        print("Student data folder not found!")
        return

    for student_folder in STUDENT_DIR.iterdir():

        if not student_folder.is_dir():
            continue

        student_name = student_folder.name

        print(f"\nProcessing student: {student_name}")

        for image_path in student_folder.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            # Read image
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path.name}")
                continue

            print(f"\nImage: {image_path.name}")
            print(f"Original image size: {image.shape}")

            # Apply preprocessing
            processed_image = preprocess_image(image)

            # Detect face
            faces = app.get(processed_image)

            print(f"Faces detected: {len(faces)}")

            if len(faces) == 0:
                print(f"No face detected: {image_path.name}")
                continue

            # Select largest detected face
            face = max(
                faces,
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) *
                    (f.bbox[3] - f.bbox[1])
                )
            )

            # Extract normalized embedding
            embedding = face.normed_embedding.astype(
                np.float32
            )

            embeddings.append(embedding)
            labels.append(student_name)

            print(f"Embedding created for: {student_name}")

    # Check whether embeddings were created
    if len(embeddings) == 0:

        print("\nNo face embeddings were created.")
        print("Please check your student images.")
        return

    # Create models folder
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save embeddings and labels
    np.savez_compressed(
        EMBEDDING_FILE,
        embeddings=np.stack(embeddings),
        labels=np.array(labels)
    )

    print("\n================================")
    print("FACE EMBEDDINGS CREATED!")
    print("================================")

    print(f"Total embeddings: {len(embeddings)}")
    print(f"Saved file: {EMBEDDING_FILE}")


# ==========================================
# 5. RECOGNIZE FACES USING WEBCAM
# ==========================================

def recognize_faces():

    if not EMBEDDING_FILE.exists():

        print("Face embeddings not found!")
        print("Please select option 1 first.")
        return

    # Load saved embeddings
    data = np.load(EMBEDDING_FILE)

    known_embeddings = data["embeddings"]
    known_labels = data["labels"].astype(str)

    print("\nLoaded face embeddings.")
    print(f"Registered images: {len(known_labels)}")

    # Start webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Could not open webcam.")
        return

    print("\nWebcam started.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read webcam frame.")
            break

        # Detect faces in original webcam frame
        faces = app.get(frame)

        for face in faces:

            # Extract normalized embedding
            embedding = face.normed_embedding.astype(
                np.float32
            )

            # Compare with registered embeddings
            similarities = known_embeddings @ embedding

            best_index = int(np.argmax(similarities))

            best_score = float(similarities[best_index])

            # Identify or reject person
            if best_score >= SIMILARITY_THRESHOLD:

                name = known_labels[best_index]

            else:

                name = "Unknown"

            # Face bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Green = recognized, Red = unknown
            color = (
                (0, 255, 0)
                if name != "Unknown"
                else (0, 0, 255)
            )

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display name and similarity
            cv2.putText(
                frame,
                f"{name} ({best_score:.2f})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Display webcam
        cv2.imshow(
            "Face Embedding Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Webcam closed.")


# ==========================================
# 6. MAIN MENU
# ==========================================

if __name__ == "__main__":

    print("\n========== FACE RECOGNITION ==========")
    print("1. Build face embeddings")
    print("2. Test face recognition")
    print("======================================")

    choice = input("\nEnter your choice (1 or 2): ")

    if choice == "1":

        build_embeddings()

    elif choice == "2":

        recognize_faces()

    else:

        print("Invalid choice. Please select 1 or 2.")
