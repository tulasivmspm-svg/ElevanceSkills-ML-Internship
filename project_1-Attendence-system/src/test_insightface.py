
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
    det_thresh=0.1
)

image_path = "student_data/Tulasi/Tulasi_1.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image could not be loaded!")
    raise SystemExit

print("Original image:", image.shape)

# Add space around the tightly cropped face
padded = cv2.copyMakeBorder(
    image,
    100, 100, 100, 100,
    cv2.BORDER_CONSTANT,
    value=(0, 0, 0)
)

# Improve contrast
lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l_channel = clahe.apply(l_channel)

improved = cv2.merge((l_channel, a_channel, b_channel))
improved = cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)

# Enlarge image
improved = cv2.resize(
    improved,
    (800, 800),
    interpolation=cv2.INTER_CUBIC
)

faces = app.get(improved)

print("Faces detected after preprocessing:", len(faces))

if faces:
    print("SUCCESS! InsightFace detected the face.")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            improved,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection Test", improved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Still no face detected.")
