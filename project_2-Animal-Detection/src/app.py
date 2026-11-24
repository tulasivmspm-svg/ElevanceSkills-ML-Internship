
import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd

from pathlib import Path
from PIL import Image

from hybrid_detection_engine import detect_animals_hybrid

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Wildlife Animal Detection",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 AI-Based Wildlife Animal Detection System")
st.caption("Hybrid animal detection using YOLOv8 and Google SpeciesNet")
st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Detection Information")

st.sidebar.markdown(
    """
    ### Detection colours

    🔴 **Red:** Carnivores

    🟢 **Green:** Herbivores

    🟡 **Yellow:** Unknown / Other

    ---
    ### AI Models

    - YOLOv8: Animal detection
    - SpeciesNet: Species identification
    - Hybrid engine: Combines both models
    """
)

st.sidebar.info(
    "Video analysis processes one frame per second. "
    "Latest detections are displayed between analysed frames."
)

# =====================================================
# VIDEO LABEL FUNCTIONS - UPDATED BIGGER TEXT
# =====================================================

def draw_video_label(frame, text, x, y, color):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3

    text_size, baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    text_width, text_height = text_size

    frame_height, frame_width = frame.shape[:2]

    # Keep label inside the video frame
    x = max(5, min(x, frame_width - text_width - 20))
    y = max(text_height + 20, y)

    # Dark background behind text
    cv2.rectangle(
        frame,
        (x, y - text_height - 15),
        (min(x + text_width + 20, frame_width - 1),
         min(y + baseline + 5, frame_height - 1)),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 8, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_video_counter(frame, count):

    text = f"CARNIVORES: {count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 4

    text_size, baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (10, 10),
        (text_width + 55, text_height + 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (25, text_height + 25),
        font,
        font_scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )


def annotate_video_frame(frame, detections, carnivore_count):

    output = frame.copy()

    for detection in detections:

        animal = detection["animal"]
        diet = detection["diet_category"]

        x1, y1, x2, y2 = detection["bbox"]

        if diet == "carnivore":
            color = (0, 0, 255)

        elif diet == "herbivore":
            color = (0, 255, 0)

        else:
            color = (0, 255, 255)

        # Bigger bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            5
        )

        label = (
            f"{animal} | {detection['confidence']:.2f}"
        )

        draw_video_label(
            output,
            label,
            x1,
            y1,
            color
        )

    draw_video_counter(output, carnivore_count)

    return output


# =====================================================
# DETECTION SUMMARY
# =====================================================

def show_detection_summary(detections, carnivore_count):

    total = len(detections)

    herbivore_count = sum(
        1 for d in detections
        if d["diet_category"] == "herbivore"
    )

    unknown_count = sum(
        1 for d in detections
        if d["diet_category"] == "unknown"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Animals", total)
    col2.metric("Carnivores", carnivore_count)
    col3.metric("Herbivores", herbivore_count)
    col4.metric("Unknown", unknown_count)

    if carnivore_count > 0:
        st.error(
            f"⚠️ Alert! {carnivore_count} carnivore(s) detected!"
        )
    else:
        st.success("No carnivores detected in this image/frame.")


def show_detection_table(detections):

    if not detections:
        st.info("No animals detected.")
        return

    rows = []

    for index, detection in enumerate(detections, start=1):

        rows.append({
            "Animal": index,
            "Species": detection["animal"],
            "Diet": detection["diet_category"],
            "Detection Confidence": round(
                detection["confidence"], 3
            ),
            "Classification Confidence": round(
                detection["classification_score"], 3
            ),
            "Source": detection["classification_source"]
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# DETECTION MODE
# =====================================================

mode = st.radio(
    "Choose Detection Mode",
    ["Image Detection", "Video Detection"],
    horizontal=True
)


# =====================================================
# IMAGE DETECTION
# =====================================================

if mode == "Image Detection":

    st.header("🖼️ Image Animal Detection")

    uploaded_image = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image_bytes = uploaded_image.getvalue()

        image = Image.open(uploaded_image).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image_array, use_container_width=True)

        if st.button(
            "🔍 Detect Animals",
            type="primary",
            key="image_detect"
        ):

            with st.spinner("Running YOLOv8 and SpeciesNet..."):

                try:

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp_file:

                        temp_file.write(image_bytes)
                        temp_image_path = Path(temp_file.name)

                    try:
                        (
                            annotated,
                            carnivore_count,
                            detections
                        ) = detect_animals_hybrid(temp_image_path)

                    finally:
                        temp_image_path.unlink(missing_ok=True)

                    annotated_rgb = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    with col2:
                        st.subheader("Detection Result")
                        st.image(
                            annotated_rgb,
                            use_container_width=True
                        )

                    st.divider()
                    st.header("📊 Detection Summary")

                    show_detection_summary(
                        detections,
                        carnivore_count
                    )

                    st.subheader("Detected Animal Details")
                    show_detection_table(detections)

                    success, encoded_image = cv2.imencode(
                        ".jpg", annotated
                    )

                    if success:
                        st.download_button(
                            "⬇️ Download Detection Image",
                            data=encoded_image.tobytes(),
                            file_name="hybrid_animal_detection.jpg",
                            mime="image/jpeg"
                        )

                except Exception as error:
                    st.error(f"Image detection failed: {error}")


# =====================================================
# VIDEO DETECTION
# =====================================================

elif mode == "Video Detection":

    st.header("🎬 Video Animal Detection")

    uploaded_video = st.file_uploader(
        "Upload an animal video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.subheader("Original Video")
        st.video(uploaded_video)

        if st.button(
            "🚀 Start Hybrid Video Detection",
            type="primary",
            key="video_detect"
        ):

            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_dir = Path(temp_dir)

                input_path = (
                    temp_dir / f"input{Path(uploaded_video.name).suffix}"
                )

                output_path = temp_dir / "hybrid_detected_video.mp4"

                input_path.write_bytes(uploaded_video.getvalue())

                cap = cv2.VideoCapture(str(input_path))

                if not cap.isOpened():
                    st.error("Unable to open uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0 or width <= 0 or height <= 0:
                    cap.release()
                    st.error("Invalid video properties.")
                    st.stop()

                frame_interval = max(1, round(fps))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    cap.release()
                    st.error("Unable to create output video.")
                    st.stop()

                frame_number = 0
                analysed_frames = 0

                total_animal_detections = 0
                total_carnivore_sightings = 0
                max_carnivores_in_frame = 0

                latest_detections = []
                latest_carnivore_count = 0

                try:

                    with st.spinner(
                        "Processing video with hybrid AI models..."
                    ):

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break

                            # Analyse one frame per second
                            if frame_number % frame_interval == 0:

                                frame_path = (
                                    temp_dir / "current_frame.jpg"
                                )

                                cv2.imwrite(str(frame_path), frame)

                                (
                                    annotated,
                                    latest_carnivore_count,
                                    latest_detections
                                ) = detect_animals_hybrid(frame_path)

                                analysed_frames += 1

                                total_animal_detections += len(
                                    latest_detections
                                )

                                total_carnivore_sightings += (
                                    latest_carnivore_count
                                )

                                max_carnivores_in_frame = max(
                                    max_carnivores_in_frame,
                                    latest_carnivore_count
                                )

                            output_frame = annotate_video_frame(
                                frame,
                                latest_detections,
                                latest_carnivore_count
                            )

                            writer.write(output_frame)

                            frame_number += 1

                            if (
                                frame_number % max(1, int(fps)) == 0
                                or frame_number == total_frames
                            ):

                                progress.progress(
                                    min(frame_number / total_frames, 1.0)
                                )

                                status.text(
                                    f"Processing frame {frame_number} "
                                    f"of {total_frames}"
                                )

                                preview.image(
                                    cv2.cvtColor(
                                        output_frame,
                                        cv2.COLOR_BGR2RGB
                                    ),
                                    caption=f"Processed frame: {frame_number}",
                                    use_container_width=True
                                )

                finally:
                    cap.release()
                    writer.release()

                progress.progress(1.0)
                status.success("Hybrid video detection completed!")

                st.divider()
                st.header("📊 Video Detection Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric("Frames Processed", frame_number)
                col2.metric("Frames Analysed", analysed_frames)
                col3.metric("Animal Detections", total_animal_detections)

                st.metric(
                    "Carnivore Sightings",
                    total_carnivore_sightings
                )

                st.metric(
                    "Maximum Carnivores in One Frame",
                    max_carnivores_in_frame
                )

                if total_carnivore_sightings > 0:
                    st.warning(
                        f"⚠️ Carnivores detected in "
                        f"{total_carnivore_sightings} "
                        "sampled-frame detections."
                    )
                else:
                    st.success(
                        "No carnivores detected in the analysed frames."
                    )

                if (
                    output_path.exists()
                    and output_path.stat().st_size > 0
                ):

                    st.header("🎥 Processed Video")

                    video_bytes = output_path.read_bytes()

                    st.video(video_bytes)

                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name="hybrid_animal_detection.mp4",
                        mime="video/mp4"
                    )

                else:
                    st.error(
                        "Processed video could not be generated."
                    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Wildlife Animal Detection System | "
    "Powered by YOLOv8 + Google SpeciesNet"
)
