import os
import streamlit as st
from keras.models import load_model
from tensorflow import keras
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20
CLASS_VOCAB = ['FAKE', 'REAL']

# --- Build Feature Extractor (cached to avoid reload) ---
@st.cache_resource
def build_feature_extractor():
    base_model = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = base_model(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# --- Helper Functions ---
def crop_face_center(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]
    return None

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(total_frames / SEQ_LENGTH), 1)
    frames = []
    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1
    return frame_features, frame_mask

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="DeepFake Detector", layout="centered")
    st.title("🧠 DeepFake Video Detection")
    st.markdown("Upload a video, and the model will predict whether it's **REAL** or **FAKE**.")

    with st.expander("📌 Instructions", expanded=False):
        st.markdown("""
        - Upload a short video in `.mp4`, `.avi`, or `.mov` format.
        - Ensure the video contains a clear face for best results.
        - Click on **Predict** to see the result.
        """)

    uploaded_file = st.file_uploader("📤 Upload Video File", type=["mp4", "avi", "mov"])

    if uploaded_file:
        st.video(uploaded_file)
        if st.button("🔍 Predict"):
            with st.spinner("Processing and predicting... ⏳"):
                with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                try:
                    sequence_model = load_model('./models/mobileNet_model.h5')
                    frames = load_video(tmp_file_path)
                    if len(frames) == 0:
                        st.error("No face was detected in the video. Try another video.")
                        return
                    frame_features, frame_mask = prepare_single_video(frames)
                    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
                    pred_index = probabilities.argmax()
                    confidence = probabilities[pred_index] * 100

                    # Show prediction with confidence
                    st.success("✅ Prediction Completed!")
                    st.markdown(f"### 🧾 Result: **{CLASS_VOCAB[pred_index]}**")
                    st.progress(int(confidence))
                    st.metric(label="Confidence", value=f"{confidence:.2f} %")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")

                finally:
                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
