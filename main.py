import os
import streamlit as st
from keras.models import load_model
from tensorflow import keras
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20

# Function to load video and extract frames
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / SEQ_LENGTH), 1)
    frames = []
    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
        print("Completed extracting frames")
    finally:
        cap.release()
    return np.array(frames)

# Function to prepare single video
def prepare_single_video(frames):
    print("Preparing Frames")
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
    print("Completed preparing frames")
    return frame_features, frame_mask

# Function to build feature extractor
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Function to crop face center
def crop_face_center(frame):
    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Use the first detected face
        (x, y, w, h) = faces[0]
        face_image = frame[y:y+h, x:x+w]
        return face_image
    else:
        return None

# Load the feature extractor
feature_extractor = build_feature_extractor()

# Streamlit app
def main():
    st.title("DeepFake Detection")
    st.write("Upload a video to classify it as FAKE or REAL.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Display the video
        st.video(uploaded_file)

        if st.button("Predict"):
            # Use a temporary file to process the video
            with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load the sequence model
            sequence_model = load_model('./models/mobileNet_model.h5')
            class_vocab = ['FAKE', 'REAL']

            # Load and process the video
            frames = load_video(tmp_file_path)
            frame_features, frame_mask = prepare_single_video(frames)

            # Make a prediction
            probabilities = sequence_model.predict([frame_features, frame_mask])[0]
            pred = probabilities.argmax()
            st.write(f"Prediction: {class_vocab[pred]}")

            # Clean up the temporary file
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
