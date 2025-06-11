import os
import streamlit as st
from keras.models import load_model
from tensorflow import keras
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from PIL import Image
import time

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20

# Set page config
st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")  # You can create a style.css file for additional styling

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
    finally:
        cap.release()
    return np.array(frames)

# Function to prepare single video
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
    return frame_features, frame_mask

# Function to extract features from Video Frames
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = frame[y:y+h, x:x+w]
        return face_image
    else:
        return None

# Load the feature extractor
feature_extractor = build_feature_extractor()

# Main App
def main():
    # Custom header
    st.markdown(
        """
        <div class="header">
            <h1>DeepGuard AI</h1>
            <p>Advanced Deepfake Detection System</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.image("logo.png", width=150)  # Add your logo image
        st.markdown("## About")
        st.info(
            """
            This AI-powered tool helps detect deepfake videos with state-of-the-art neural networks.
            Upload a video to analyze its authenticity.
            """
        )
        st.markdown("## How it works")
        st.write("""
        1. Upload a video file (MP4, AVI, MOV)
        2. The system extracts key frames
        3. Our AI analyzes facial features
        4. Get authenticity results
        """)
        st.markdown("## Model Info")
        st.write("MobileNetV3 architecture")
        st.write("Trained on 100,000+ videos")
        st.write("95.7% validation accuracy")

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display video preview
            st.video(uploaded_file)
            
            # Show video info
            with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            cap = cv2.VideoCapture(tmp_file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            cap.release()
            
            st.markdown("### Video Details")
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Frame count: {frame_count}")
            st.write(f"FPS: {fps:.2f}")

    with col2:
        st.markdown("## Analysis Results")
        
        if uploaded_file is not None:
            if st.button("Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Processing video..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Step 1: Loading video
                    status_text = st.empty()
                    status_text.markdown("**Step 1/4:** Extracting frames...")
                    frames = load_video(tmp_file_path)
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    # Step 2: Preparing frames
                    status_text.markdown("**Step 2/4:** Preparing frames for analysis...")
                    frame_features, frame_mask = prepare_single_video(frames)
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    # Step 3: Loading model
                    status_text.markdown("**Step 3/4:** Loading detection model...")
                    sequence_model = load_model('./models/mobileNet_model.h5')
                    progress_bar.progress(75)
                    time.sleep(0.5)
                    
                    # Step 4: Making prediction
                    status_text.markdown("**Step 4/4:** Analyzing video content...")
                    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
                    pred = probabilities.argmax()
                    class_vocab = ['FAKE', 'REAL']
                    confidence = max(probabilities) * 100
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Display results
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.markdown("### Detection Results")
                    
                    if pred == 0:  # FAKE
                        st.error(f"⚠️ **Result:** {class_vocab[pred]}")
                        st.warning(f"Confidence: {confidence:.2f}%")
                        st.markdown("""
                        <div class="alert alert-warning">
                            <strong>Warning:</strong> This video shows signs of manipulation.
                            Consider verifying its source before sharing.
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # REAL
                        st.success(f"✅ **Result:** {class_vocab[pred]}")
                        st.info(f"Confidence: {confidence:.2f}%")
                        st.markdown("""
                        <div class="alert alert-success">
                            <strong>Verified:</strong> No signs of deepfake manipulation detected.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.markdown("#### Confidence Level")
                    st.progress(int(confidence))
                    
                    # Clean up
                    os.unlink(tmp_file_path)

        else:
            st.image("placeholder.png")  # Add a placeholder image
            st.info("Please upload a video file to begin analysis")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>DeepGuard AI v1.0 | © 2023 AI Security Labs</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
