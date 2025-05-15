import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from zipfile import ZipFile
from io import BytesIO
import random

# Function to compute blur score (higher = sharper)
def blur_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_sharpest_frames(video_path, min_chunk=4, max_chunk=7):
    cap = cv2.VideoCapture(video_path)
    frames = []
    sharpest_frames = []
    frame_count = 0
    chunk_size = random.randint(min_chunk, max_chunk)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if len(frames) == chunk_size:
            # Find sharpest frame in this chunk
            scores = [blur_score(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
            idx = int(np.argmax(scores))
            sharpest_frames.append(frames[idx])
            frames = []
            chunk_size = random.randint(min_chunk, max_chunk)
    # Handle last chunk
    if frames:
        scores = [blur_score(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
        idx = int(np.argmax(scores))
        sharpest_frames.append(frames[idx])
    cap.release()
    return sharpest_frames

def all_frames_to_zip(frames_dict):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for video_name, frames in frames_dict.items():
            for i, frame in enumerate(frames):
                _, img_encoded = cv2.imencode('.jpg', frame)
                # Use video name and frame index in filename
                safe_name = os.path.splitext(os.path.basename(video_name))[0]
                zip_file.writestr(f'{safe_name}_frame_{i+1}.jpg', img_encoded.tobytes())
    zip_buffer.seek(0)
    return zip_buffer

st.title('Super Shomaila Extractor (Batch)')
st.write('Drag and drop up to 200 videos. The app will extract the sharpest frame in each video and let you download all results as a zip.')

uploaded_files = st.file_uploader('Upload Videos', type=['mp4', 'avi', 'mov', 'mkv'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 200:
        st.error('Please upload no more than 200 videos at once.')
    else:
        frames_dict = {}
        progress = st.progress(0)
        status = st.empty()
        for idx, uploaded_file in enumerate(uploaded_files):
            video_name = uploaded_file.name
            status.info(f'Processing {video_name} ({idx+1}/{len(uploaded_files)})...')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_video_path = tmpfile.name
            sharpest_frames = extract_sharpest_frames(tmp_video_path, min_chunk=7, max_chunk=12)
            frames_dict[video_name] = sharpest_frames
            os.remove(tmp_video_path)
            progress.progress((idx+1)/len(uploaded_files))
        status.success(f'Processed {len(uploaded_files)} videos. Ready to download!')
        # Optionally display a few frames from each video
        for video_name, frames in frames_dict.items():
            st.write(f'**{video_name}**: {len(frames)} sharpest frames extracted')
            for i, frame in enumerate(frames[:2]):  # Show up to 2 frames per video
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f'{video_name} - Sharpest Frame {i+1}', use_column_width=True)
        # Download as zip
        zip_buffer = all_frames_to_zip(frames_dict)
        st.download_button('Download All as Zip', zip_buffer, file_name='all_sharpest_frames.zip', mime='application/zip') 
