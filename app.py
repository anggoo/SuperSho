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
        # Group sharpest frame images by rules and create four zip files
        import re
        from io import BytesIO
        
        # Helper to determine zip group
        def get_zip_group(filename):
            match = re.search(r'region_(\d+).*frame_(\d+)', filename, re.IGNORECASE)
            if not match:
                return None
            region = int(match.group(1))
            frame = int(match.group(2))
            if region == 2 and 1 <= frame <= 16:
                return 'upper right'
            elif region == 2 and frame >= 17:
                return 'upper left'
            elif region == 0 and frame >= 17:
                return 'lower right'
            elif region == 0 and 1 <= frame <= 16:
                return 'lower left'
            return None

        # Collect all sharpest frame images into a dict by group
        grouped_images = {'upper right': [], 'upper left': [], 'lower right': [], 'lower left': []}
        for video_name, frames in frames_dict.items():
            safe_name = os.path.splitext(os.path.basename(video_name))[0]
            for i, frame in enumerate(frames):
                img_name = f'{safe_name}_frame_{i+1}.jpg'
                group = get_zip_group(img_name)
                if group:
                    grouped_images[group].append((img_name, frame))

        # Function to create zip buffer for a group
        def make_zip_buffer(image_list):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                for img_name, frame in image_list:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    zip_file.writestr(img_name, img_encoded.tobytes())
            zip_buffer.seek(0)
            return zip_buffer

        # Create a master zip file containing categorized zip files (if they have content)
        master_zip_buffer = BytesIO()
        any_group_has_images = False
        with ZipFile(master_zip_buffer, 'w') as master_zipf:
            for group_name, image_list in grouped_images.items():
                if image_list:  # Only include category zip if it has images
                    any_group_has_images = True
                    category_zip_filename = f'{group_name.replace(" ", "_")}.zip'
                    # Use the existing make_zip_buffer to create the content for this category's zip
                    category_zip_content_buffer = make_zip_buffer(image_list) 
                    master_zipf.writestr(category_zip_filename, category_zip_content_buffer.getvalue())
        
        master_zip_buffer.seek(0)

        if any_group_has_images:
            st.download_button(
                label='Download All Grouped Zips',
                data=master_zip_buffer, 
                file_name='grouped_frames_archive.zip', 
                mime='application/zip'
            )
        else:
            st.info('No images were categorized into groups, so no archive was created.')
 
