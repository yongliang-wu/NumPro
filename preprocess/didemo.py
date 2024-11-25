import os
import cv2
import concurrent.futures
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_hash_file(hash_file):
    hash_dict = {}
    with open(hash_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            hash_dict[key] = value
    return hash_dict

hash_dict = load_hash_file('yfcc100m_hash.txt')

# Input and output folder paths
input_folders = ['data/didemo/train']
output_folder = 'data/didemo/videos_0.5FPS_number_red_40_br'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

def process_video(file_path, output_folder):
    """Process video: resize, extract frames and add numbers"""
    try:
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        id_before_hash = name.split('_')[1]
        id_hash = hash_dict.get(id_before_hash, None)
        
        if id_hash is None:
            return
        
        output_file_path = os.path.join(output_folder, f"{id_hash}.mp4")

        if os.path.exists(output_file_path):
            return

        # Read input video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {file_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for 0.5 FPS
        sample_interval = fps * 2  # Extract every 2 seconds

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 0.5, (336, 336))

        frame_count = 0
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                # Resize frame
                resized_frame = cv2.resize(frame, (336, 336))
                
                # Add frame number
                numbered_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(numbered_frame)
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)
                
                # Calculate text position
                width, height = numbered_frame.size
                text = str(frame_number)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = width - text_width
                y = height - text_height - text_height/3

                draw.text((x, y), text, font=font, fill='red')
                final_frame = cv2.cvtColor(np.array(numbered_frame), cv2.COLOR_RGB2BGR)
                
                out.write(final_frame)
                frame_number += 1

            frame_count += 1

        cap.release()
        out.release()

    except Exception as e:
        print(f"Error processing video {file_path}: {e}")

def process_videos():
    """Main processing pipeline"""
    video_files = []
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                video_files.append(os.path.join(root, file))

    print("Processing videos...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in video_files:
            futures.append(executor.submit(process_video, file, output_folder))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

if __name__ == "__main__":
    process_videos()
