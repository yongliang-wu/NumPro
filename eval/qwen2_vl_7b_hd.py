import os
import argparse
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import cv2
import re
import shutil
import tempfile
import torch
import numpy as np
import random


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


# Initialize processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def annotate_frame_with_pil(frame, text, position, font_size, color):
    # Convert BGR to RGB format
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    
    width, height = frame.size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    margin = 0
    if position == "top_left":
        x, y = margin, margin
    elif position == "top_right":
        x, y = width - text_width - margin, margin
    elif position == "bottom_left":
        x, y = margin, height - text_height - margin
    elif position == "bottom_right":
        x, y = width - text_width - margin, height - text_height - margin
    elif position == "center":
        x, y = (width - text_width) // 2, (height - text_height) // 2
    else:
        raise ValueError("Invalid position argument")

    if position in ["bottom_left", "bottom_right"]:
        y -= text_height / 3

    draw.text((x, y), text, font=font, fill=color)
    # Convert back to BGR format
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame


def annotate_and_save_video(file_path, output_file_path, position, font_size, color, annotated=False):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {file_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define target size
        target_width = 336
        target_height = 336

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 1, (target_width, target_height))  # Set fps to 1 and use target size

        # Extract frames every 2 seconds (2 * fps frames)
        frame_indices = list(range(0, total_frames, int(2 * fps)))  

        for idx, frame_number in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame at index {frame_number} from video: {file_path}")
                continue

            # Resize frame before annotation
            frame = cv2.resize(frame, (target_width, target_height))

            if annotated:
                frame = annotate_frame_with_pil(frame, str(idx), position, font_size, color)
            out.write(frame)

        cap.release()
        out.release()

    except Exception as e:
        print(f"Error processing video {file_path}: {e}")

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', default='data/qvhighlights/videos', help='Directory containing video files.')
    parser.add_argument('--gt_file', default='data/highlight_val_release.jsonl', help='Path to the ground truth file.')
    parser.add_argument('--output_dir', default='results', help='Directory to save the model results JSON.')
    parser.add_argument('--output_name', default='qwen2_vl_7b_hd', help='Name of the file for storing results JSON.')
    parser.add_argument('--annotated', help='Whether to annotate the video.', default=True)
    parser.add_argument('--input_format', default="The red numbers on each frame represent the frame number. Please find the highlight contents in the video described by the query {}. Determine the highlight frames and its saliency score on a scale from 1 to 5. If the video content more related to the query, the saliency score should be higher. The output format should be like: 'The highlight frames are in the 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 frames. Their saliency scores are 1.3, 1.5, 2.6, 3.0, 2.9, 4.0, 3.7, 3.2, 2.1, 2.3'.",required=True)
    parser.add_argument('--device', help='Device to use for inference.', default="cuda")
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Create temporary directory for processed videos
    temp_dir = tempfile.mkdtemp()
    
    # Load the ground truth file
    gt_contents = []
    with open(args.gt_file) as file:
        for line in file:
            gt_contents.append(json.loads(line))

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    
    output_file_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as file:
            output_list = json.load(file)
    output_ids = [sample['qid'] for sample in output_list]

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['vid']
        sample_set = sample
        question = sample['query']
        if sample['qid'] in output_ids:
            continue
        
        question = args.input_format.format(question)

        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")

        annotated_video_path = os.path.join(temp_dir, f"{video_name}.mp4")
        annotate_and_save_video(
                video_path,
                annotated_video_path,
                position="bottom_right",
                font_size=40,
                color='red',
                annotated=args.annotated
        )

        # Process video if it exists
        if annotated_video_path is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_video_path,
                            "fps": 1
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            # Prepare inputs for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.cuda()

            # Run inference
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(output_text)
            
            sample_set['pred'] = output_text
            output_list.append(sample_set)

            # Save results to JSON file
            with open(output_file_path, 'w') as file:
                json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)