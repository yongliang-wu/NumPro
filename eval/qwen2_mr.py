import cv2
import json
import os
import re
import shutil
import tempfile
from tqdm import tqdm
import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import argparse
import random

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("./Qwen2-VL-7B-Instruct")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def annotate_frame_with_pil(frame, text, position, font_size, color):
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
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame

def annotate_and_save_video(file_path, output_file_path, position, font_size, color, sampling="uniform", ratio=1.0):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {file_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
        
        if sampling == "uniform":
            step = total_frames // int(total_frames * ratio)
            frame_indices = list(range(0, total_frames, step))[:int(total_frames * ratio)]
        elif sampling == "random":
            frame_indices = random.sample(range(total_frames), int(total_frames * ratio))
        else:
            raise ValueError("Invalid sampling argument")


        for idx in frame_indices:
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame at index {idx} from video: {file_path}")
                continue

            frame = annotate_frame_with_pil(frame, str(idx), position, font_size, color)
            out.write(frame)

        cap.release()
        out.release()

    except Exception as e:
        print(f"Error processing video {file_path}: {e}")

def process_video_queries(model, processor, data_path, save_path, input_format, instruction, device="cuda", video_path=None, position='top_right', font_size=80, color='red', sampling="uniform", ratio=1.0):
    temp_dir = tempfile.mkdtemp()
    try:
        with open(data_path, 'r') as f:
            video_list = json.load(f)
        
        responses = []
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                responses = json.load(f)
        processed_ids = {response["id"] for response in responses}
        
        for video_info in tqdm(video_list):
            if video_info["id"] in processed_ids:
                continue
            
            video_file_path = os.path.join(video_path, video_info["video"])
            annotated_video_path = os.path.join(temp_dir, video_info['video'])
            annotate_and_save_video(
                video_file_path,
                annotated_video_path,
                position=position,
                font_size=font_size,
                color=color,
                sampling=sampling,
                ratio=ratio
            )

            input_context = instruction + input_format.format(video_info["query"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_video_path,
                            "fps": 1
                        },
                        {"type": "text", "text": input_context},
                    ],
                }
            ]

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
            inputs = inputs.to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response = {
                "id": video_info["id"],
                "response": output_text[0],
                "gt_start": video_info.get("start_time"),
                "gt_end": video_info.get("end_time"),
                "pred_start": 0,
                "pred_end": 0,
                "duration": video_info.get("duration")
            }
            
            match = re.search(r"from\s*(?:frame\s*)?(\d+)\s*to\s*(?:frame\s*)?(\d+)", response['response'], re.IGNORECASE)
            if match:
                response["pred_start"] = int(match.group(1))
                response["pred_end"] = int(match.group(2))
            
            responses.append(response)
            with open(save_path, 'w') as f:
                json.dump(responses, f, indent=4)

    finally:
        shutil.rmtree(temp_dir)  # 在所有视频处理完成后删除临时文件夹

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON file containing video queries.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output JSON file with responses.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--input_format", type=str, required=True, help="Input format string for the query.")
    parser.add_argument("--instruction", type=str, required=True, help="Instruction for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--position", type=str, default="top_right", help="Position of the frame number annotation.")
    parser.add_argument("--font_size", type=int, default=80, help="Font size of the frame number annotation.")
    parser.add_argument("--color", type=str, default="red", help="Color of the frame number annotation.")
    parser.add_argument("--sampling", type=str, default="uniform", help="Sampling method for the annotated method.")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of the video frames to be annotated.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    process_video_queries(model, processor, args.data_path, args.save_path, args.input_format, args.instruction, args.device, args.video_path, args.position, args.font_size, args.color, args.sampling, args.ratio)