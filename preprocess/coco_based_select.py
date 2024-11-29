import json
import torch
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_high_similarity_pairs(similarity_file, threshold=0.8, num_pairs=10000):
    """Load image-text pairs with similarity above threshold"""
    with open(similarity_file, 'r') as f:
        data = json.load(f)
    
    filtered_pairs = [(item['image'], item['caption']) 
                     for item in data if item['similarity'] > threshold]
    
    return random.sample(filtered_pairs, min(num_pairs, len(filtered_pairs)))

def annotate_image(image_path, number, position, font_size, color):
    """Add number annotation to image"""
    # Read and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((336, 336))
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Determine position
    width, height = image.size
    margin = 5
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
    
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    return image

def evaluate_accuracies(model, processor, original_pairs, annotated_images, device):
    """Evaluate caption accuracy and number accuracy"""
    # Prepare all possible number texts
    number_texts = [str(i) for i in range(100)]
    
    # Get features for all original captions
    all_captions = [pair[1] for pair in original_pairs]
    caption_inputs = processor(text=all_captions, return_tensors="pt", padding=True)
    with torch.no_grad():
        caption_features = model.get_text_features(**{k: v.to(device) for k, v in caption_inputs.items()})
        caption_features = F.normalize(caption_features, dim=-1)
    
    # Get features for number texts
    number_inputs = processor(text=number_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        number_features = model.get_text_features(**{k: v.to(device) for k, v in number_inputs.items()})
        number_features = F.normalize(number_features, dim=-1)
    
    caption_correct = 0
    number_correct = 0
    total = len(annotated_images)
    
    for idx, (image, true_number) in enumerate(tqdm(annotated_images)):
        # Get image features
        image_inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**{k: v.to(device) for k, v in image_inputs.items()})
            image_features = F.normalize(image_features, dim=-1)
        
        # Calculate similarities with all captions
        similarities = (image_features @ caption_features.T).squeeze()
        pred_caption_idx = similarities.argmax().item()
        if pred_caption_idx == idx:  # If predicted caption index is correct
            caption_correct += 1
        
        # Calculate similarities with number texts
        number_similarities = (image_features @ number_features.T).squeeze()
        pred_number = number_similarities.argmax().item()
        if pred_number == true_number:  # If predicted number is correct
            number_correct += 1
    
    return caption_correct / total, number_correct / total

def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Configure parameters
    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
    font_sizes = list(range(20, 81, 15))  # [20, 35, 50, 65, 80]
    colors = ["red", "green", "blue", "black"]
    
    # Load image-text pairs
    pairs = load_high_similarity_pairs("path/to/similarity.json")
    
    results = []
    
    # Test each combination
    for position in positions:
        for font_size in font_sizes:
            for color in colors:
                print(f"\nTesting: position={position}, font_size={font_size}, color={color}")
                
                # Annotate images
                annotated_images = []
                for idx, (image_path, _) in enumerate(tqdm(pairs)):
                    number = idx % 100  # Numbers from 0-99
                    annotated = annotate_image(image_path, number, position, font_size, color)
                    annotated_images.append((annotated, number))
                
                # Evaluate accuracies
                caption_acc, number_acc = evaluate_accuracies(
                    model, processor, pairs, annotated_images, device)
                
                result = {
                    "position": position,
                    "font_size": font_size,
                    "color": color,
                    "caption_accuracy": caption_acc,
                    "number_accuracy": number_acc
                }
                results.append(result)
                
                # Save intermediate results
                with open("annotation_results.json", "w") as f:
                    json.dump(results, f, indent=4)
                
                print(f"Caption Accuracy: {caption_acc:.4f}")
                print(f"Number Accuracy: {number_acc:.4f}")

if __name__ == "__main__":
    main()