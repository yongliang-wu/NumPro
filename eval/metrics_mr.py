import json
import re


def cal_iou(A, B):
    """Calculate Intersection over Union (IoU) between two segments A and B."""
    max0 = max(A[0], B[0])  # Start of intersection
    min0 = min(A[0], B[0])  # Start of union
    max1 = max(A[1], B[1])  # End of union
    min1 = min(A[1], B[1])  # End of intersection
    intersection = max(min1 - max0, 0)
    union = max1 - min0
    return intersection / union

def calculate_iou(response, gt_start, gt_end):
    """Extract frame numbers from response text and calculate IoU with ground truth."""
    pattern = r"from\s*(?:frame\s*)?(\d+)\s*to\s*(?:frame\s*)?(\d+)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        # Convert to actual frame numbers based on video frame rate, here we assume the frame rate is 0.5 FPS
        start_frame = int(match.group(1)) * 2
        end_frame = int(match.group(2)) * 2
        return cal_iou([start_frame, end_frame], [gt_start, gt_end])
    return 0
    
def calculate_metrics(json_file):
    """Calculate evaluation metrics from predictions."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    num_samples = len(data)
    total_iou = sum(sample['iou'] for sample in data)

    # Calculate metrics
    miou = (total_iou / num_samples) * 100
    r_03 = sum(1 for sample in data if sample['iou'] >= 0.3) / num_samples * 100
    r_05 = sum(1 for sample in data if sample['iou'] >= 0.5) / num_samples * 100
    r_07 = sum(1 for sample in data if sample['iou'] >= 0.7) / num_samples * 100

    return round(miou, 2), round(r_03, 2), round(r_05, 2), round(r_07, 2)


# Input data path
data_path = "your_prediction_file.json"

# Load prediction data
data = json.load(open(data_path, "r"))

# Calculate IoU for each prediction
for item in data:
    response = item["response"]
    gt_start = item["gt_start"]
    gt_end = item["gt_end"]
    item["iou"] = calculate_iou(response, gt_start, gt_end)

# Save updated results
with open(data_path, "w") as outfile:
    json.dump(data, outfile, indent=4)

# Calculate and display final metrics
miou, r_03, r_05, r_07 = calculate_metrics(data_path)
print(f"R@0.3: {r_03:.2f}")
print(f"R@0.5: {r_05:.2f}")
print(f"R@0.7: {r_07:.2f}")
print(f"mIoU: {miou:.2f}")
