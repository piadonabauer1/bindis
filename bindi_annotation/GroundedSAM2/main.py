import argparse
import os
import glob
import cv2
import sys
import numpy as np
import json
import pycocotools.mask as mask_util
import re

from PIL import Image
from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from face_landmarks import FaceDetection, FaceMeshDetector
from grounded_sam2_hf_model import GroundedSAM2Pipeline
from grounded_sam2_florence2_image_demo import Florence2Pipeline


def merge_bboxes(detections, iou_threshold=0.6):
    merged_detections = []
    bboxes = [d['bbox'] for d in detections]
    masks = [d['mask'] for d in detections]
    confidences = [d['confidence'] for d in detections]
    class_names = [d['class_name'] for d in detections]
    scores = [d.get('score', 0) for d in detections]
    
    # Function to compute IOU (Intersection over Union)
    def compute_iou(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1b, y1b, x2b, y2b = bbox2
        inter_x1 = max(x1, x1b)
        inter_y1 = max(y1, y1b)
        inter_x2 = min(x2, x2b)
        inter_y2 = min(y2, y2b)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou
    
    while len(bboxes) > 0:
        bbox = bboxes.pop(0)
        mask = masks.pop(0)
        confidence = confidences.pop(0)
        class_name = class_names.pop(0)
        score = scores.pop(0)

        indices_to_merge = []
        for idx, bbox_2 in enumerate(bboxes):
            iou = compute_iou(bbox, bbox_2)
            if iou >= iou_threshold:
                indices_to_merge.append(idx)
        
        # Merge all the overlapping bounding boxes
        if indices_to_merge:
            merged_bbox = bbox
            merged_mask = mask
            merged_confidence = confidence
            merged_class_name = class_name
            merged_score = score
            
            # Merge other boxes, update the merged bbox and confidence
            for idx in indices_to_merge:
                merged_bbox = [
                    min(merged_bbox[0], bboxes[idx][0]),
                    min(merged_bbox[1], bboxes[idx][1]),
                    max(merged_bbox[2], bboxes[idx][2]),
                    max(merged_bbox[3], bboxes[idx][3]),
                ]
                merged_mask = np.maximum(merged_mask, masks[idx])
                merged_confidence = max(merged_confidence, confidences[idx])
                merged_class_name = class_name  # Assuming same class for now


            # Remove merged boxes from the original list
            bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx not in indices_to_merge]
            masks = [mask for idx, mask in enumerate(masks) if idx not in indices_to_merge]
            confidences = [confidence for idx, confidence in enumerate(confidences) if idx not in indices_to_merge]
            class_names = [class_name for idx, class_name in enumerate(class_names) if idx not in indices_to_merge]
            scores = [score for idx, score in enumerate(scores) if idx not in indices_to_merge]

            # Append the merged detection
            merged_detections.append({
                'bbox': merged_bbox,
                'confidence': merged_confidence,
                'mask': merged_mask,
                'class_name': merged_class_name,
                'score': merged_score,
            })
        else:
            merged_detections.append({
                'bbox': bbox,
                'confidence': confidence,
                'mask': mask,
                'class_name': class_name,
                'score': score,
            })
    
    return merged_detections



# Define left and right iris boundaries (lines passing through each iris)
def point_line_distance(point, line_point, line_direction):
    """Compute distance of point to a line given by point+direction vector"""
    point_vector = point - line_point
    return np.abs(np.cross(point_vector, line_direction))

# check if predicted bounding box has a max. size of distance between eyes (sometimes, faces are detected as bindis)
def size_check(bbox, iris_length):
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    return max(bbox_width, bbox_height) <= iris_length

# check if predicted bindi is on forehead
def above_line_check(bbox_center, left_iris, right_iris):
    eye_line_y = np.interp(bbox_center[0], [left_iris[0], right_iris[0]], [left_iris[1], right_iris[1]])
    return bbox_center[1] <= eye_line_y

def corridor_check(bbox_center, left_iris, right_iris, iris_length, perp_vector):
    left_distance = point_line_distance(bbox_center, left_iris, perp_vector)
    right_distance = point_line_distance(bbox_center, right_iris, perp_vector)
    return left_distance + right_distance <= iris_length

def filter_predictions(landmarks, predicted_objects, face_parts):
    try:
        left_iris = np.array(landmarks[face_parts[5]][0])  
        right_iris = np.array(landmarks[face_parts[6]][0])  

    # no landmarks detected -> prior face detection was wrong
    except KeyError:
        return None

    iris_vector = right_iris - left_iris
    iris_length = np.linalg.norm(iris_vector)
    perp_vector = np.array([-iris_vector[1], iris_vector[0]])  
    perp_vector = perp_vector / np.linalg.norm(perp_vector)  


    # predicted objects: [ 89.3244,  278.6940, 1710.3505,  851.5143] -> bounding box
    # left_iris e.g. [(2110, 1681), (2054, 1648), (2016, 1694), (2072, 1728)], -> landmarks
    filtered_predictions = []
    for obj in predicted_objects:
        bbox = obj['bbox']
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        if not size_check(bbox, iris_length):
            continue
        if not above_line_check(bbox_center, left_iris, right_iris):
            continue
        if not corridor_check(bbox_center, left_iris, right_iris, iris_length, perp_vector):
            continue

        filtered_predictions.append(obj)

    if filtered_predictions:
        filtered_predictions = merge_bboxes(filtered_predictions)
    return filtered_predictions


@staticmethod
def _mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# save cropped face as jpg and predictions within json
def save(image, predictions, base_name, save_imgs, save_jsons):

    img_height, img_width = image.shape[:2]

    results_json = {
        "image_path": base_name,
        "annotations": [
            {
                "class_name": "bindi", #p["class_name"],
                "bbox": p["bbox"],
                "segmentation": _mask_to_rle(p["mask"]),
                "confidence": p["confidence"],
                "score": p["score"],
            }
            for p in predictions
        ],
        "box_format": "xyxy",
        "img_width": img_width,
        "img_height": img_height,
    }

    base_name_json = re.sub(r'\.(jpg|png|jpeg)$', '.json', base_name)

    with open(os.path.join(save_jsons, base_name_json), "w") as f:
        json.dump(results_json, f, indent=4)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image in the correct format
    cv2.imwrite(os.path.join(save_imgs, base_name), image_bgr)


def convert_detections_to_dicts(detections):
    class_names = {0: 'dot', 1: 'bindi', 2: 'small_particle'}
    converted = []
    for i in range(len(detections.xyxy)):
        obj = {
            'class_name': "bindi", #class_names[detections.class_id[i]] if detections.class_id is not None else '',
            'bbox': detections.xyxy[i].tolist(),
            'mask': detections.mask[i].astype(float),  # optional: oder bool, je nach Bedarf
            'confidence': float(detections.confidence[i]) if detections.confidence is not None else 0.0,
            'score': [float(detections.confidence[i])] if detections.confidence is not None else []
        }
        converted.append(obj)
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Path to the input images")
    parser.add_argument("--prompt", type=str, help="Prompt for processing")
    parser.add_argument("--save_path", type=str, help="Path to save the output image")

    args = parser.parse_args()
    IMG_DIR = args.image_dir
    PROMPTS = args.prompt

    OUTPUT_DIR = args.save_path

    save_imgs = os.path.join(OUTPUT_DIR, "img")
    save_jsons = os.path.join(OUTPUT_DIR, "json")
    save_vis = os.path.join(OUTPUT_DIR, "vis")

    os.makedirs(save_imgs, exist_ok=True)
    os.makedirs(save_jsons, exist_ok=True)
    os.makedirs(save_vis, exist_ok=True)

    face_detector = FaceDetection()
    # Initialize the FaceMeshDetector with refined iris landmarks for better precision
    face_landmark_detector = FaceMeshDetector(refine_landmarks=True)

    object_detector_dino = GroundedSAM2Pipeline()
    object_detector_florence2 = Florence2Pipeline()

    face_parts = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "mouth_landmarks", "all_landmarks", "left_iris_landmarks",
              "right_iris_landmarks"]

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []

    # Gather all image files with the specified extensions
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMG_DIR, extension)))

    # Process each image file
    for image_path in image_files:
        image_name = os.path.basename(image_path)

        image_cv2 = cv2.imread(image_path)
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        detected_faces = face_detector.detect(image_pil) # output: image array

        for idx, face_array in enumerate(detected_faces):

            face_pil = Image.fromarray(face_array)
            _, landmarks = face_landmark_detector.findMeshInFace(face_array)
 
            try:
                predicted_objects_dino = object_detector_dino.predict(face_pil, prompt=PROMPTS, confidence_threshold=0.4)
            except ValueError as e:  # Replace with actual exceptions
                print(f"Error occurred while predicting: {e}")
                continue
            
            try:
                predicted_objects_florence2 = object_detector_florence2.predict(face_pil, PROMPTS.replace(".", " <and>"))
                predicted_objects_florence2 = convert_detections_to_dicts(predicted_objects_florence2)
                
            except ValueError as e:  # Replace with actual exceptions
                print(f"Error occurred while predicting: {e}")
                continue

            combined_detections = predicted_objects_dino + predicted_objects_florence2
            filtered_predicted_objects = filter_predictions(landmarks, combined_detections, face_parts)

            if not filtered_predicted_objects:
                continue

            image_name_with_idx = f"{os.path.splitext(image_name)[0]}_{idx}{os.path.splitext(image_name)[1]}"

            vis_path = OUTPUT_DIR + "/vis"  # direkt das Bild in 'vis/' speichern
            object_detector_dino._visualize_and_save(face_array, filtered_predicted_objects, vis_path, image_name_with_idx)

            #filtered_predicted_objects_json = json.dumps(filtered_predicted_objects, indent=2)
            #print(filtered_predicted_objects_json)

            save(face_array, filtered_predicted_objects, image_name_with_idx, save_imgs, save_jsons)


if __name__ == "__main__":
    main()