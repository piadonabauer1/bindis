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
from GroundedSAM2.grounded_sam2_hf_model import GroundedSAM2Pipeline
from GroundedSAM2.grounded_sam2_florence2_image_demo import Florence2Pipeline
from AcneDetection.acne_detection import AcneDetection


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

def is_inside(inner, outer):
    x1, y1, x2, y2 = inner
    xb1, yb1, xb2, yb2 = outer
    return xb1 <= x1 <= xb2 and xb1 <= x2 <= xb2 and yb1 <= y1 <= yb2 and yb1 <= y2 <= yb2

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def merge_bboxes(detections, iou_threshold=0.6):
    changed = True
    while changed:
        changed = False
        merged_detections = []
        while detections:
            current = detections.pop(0)
            current_bbox = current['bbox']
            indices_to_merge = []
            for idx, other in enumerate(detections):
                # Gleiche Klassen zusammenführen
                if current['class_name'] == other['class_name']:
                    iou = compute_iou(current_bbox, other['bbox'])
                    if iou >= iou_threshold or is_inside(current_bbox, other['bbox']) or is_inside(other['bbox'], current_bbox):
                        print(f"IOU: {iou} - Merging {current_bbox} and {other['bbox']}")
                        indices_to_merge.append(idx)

                # Spezielle Regeln für verschiedene Klassen
                elif (current['class_name'] == 'bindi' and other['class_name'] == 'piercing') or \
                     (current['class_name'] == 'piercing' and other['class_name'] == 'bindi'):
                    iou = compute_iou(current_bbox, other['bbox'])
                    if iou >= iou_threshold or is_inside(current_bbox, other['bbox']) or is_inside(other['bbox'], current_bbox):
                        print(f"IOU: {iou} - Merging bindi and piercing, keeping bindi")
                        indices_to_merge.append(idx)

                elif (current['class_name'] == 'piercing' and other['class_name'] == 'acne') or \
                     (current['class_name'] == 'acne' and other['class_name'] == 'piercing'):
                    iou = compute_iou(current_bbox, other['bbox'])
                    if iou >= iou_threshold or is_inside(current_bbox, other['bbox']) or is_inside(other['bbox'], current_bbox):
                        print(f"IOU: {iou} - Merging piercing and acne, keeping piercing")
                        indices_to_merge.append(idx)

                elif (current['class_name'] == 'bindi' and other['class_name'] == 'acne') or \
                     (current['class_name'] == 'acne' and other['class_name'] == 'bindi'):
                    iou = compute_iou(current_bbox, other['bbox'])
                    if iou >= iou_threshold or is_inside(current_bbox, other['bbox']) or is_inside(other['bbox'], current_bbox):
                        print(f"IOU: {iou} - Merging bindi and acne, keeping bindi")
                        indices_to_merge.append(idx)

            # Merge the objects and decide which box remains based on area
            for idx in reversed(indices_to_merge):
                match = detections.pop(idx)

                if bbox_area(current_bbox) >= bbox_area(match['bbox']):
                    larger_bbox = current_bbox
                else:
                    larger_bbox = match['bbox']

                current_bbox = larger_bbox
                current['confidence'] = max(current['confidence'], match['confidence'])
                changed = True

            merged_detections.append({
                'bbox': current_bbox,
                'confidence': current['confidence'],
                'class_name': current['class_name']
            })
        
        detections = merged_detections

    # remove acne label if bindi was detected
    final_detections = []
    for detection in detections:
        if detection['class_name'] == 'acne':
            bindi_detections = [d for d in detections if d['class_name'] in ('bindi', 'piercing')]
            if any(compute_iou(detection['bbox'], b['bbox']) > 0 for b in bindi_detections):
                print(f"Removing acne: {detection['bbox']} due to overlap with bindi")
                continue
        final_detections.append(detection)
    
    return final_detections



# Utility functions for calculating position checks
def point_line_distance(point, line_point, line_direction):
    point_vector = point - line_point
    return np.abs(np.cross(point_vector, line_direction))

def size_check(bbox, iris_length):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, y2 - y1) <= 0.8 * iris_length

def above_line_check(bbox_center, left_iris, right_iris):
    eye_line_y = np.interp(bbox_center[0], [left_iris[0], right_iris[0]], [left_iris[1], right_iris[1]])
    return bbox_center[1] <= eye_line_y

"""
def corridor_check(bbox_center, left_iris, right_iris, iris_length, perp_vector):
    left_distance = point_line_distance(bbox_center, left_iris, perp_vector)
    right_distance = point_line_distance(bbox_center, right_iris, perp_vector)
    return left_distance + right_distance <= iris_length
"""

def corridor_check(bbox, left_iris, right_iris, iris_length, perp_vector):
    x1, y1, x2, y2 = bbox
    bbox_points = np.array([
        [x1, y1],  
        [x2, y1],
        [x1, y2],
        [x2, y2],
        [(x1 + x2) / 2, y1],
        [(x1 + x2) / 2, y2],
        [x1, (y1 + y2) / 2],
        [x2, (y1 + y2) / 2],
        [(x1 + x2) / 2, (y1 + y2) / 2]
    ])

    for point in bbox_points:
        left_distance = point_line_distance(point, left_iris, perp_vector)
        right_distance = point_line_distance(point, right_iris, perp_vector)
        if left_distance + right_distance <= iris_length:
            return True 

    return False 

def point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def filter_predictions(landmarks, predicted_objects, face_parts):
    try:
        left_iris = np.array(landmarks[face_parts[0]][0])
        right_iris = np.array(landmarks[face_parts[1]][0])
        nose = np.array(landmarks[face_parts[2]][0])
    except KeyError:
        return None

    iris_vector = right_iris - left_iris
    iris_length = np.linalg.norm(iris_vector)
    perp_vector = np.array([-iris_vector[1], iris_vector[0]])
    perp_vector = perp_vector.astype(float) / np.linalg.norm(perp_vector)

    filtered_predictions = []

    for obj in predicted_objects:
        bbox = obj['bbox']
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        if obj['class_name'] == 'bindi':
            """
            if not (size_check(bbox, iris_length) and
                    above_line_check(bbox_center, left_iris, right_iris) and
                    corridor_check(bbox_center, left_iris, right_iris, iris_length, perp_vector)):
                continue"
            """
            if not size_check(bbox, iris_length):
                continue
            if not above_line_check(bbox_center, left_iris, right_iris) or not corridor_check(bbox, left_iris, right_iris, iris_length, perp_vector):
                obj['class_name'] = 'piercing'

        if obj['class_name'] == 'acne':
            # Filter out if any landmark is inside an 'acne' bbox
            if any(point_in_bbox(landmark, bbox) for landmark in [left_iris, right_iris, nose]):
                continue

        filtered_predictions.append(obj)

    if filtered_predictions:
        return merge_bboxes(filtered_predictions)
    return filtered_predictions





def format_acne_det(acne_pred):
    detections = [
        {
            'class_name': "acne",
            'bbox': bbox,
            #'mask': np.zeros((10, 10), dtype=np.float32), 
            'confidence': score, 
            #'score': [score] 
        }
        for label, bbox, score in zip(acne_pred['labels'], acne_pred['bboxes'], acne_pred['scores'])
    ]
    
    return detections


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
                #"segmentation": _mask_to_rle(p["mask"]),
                "confidence": p["confidence"],
                #"score": p["score"],
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
    converted = []
    for i in range(len(detections.xyxy)):
        obj = {
            'class_name': "bindi", #class_names[detections.class_id[i]] if detections.class_id is not None else '',
            'bbox': detections.xyxy[i].tolist(),
            #'mask': detections.mask[i].astype(float),  # optional: oder bool, je nach Bedarf
            'confidence': float(detections.confidence[i]) if detections.confidence is not None else 0.0,
            #'score': [float(detections.confidence[i])] if detections.confidence is not None else []
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
    save_lm = os.path.join(OUTPUT_DIR, "lm")

    os.makedirs(save_imgs, exist_ok=True)
    os.makedirs(save_jsons, exist_ok=True)
    os.makedirs(save_vis, exist_ok=True)
    os.makedirs(save_lm, exist_ok=True)

    acnedet = AcneDetection()

    face_detector = FaceDetection()
    # Initialize the FaceMeshDetector with refined iris landmarks for better precision
    face_landmark_detector = FaceMeshDetector(refine_landmarks=True)

    object_detector_dino = GroundedSAM2Pipeline()
    object_detector_florence2 = Florence2Pipeline()

    face_parts = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "mouth_landmarks", "all_landmarks", "left_iris_landmarks",
              "right_iris_landmarks"]
    face_parts_of_interest = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "left_iris_landmarks", "right_iris_landmarks"]
    
    colors = [
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255)  # Yellow
    ]

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
            image_name_with_idx = f"{os.path.splitext(image_name)[0]}_{idx}{os.path.splitext(image_name)[1]}"

            face_pil = Image.fromarray(face_array)
            image_lm, landmarks = face_landmark_detector.findMeshInFace(face_array)

            for idx, face_part in enumerate(face_parts_of_interest):
                try:
                    color = colors[idx % len(colors)]  # Cycle through colors if there are more face parts than colors
                    for landmark in landmarks[face_part]:
                        cv2.circle(image_lm, (landmark[0], landmark[1]), 3, color, -1)
                except KeyError:
                    pass

            image_lm = cv2.cvtColor(image_lm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_lm, image_name_with_idx), image_lm)
 
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

            acne_pred, _ = acnedet(img=face_array, threshold=0.2)
            predicted_acne = format_acne_det(acne_pred)

            combined_detections = predicted_objects_dino + predicted_objects_florence2 + predicted_acne
            filtered_predicted_objects = filter_predictions(landmarks, combined_detections, face_parts)
            print(filtered_predicted_objects)

            if not filtered_predicted_objects:
                continue

            object_detector_dino._visualize_and_save(face_array, filtered_predicted_objects, save_vis, image_name_with_idx)

            #filtered_predicted_objects_json = json.dumps(filtered_predicted_objects, indent=2)
            #print(filtered_predicted_objects_json)
            save(face_array, filtered_predicted_objects, image_name_with_idx, save_imgs, save_jsons)


if __name__ == "__main__":
    main()