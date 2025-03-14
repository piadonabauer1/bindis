import argparse
import os
import glob
import cv2
import numpy as np
import json
import re
import copy

from PIL import Image
from scipy.spatial import ConvexHull

from face_landmarks import FaceDetection, FaceMeshDetector
from GroundedSAM2.grounded_sam2_hf_model import GroundedSAM2Pipeline
from GroundedSAM2.grounded_sam2_florence2_image_demo import Florence2Pipeline
from AcneDetection.acne_detection import AcneDetection


# Computes intersection over union (IoU) for two bounding boxes
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

# Calculates how much two bounding boxes overlap
def intersection_area(box1, box2):
    x1, y1, x2, y2 = box1
    xb1, yb1, xb2, yb2 = box2

    ix1 = max(x1, xb1)
    iy1 = max(y1, yb1)
    ix2 = min(x2, xb2)
    iy2 = min(y2, yb2)

    if ix1 < ix2 and iy1 < iy2:
        return (ix2 - ix1) * (iy2 - iy1) 
    else:
        return 0  

# Checks if two bounding boxes overlap at least 0.8
def is_inside(inner, outer, ratio=0.8):
    box1_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    overlap_area = intersection_area(inner, outer)

    overlap_ratio = overlap_area / box1_area
    return overlap_ratio >= ratio

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

# Merges bounding boxes based on defined rules
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
                # Classes that should be merged
                merge_classes = [ # prompt may not include all of those!
                    (current['class_name'], other['class_name']) in [
                        ('bindi', 'piercing'), ('piercing', 'bindi'),
                        ('piercing', 'acne'), ('acne', 'piercing'),
                        ('bindi', 'acne'), ('acne', 'bindi')
                    ]
                ]

                if current['class_name'] == other['class_name'] or any(merge_classes):
                    iou = compute_iou(current_bbox, other['bbox'])
                    if iou >= iou_threshold or is_inside(current_bbox, other['bbox']) or is_inside(other['bbox'], current_bbox):
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

    # Remove acne label if bindi was labaled as acne
    final_detections = []
    for detection in detections:
        if detection['class_name'] == 'acne':
            bindi_detections = [d for d in detections if d['class_name'] in ('bindi', 'piercing')]
            if any(compute_iou(detection['bbox'], b['bbox']) > 0 for b in bindi_detections):
                continue
        final_detections.append(detection)
    
    return final_detections



# Position checks
def point_line_distance(point, line_point, line_direction):
    point_vector = point - line_point
    return np.abs(np.cross(point_vector, line_direction))

# Filters out bounding boxes bigger than distance of both eyes' iris
def size_check(bbox, iris_length):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, y2 - y1) <= 0.75 * iris_length

# Checks if bindi detection is above eye level
def above_line_check(bbox_center, left_iris, right_iris):
    eye_line_y = np.interp(bbox_center[0], [left_iris[0], right_iris[0]], [left_iris[1], right_iris[1]])
    return bbox_center[1] <= eye_line_y

# Checks if bindi detection is between x-coordinates of both eyes
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

# Checks if a landmark (x,y) is in a bounding box (x1,y1,x2,y2) 
def point_in_bbox(landmark, bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 <= x <= x2 and y1 <= y <= y2) for x, y in landmark]

def reorder_points(landmarks):
    hull = ConvexHull(landmarks)
    landmarks_sorted = landmarks[hull.vertices]
    return landmarks_sorted

# Checks if bounding box is inside face (via facial landmarks)
def bbox_in_landmarks(landmarks, bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = np.array([center_x, center_y])
    landmarks = np.array(landmarks, dtype=np.int32)
    landmarks_sorted = reorder_points(landmarks)
    result = cv2.pointPolygonTest(landmarks_sorted, tuple(center_point), False)
    return result >= 0 

# Computes potential overlap between bbox and philtrum (most likely a piercing being detected as acne)
def in_between_nose_mouth(nose, mouth, bbox):
    nose_bottom = max(nose, key=lambda p: p[1])
    mouth_top = min(mouth, key=lambda p: p[1])

    region_top = nose_bottom[1]
    region_bottom = mouth_top[1]

    region_left = max(min(nose, key=lambda p: p[0])[0], min(mouth, key=lambda p: p[0])[0])
    region_right = min(max(nose, key=lambda p: p[0])[0], max(mouth, key=lambda p: p[0])[0])

    bbox_top = bbox[1]  
    bbox_bottom = bbox[3] 
    bbox_left = bbox[0] 
    bbox_right = bbox[2]

    overlap_top = max(bbox_top, region_top)
    overlap_bottom = min(bbox_bottom, region_bottom)
    overlap_height = max(0, overlap_bottom - overlap_top) 

    bbox_height = bbox_bottom - bbox_top

    overlap_left = max(bbox_left, region_left)
    overlap_right = min(bbox_right, region_right)
    overlap_width = max(0, overlap_right - overlap_left) 

    bbox_width = bbox_right - bbox_left

    overlap_ratio_bbox_height = overlap_height / bbox_height
    overlap_ratio_bbox_width = overlap_width / bbox_width

    if overlap_ratio_bbox_height >= 0.5 and overlap_ratio_bbox_width >= 0.5:
        return True
    return False

# Filters predictions
def filter_predictions(landmarks, predicted_objects, face_parts):
    try:
        left_eye = np.array(landmarks[face_parts[0]][0])
        right_eye = np.array(landmarks[face_parts[1]][0])
        nose = np.array(landmarks[face_parts[2]])
        all_landmarks = np.array(landmarks[face_parts[4]])
        left_iris = np.array(landmarks[face_parts[5]])
        right_iris = np.array(landmarks[face_parts[6]])
        mouth = np.array(landmarks[face_parts[3]])
    except KeyError:
        return None

    iris_vector = right_eye - left_eye
    iris_length = np.linalg.norm(iris_vector)
    perp_vector = np.array([-iris_vector[1], iris_vector[0]])
    perp_vector = perp_vector.astype(float) / np.linalg.norm(perp_vector)

    filtered_predictions = []

    for obj in predicted_objects:
        bbox = obj['bbox']
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        # If potential detected bindi is not on forehead, then most likely not a bindi -> remove
        if obj['class_name'] == 'bindi':
            if not size_check(bbox, iris_length):
                continue
            if not above_line_check(bbox_center, left_eye, right_eye) or not corridor_check(bbox, left_eye, right_eye, iris_length, perp_vector):
                continue
        
        # Prevents detections of eyes or piercings on nose
        if any(point_in_bbox(left_iris, bbox)) or any(point_in_bbox(right_iris, bbox)) or any(point_in_bbox(nose, bbox)) :
            #print(f"{obj['class_name']} is on landmark and was removed.")
            continue

        # Keep only predictions on face
        if not bbox_in_landmarks(all_landmarks, bbox):
            #print(f"{obj['class_name']} is outside of all landmarks.")
            continue

        # Prevents detections of septa
        if obj['class_name'] == 'acne' and in_between_nose_mouth(nose, mouth, bbox):
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
            'confidence': score, 
        }
        for label, bbox, score in zip(acne_pred['labels'], acne_pred['bboxes'], acne_pred['scores'])
    ]
    return detections


# Saves cropped face and json
def save(image, predictions, base_name, save_imgs, save_jsons):

    img_height, img_width = image.shape[:2]

    results_json = {
        "image_path": base_name,
        "annotations": [
            {
                "class_name": "acne" if p["class_name"] == "acne" else "bindi",
                "bbox": p["bbox"],
                "confidence": p["confidence"],
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
    cv2.imwrite(os.path.join(save_imgs, base_name), image_bgr)

# Adjusts Forence2 predictions to Grounding DINO format
def convert_detections_to_dicts(detections):
    converted = []
    for i in range(len(detections.xyxy)):
        obj = {
            'class_name': "bindi",
            'bbox': detections.xyxy[i].tolist(),
            'confidence': float(detections.confidence[i]) if detections.confidence is not None else 0.0,
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

    # Initialize models
    acnedet = AcneDetection()
    face_detector = FaceDetection()
    face_landmark_detector = FaceMeshDetector(refine_landmarks=True)
    object_detector_dino = GroundedSAM2Pipeline()
    object_detector_florence2 = Florence2Pipeline()

    face_parts = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "mouth_landmarks", "all_landmarks", "left_iris_landmarks",
              "right_iris_landmarks"]
    face_parts_of_interest = ["left_eye_landmarks", "right_eye_landmarks", "nose_landmarks",
              "left_iris_landmarks", "right_iris_landmarks", "mouth_landmarks"]
    
    # Only for visualizations of facial landmarks
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMG_DIR, extension)))

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        #print(image_name)

        image_cv2 = cv2.imread(image_path)
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        detected_faces = face_detector.detect(image_pil) # Output: image array

        for idx, face_array in enumerate(detected_faces):
            image_name_with_idx = f"{os.path.splitext(image_name)[0]}_{idx}{os.path.splitext(image_name)[1]}"

            face_pil = Image.fromarray(face_array)
            image, landmarks = face_landmark_detector.findMeshInFace(face_array)
            image_lm = copy.deepcopy(image)

            # Visualize facial landmarks of interest
            for idx, face_part in enumerate(face_parts_of_interest):
                try:
                    color = colors[idx % len(colors)] 
                    for landmark in landmarks[face_part]:
                        cv2.circle(image_lm, (landmark[0], landmark[1]), 3, color, -1)
                except KeyError:
                    pass

            image_lm = cv2.cvtColor(image_lm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_lm, image_name_with_idx), image_lm)

            no_bindi_found = False
            combined_detections = []

            # Predict with Grounding DINO
            try:
                predicted_objects_dino = object_detector_dino.predict(face_pil, prompt=PROMPTS, confidence_threshold=0.2)
                combined_detections.extend(predicted_objects_dino)
            except Exception as e:
                no_bindi_found = True
                #print(f"Error occurred while predicting with DINO: {e}")

            # Predict with Florence 2
            try:
                predicted_objects_florence2 = object_detector_florence2.predict(face_pil, PROMPTS.replace(".", " <and>"))
                predicted_objects_florence2 = convert_detections_to_dicts(predicted_objects_florence2)
                combined_detections.extend(predicted_objects_florence2)  
            except Exception as e:
                #print(f"Error occurred while predicting with Florence2: {e}")
                if no_bindi_found and not combined_detections:
                    continue
            
            # Predict acne with acne net
            acne_pred, _ = acnedet(img=face_array, threshold=0.19)
            predicted_acne = format_acne_det(acne_pred)
            combined_detections.extend(predicted_acne)

            # Filter all predictions
            filtered_predicted_objects = filter_predictions(landmarks, combined_detections, face_parts)
            #print(filtered_predicted_objects)

            # Continue if all predictions were filtered
            if not filtered_predicted_objects:
                continue

            object_detector_dino._visualize_and_save(face_array, filtered_predicted_objects, save_vis, image_name_with_idx)

            resized_image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
            save(resized_image, filtered_predicted_objects, image_name_with_idx, save_imgs, save_jsons)


if __name__ == "__main__":
    main()