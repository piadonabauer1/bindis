import os
import cv2
import json
import numpy as np
import torch
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP

import os

os.chdir('/home/coder/bindis/bindi_annotation/GroundedSAM2')


class GroundedSAM2Pipeline:
    def __init__(self, 
                 grounding_model="IDEA-Research/grounding-dino-tiny", 
                 sam2_checkpoint="/home/coder/bindis/bindi_annotation/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 device=None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use bfloat16 if possible
        self.autocast = torch.autocast(device_type=self.device, dtype=torch.bfloat16)

        if self.device.startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load Grounding DINO
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(self.device)

        # Load SAM2
        self.sam2_predictor = self._load_sam2(sam2_checkpoint, sam2_model_config)

        # Load YOLO face detection (optional, in case you need to crop faces — remove if unnecessary)
        self.yolo_model = YOLO(hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"))

        # Initialize color palettes for visualization
        self.color_palette = ColorPalette.from_hex(CUSTOM_COLOR_MAP)

    def _load_sam2(self, checkpoint, config):
        model = build_sam2(config, checkpoint, device=self.device)
        return SAM2ImagePredictor(model)

    def predict(self, img, prompt, confidence_threshold=0.4, save_dir=None):
        image = img.convert("RGB")
        self.sam2_predictor.set_image(np.array(image))

        # Run Grounding DINO
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids, 
            box_threshold=confidence_threshold,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        predictions = []
        for class_name, box, mask, confidence, score in zip(class_names, boxes, masks, confidences, scores.tolist()):
            predictions.append({
                "class_name": "bindi",#class_name,
                "bbox": box.tolist(),
                #"mask": mask,
                "confidence": confidence,
                #"score": score
            })

        if save_dir:
            self._visualize_and_save(image, predictions, save_dir)

        return predictions

    def _visualize_and_save(self, image, predictions, save_dir, img_name):
        os.makedirs(save_dir, exist_ok=True)

        # Prepare for supervision visualization
        xyxy_boxes = np.array([p["bbox"] for p in predictions])
        
        # Ensure that masks are of type bool or int
        #masks = np.array([p["mask"].astype(bool) for p in predictions])  # Ensure mask is of type bool or int
        class_ids = np.arange(len(predictions))

        #detections = sv.Detections(xyxy=xyxy_boxes, mask=masks, class_id=class_ids)
        detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)

        labels = [
            f"{p['class_name']} {p['confidence']:.2f}" for p in predictions
        ]

        annotated_image = np.array(image).copy()

        # Box Annotator
        box_annotator = sv.BoxAnnotator(color=self.color_palette)
        annotated_image = box_annotator.annotate(annotated_image, detections)

        # Label Annotator
        label_annotator = sv.LabelAnnotator(color=self.color_palette)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)

        # Mask Annotator
        #mask_annotator = sv.MaskAnnotator(color=self.color_palette)
        #annotated_image = mask_annotator.annotate(annotated_image, detections)

        # Save images
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name), annotated_image)

        """
        # Optionally, also save JSON results
        results_json = {
            "image_path": image.filename,
            "annotations": [
                {
                    "class_name": p["class_name"],
                    "bbox": p["bbox"],
                    "segmentation": self._mask_to_rle(p["mask"]),
                    "confidence": p["confidence"],
                    "score": p["score"],
                }
                for p in predictions
            ],
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }
        with open(os.path.join(save_dir, "grounded_sam2_predictions.json"), "w") as f:
            json.dump(results_json, f, indent=4)
        """

    @staticmethod
    def _mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
