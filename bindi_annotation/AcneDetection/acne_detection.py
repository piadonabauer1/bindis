from mmdet.apis import DetInferencer
import os

os.chdir('/home/coder/bindis/bindi_annotation/AcneDetection')

class AcneDetection:
    def __init__(self):
        self.weights = "epoch_16.pth"
        self.model_path = "CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"
        # init mmdetection inferencer
        self.inferencer = DetInferencer(model=self.model_path, weights=self.weights)

    def __call__(
        self,
        img,
        threshold,
        labels_to_exclude=[],
        out_dir="",  # default: no output directory specified
    ):
        # print(f"Path of the current file:{os.path.abspath(__file__)}")
        # perform inference

        result = self.inferencer(
            inputs=img,
            pred_score_thr=threshold,
        )

        predictions = result["predictions"][0]
        visualization = result["visualization"]

        # filter predictions based on threshold and labels to exclude
        filtered_predictions = {"labels": [], "scores": [], "bboxes": []}

        for label, score, bbox in zip(predictions["labels"], predictions["scores"], predictions["bboxes"]):
            if (not labels_to_exclude or label not in labels_to_exclude) and score >= threshold:
                filtered_predictions["labels"].append(label)
                filtered_predictions["scores"].append(score)
                filtered_predictions["bboxes"].append(bbox)

        return filtered_predictions, visualization