# Bindi Detection Dataset Creation

This repository provides the process for creating a new Bindi detection dataset using open-world object detection models. The resulting dataset can be found on [Google Drive](https://drive.google.com/drive/folders/10go6Vu6AM4S4tfWyE02AJN4Pg6E6q24q?usp=sharing).

## Dataset

Approximately 2500 images were sampled from [Pexels](https://www.pexels.com/de-de/suche/bindis/) using an image scraping library found on GitHub, called [pexels-image-downloader](https://github.com/AguilarLagunasArturo/pexels-image-downloader). Registration on Pexels is required to obtain an API key for image scraping.

## Models, Frameworks, and Libraries

- **[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)**: Used for open-world object detection. Two vision models are chosen: Florence 2 and Grounding DINO.
- **Acne Detection Model**: The acne detection model is used to ensure that both acne and bindis can be detected in the same image. This helps to prevent the model from learning that bindis and acne do not appear together in one image.
- **[YOLOv8](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)**: A pre-trained YOLOv8 model is used for face detection.
- **Mediapipe**: This library is used for facial landmark detection, which helps in face refinement and further image processing.

### Installation & Prerequisites

Setup requires the same prerequisites as in [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#installation). 

Note:
- Checkpoints inside `bindis/bindi_annotation/GroundedSAM2/checkpoints` are not included in this repository. You must obtain them from the original repository.
- Additional libraries for the acne detection model are required. A list of these libraries can be found in the [mmdetection documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).
- The model checkpoint for acne detection can be obtained from [kaleido-models](https://github.com/remove-bg/kaleido-models). It should be stored in the following path: `bindis/bindi_annotation/AcneDetection/epoch_16.pth`.

A list of all required packages used for running the dataset creation is included in `packages.txt`.

## Running the Code

To run the dataset creation process, navigate to the `bindis/bindi_annotation` directory and use the following command:

```bash
python3 main.py --image_dir bindis/bindi_annotation/test_input --prompt "dot. bindi. metal" --save_path bindis/bindi_annotation/test_output
```

**Explanation of Parameters**:

- --image_dir: Directory containing the images to be processed.
- --prompt: A list of keywords (separated by dots) that guide the object detection models to look for bindis. It's recommended to use no more than three keywords, as detection performance may decrease with longer prompts.
Example: "dot. bindi. metal" will help the model detect bindis.
- --save_path: Directory where the processed images and bounding boxes will be saved.

## Overview of Design Choices

1. Face Detection and Cropping:
- The image is processed to detect faces and crop them into square-shaped regions.
- The detected faces are processed sequentially for further analysis.

2. Model Processing Order:
- Grounding DINO & Florence 2: Detect objects based on the provided prompt.
- Acne Detection: Ensure acne and bindis are detected within the same image, with no bias towards excluding one based on the other.
- Filtering: Remove detections that are outside of the face or on irrelevant regions (like eyes or background).

3. Filtering Rules:
- Bindi Detection: Bindis are expected to appear on the forehead. This is enforced by filtering out any bindi bounding boxes outside the forehead region, using left & right eye landmarks to identify the forehead.
- Bounding Box Filtering: Filter out bounding boxes detected for the eyes and entire face, also remove any bounding boxes outside the face region using facial landmarks.
- Bounding Box Merging: If bounding boxes overlap to a certain threshold or one contains the other, they are merged to reduce duplication.

