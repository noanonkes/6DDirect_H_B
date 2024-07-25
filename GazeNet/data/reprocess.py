"""
Program for reprocessing Gaze from Afar dataset (GAFA) bounding boxes.
`preprocess.py` should be called first!


For each video:
    1. Annotations are loaded;
    2. Get consecutive frames (sometimes there are jumps due to people walking out of frame);
    3. Limit outlier width and heigths to max +- 0.75 the standard deviaten
    4. Smooth the x, y locations over 7 frames using Gaussian kernel convolution.

Do not forget to change `root_dir` to correct path!
"""
import os
import numpy as np
import joblib
from pathlib import Path
import pickle
from scipy.signal import convolve2d


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def smooth_bb_over_frames(annotations, smoothing=5):
    head_bb = annotations["head_bb"]
    body_bb = annotations["body_bb"]
    index = annotations["index"]

    if len(head_bb) == 0 or len(body_bb) == 0 or len(index) == 0:
        return False
    
    head_bboxes, body_bboxes = [[head_bb[0]]], [[body_bb[0]]]
    for i in range(1, len(head_bb)):
        # consecutive frames
        if index[i - 1] == index[i] - 1:
            head_bboxes[-1].append(head_bb[i])
            body_bboxes[-1].append(body_bb[i])
        else:
            head_bboxes.append([head_bb[i]])
            body_bboxes.append([body_bb[i]])
    
    # some bounding boxes are incorrect due to faults in Open Pose annotations
    # this makes sure the outlier w and h are limited
    for i in range(len(head_bboxes)):
        head_bboxes[i] = np.vstack(head_bboxes[i])
        upper_limit = int(np.mean(head_bboxes[i][:, 2]) + 0.75 * np.std(head_bboxes[i][:, 2]))
        lower_limit = int(np.mean(head_bboxes[i][:, 2]) - 0.75 * np.std(head_bboxes[i][:, 2]))

        head_bboxes[i][:, 2:] = np.clip(head_bboxes[i][:, 2:], lower_limit, upper_limit) # head bbs are square
        
    # some bounding boxes are incorrect due to faults in Open Pose annotations
    # this makes sure the outlier w and h are limited

    for i in range(len(body_bboxes)):
        body_bboxes[i] = np.vstack(body_bboxes[i])

        upper_limit = int(np.mean(body_bboxes[i][:, 2]) + 0.75 * np.std(body_bboxes[i][:, 2]))
        lower_limit = int(np.mean(body_bboxes[i][:, 2]) - 0.75 * np.std(body_bboxes[i][:, 2]))

        body_bboxes[i][:, 2] = np.clip(body_bboxes[i][:, 2], lower_limit, upper_limit)

        upper_limit = int(np.mean(body_bboxes[i][:, 3]) + 0.75 * np.std(body_bboxes[i][:, 3]))
        lower_limit = int(np.mean(body_bboxes[i][:, 3]) - 0.75 * np.std(body_bboxes[i][:, 3]))

        body_bboxes[i][:, 3] = np.clip(body_bboxes[i][:, 3], lower_limit, upper_limit)

    # gaussian kernel
    kernel = np.expand_dims(gaussian(np.linspace(-1, 1, num=smoothing), mu=0, sig=0.1), axis=-1)
    kernel = kernel / kernel.sum()

    for i in range(len(head_bboxes)):
        head_bboxes[i] = convolve2d(head_bboxes[i], kernel, mode="same", boundary="symm")
        body_bboxes[i] = convolve2d(body_bboxes[i], kernel, mode="same", boundary="symm")

    head_bboxes = np.vstack(head_bboxes)
    body_bboxes = np.vstack(body_bboxes)

    annotations["head_bb"] = head_bboxes
    annotations["body_bb"] = body_bboxes

    return annotations


if __name__ == "__main__":
    DEBUG = False
    SMOOTHING = 7 # amount of frames to smooth bounding box over

    root_dir = Path("data/preprocessed/")

    if DEBUG:
        scenes = ["living_room"]
    else:
        scenes = ["library", "lab", "kitchen", "courtyard", "living_room"]

    # from sequence name to amount
    mapping_dict = {}

    for scene in scenes:

        # e.g. ["006", "007"]
        sequences = os.listdir(root_dir / scene)

        for sequence in sequences:
            print("Current sequence:", str(root_dir / scene / sequence))

            # e.g. ["Camera_13_4", "Camera_13_3"]
            cameras = os.listdir(root_dir / scene / sequence)

            for camera in cameras:
                if camera.endswith(".pkl"):
                    continue

                annotation_file_path = root_dir / scene / sequence / camera

                annotations = sorted([file for file in os.listdir(annotation_file_path) if file.startswith("annotations")])

                n2ann = dict()
                for ann_file in annotations:
                    
                    try:
                        print("Annotations file:", str(annotation_file_path / ann_file))
                        annotations = joblib.load(annotation_file_path / ann_file)
                    except:
                        print(f"Annotations file {str(annotation_file_path)} not found")
                        continue

                    cleaned_annotations = smooth_bb_over_frames(annotations, SMOOTHING)
                    if not cleaned_annotations:
                        print("Failed")
                        continue
                    
                    with open(annotation_file_path / ("clean_" + ann_file), "wb") as f:
                        pickle.dump(cleaned_annotations, f)