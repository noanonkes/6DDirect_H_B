


# 6DDirect H+B: Body-Aware Head Pose Estimation
In a well-functioning democratic constitutional state, it is crucial for professionals such as journalists and politicians to work unimpeded, yet threats against them have increased, necessitating surveillance.
Manual surveillance requires extensive manpower and is limited in effectiveness, leading to the adoption of computer vision systems.
Our research aims to enhance surveillance by accurately predicting head and body rotation, as well as gaze direction, in surveillance footage.

To achieve this, we developed a model named *6DDirect H+B* that accurately determines the 6D poses of head and body for multiple individuals in surveillance images.
This model addresses challenges such as occlusions, varying subject distances from the camera, and diverse lighting conditions.
By integrating localization, classification, and rotation learning within a unified framework using a fine-tuned YOLOv5 backbone, our approach enhances the accuracy of rotation estimation.
Then, we apply 6DDirect H+B to the task of gaze direction estimation, using an LSTM to leverage changes in head and body rotations over time to predict where a person is looking, to demonstrate the effectiveness of our approach.

## Table of Contents
* [Datasets - Head Pose and Body Orientation Estimation](#datasets-head-pose-and-body-orientation-estimation)
    - [AGORA](#agora)
    - [CMU](#cmu)
* [Datasets - Gaze Direction Estimation](#datasets-gaze-direction-estimation)
    - [GAFA](#gafa)
* [Baselines for Head Pose Estimation](#baselines-for-head-pose-estimation)
    - [6DRepNet](#6drepnet)
    - [DirectMHP](#directmhp)

<a name="datasets-head-pose-and-body-orientation-estimation"></a>
## Datasets - Head Pose and Body Orientation Estimation

<!-- TOC --><a name="agora"></a>
### AGORA
* Project link: [https://agora.is.tue.mpg.de/]. Github link: [https://github.com/pixelite1201/agora_evaluation]. Using and downloading this dataset needs personal registration. We have no right to directly disseminate its data. You can construct AGORA-HPE following the steps below.

<!-- TOC --><a name="step-1-download-raw-images"></a>
#### Step 1: Download Raw Images
1. **Download the raw images for train-set and validation-set from [AGORA website](https://agora.is.tue.mpg.de/download.php).**
2. **Create necessary directories and extract the downloaded images:**

    ```bash
    mkdir -p AGORA/demo/images
    cd AGORA
    unzip ./path/to/download/validation_images_1280x720.zip -d demo/images
    unzip ./path/to/download/train_images_1280x720_<id>.zip -d demo/images

    # Move 10 folders of train-set raw images into one folder. The id is from 0 to 9.
    mv demo/images/train_<id>/* demo/images/train
    ```

<!-- TOC --><a name="step-2-download-and-extract-raw-data"></a>
#### Step 2: Download and Extract Raw Data
1. **Download the following raw data from AGORA:**
   - Camera (approx. 392KB) -> train_Cam.zip & validation_Cam.zip
   - SMPL-X fits gendered (1.3GB) -> smplx_gt.zip
   - Scan/Fit Info (43KB) -> gt_scan_info.zip
   - SMIL (SMPL-X format), the kid model -> smplx_kid_template.npy

2. **Create necessary directories and extract the downloaded files:**

    ```bash
    mkdir -p demo/Cam/validation_Cam demo/Cam/train_Cam demo/GT_fits demo/model/smplx

    # Extract Camera data
    unzip -j ./path/to/download/validation_Cam.zip validation_Cam/Cam/* -d demo/Cam/validation_Cam/
    unzip -j ./path/to/download/train_Cam.zip -d demo/Cam/train_Cam/

    # Extract SMPL-X fits and Scan/Fit Info
    unzip ./path/to/download/gt_scan_info.zip -d demo/GT_fits
    unzip ./path/to/download/smplx_gt.zip -d demo/GT_fits
    ```

3. **Download and extract the SMPL-X models (npz version) from [SMPL-X website](https://smpl-x.is.tue.mpg.de/download.php):**

    ```bash
    unzip ./path/to/download/models_smplx_v1_1.zip SMPLX_FEMALE.npz SMPLX_MALE.npz SMPLX_NEUTRAL.npz -d demo/model/smplx/
    ```

<!-- TOC --><a name="step-3-download-agora-evaluation-code-and-generate-cam_withjvpkl-files"></a>
#### Step 3: Download AGORA Evaluation Code and Generate `Cam/*_withjv.pkl` Files
1. **Clone the AGORA evaluation repository:**

    ```bash
    git clone https://github.com/pixelite1201/agora_evaluation
    cd agora_evaluation
    # If this fails, change sklearn to scikit-learn in `setup.py`
    pip install .
    ```

2. **Move the downloaded SMIL kid model template:**

    ```bash
    mv ./path/to/download/smplx_kid_template.npy ./utils/
    ```

3. **Create symbolic links and replace with modified files:**

    ```bash
    # Using full path is necessary!
    ln -s ~/full/path/to/AGORA/demo ~/full/path/to/AGORA/agora_evaluation/demo

    # Replace with two modified files
    cp ../../exps/AGORA/agora_evaluation/projection.py agora_evaluation/
    cp ../../exps/AGORA/agora_evaluation/get_joints_verts_from_dataframe.py agora_evaluation/
    cp ../../exps/AGORA/agora_evaluation/project_points.py agora_evaluation/
    ```

4. **Install the SMPL-X model:**

    ```bash
    git clone https://github.com/vchoutas/smplx ../smplx
    cd ../smplx
    pip install .
    ```

5. **Run the script to generate the .pkl files with joint and vertex data:**

    ```bash
    cd ../agora_evaluation

    # Validation
    python agora_evaluation/project_joints.py --imgFolder demo/images/validation --loadPrecomputed demo/Cam/validation_Cam \
      --modeltype SMPLX --kid_template_path utils/smplx_kid_template.npy --modelFolder demo/model \
      --gt_model_path demo/GT_fits/ --imgWidth 1280 --imgHeight 720
    
    # Train
    python agora_evaluation/project_joints.py --imgFolder demo/images/train --loadPrecomputed demo/Cam/train_Cam \
      --modeltype SMPLX --kid_template_path utils/smplx_kid_template.npy --modelFolder demo/model \
      --gt_model_path demo/GT_fits/ --imgWidth 1280 --imgHeight 720
    ```

<!-- TOC --><a name="step-4-generate-the-final-agora-hb-dataset"></a>
#### Step 4: Generate the Final AGORA-H+B Dataset
1. **Create directories for the final dataset and annotations:**

    ```bash
    cd ..
    mkdir -p H_B/images/validation H_B/images/train H_B/annotations
    ```

2. **Copy necessary scripts and process the data:**

    ```bash
    cp ../exps/AGORA/H_B_data_process.py ./
    cp ../exps/AGORA/H_B_utils.py ./

    # Generate head and body bounding boxes + 6D rotations
    python H_B_data_process.py --load_pkl_flag # for head and body
    ```

<!-- TOC --><a name="step-5-prepare-labels-for-training"></a>
#### Step 5: Prepare Labels for Training
1. **Ensure labels are within the [0, 1] range and write them to the correct files for YOLOv5 integration:**

    ```bash
    cd ../6DDirect_H_B/
    python utils/labels.py --data data/agora_coco.yaml
    ```

This should give approximately the following folder setup for AGORA:

```bash
└── AGORA
    ├── agora_evaluation/
    │   ├── agora_evaluation/
    │   └── ...
    ├── demo/
    │   ├── Cam/
    │   │   ├── train_Cam/
    │   │   └── validation_Cam/
    │   ├── GT_fits/
    │   │   ├── gt_scan_info/
    │   │   └── smplx_gt/
    │   │       └── ...
    │   ├── images/
    │   │   ├── train/
    │   │   └── validation/
    │   └── model/
    │       └── smplx/
    ├── H_B/
    │   ├── 6D_body_head_yolov5_labels_coco/
    │   │   ├── img_txt/
    │       ├── train/
    │   │   └── validation/
    │   ├── annotations/
    │       ├── full_body_head_coco_style_train.json
    │       └── full_body_head_coco_style_validation.json
    │   └── images/
    │       ├── train/
    │       └── validation/
    └── smplx/
        └── ...
```

<!-- TOC --><a name="cmu"></a>
### CMU
* Project link: [http://domedb.perception.cs.cmu.edu/]. Github link: [https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox]. Using and downloading this dataset needs `personal registration`. We have no right to directly disseminate its data. You can construct CMU-H+B following steps below.

<!-- TOC --><a name="step-1-setting-up-directories-and-copying-necessary-scripts"></a>
#### Step 1: Setting Up Directories and Copying Necessary Scripts

1. **Create and navigate to the `CMU` directory:**

    ```bash
    mkdir CMU
    cd CMU
    ```

2. **Copy the necessary scripts and Python files from the `exps` directory:**

    ```bash
    cp -r ../exps/scripts ./
    cp ../exps/H_B_data_process.py ./
    cp ../exps/H_B_utils.py ./
    cp ../exps/data_split.py ./
    ```

3. **Understanding the Provided Files:**

- **`scripts/`**:
    - This directory contains scripts from the [CMU-Perceptual-Computing-Lab/panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and some modifications made by DirectMHP and this author.
    
- **`H_B_data_process.py`**:
    - Tweaked from DirectMHP GitHub to get head and body rotations.
    - Downloads data in a loop and directly removes folders to manage the large data size from the CMU Panoptic dataset.

- **`data_split.py`**:
    - Modified from DirectMHP GitHub to accommodate different COCO dictionary templates.
    
- **`H_B_utils.py`**:
    - Adjusted from DirectMHP GitHub to e.g. include a reference body.

<!-- TOC --><a name="step-3-processing-the-cmu-panoptic-dataset"></a>
#### Step 3: Processing the CMU Panoptic Dataset

The dataset is very large and downloading it will take a long time. To manage this:

- We process and sample data per sequence.
- Delete the folder after processing each sequence to save space.
- Save annotations after processing each sequence to prevent data loss if something crashes.

1. **Create a directory for head and body data processing:**

    ```bash
    mkdir H_B
    ```

2. **Run the data processing script:**

    ```bash
    python H_B_data_process.py  # Processes head and body data
    ```

<!-- TOC --><a name="step-4-combining-and-splitting-the-data"></a>
#### Step 4: Combining and Splitting the Data

After processing and saving annotations for each sequence:

1. **Combine the separate annotations into a single file and split it into training and validation datasets:**

    ```bash
    python data_split.py
    ```

<!-- TOC --><a name="step-5-prepare-labels-for-training-1"></a>
#### Step 5: Prepare Labels for Training
1. **Ensure labels are within the [0, 1] range and write them to the correct files for YOLOv5 integration:**

    ```bash
    cd ../6DDirect_H_B/
    python utils/labels.py --data data/cmu_panoptic_coco.yaml
    ```

This should approximately give the following folder structure for CMU:

```bash
└── CMU/
    ├── H_B/
    │   ├── 6D_body_head_yolov5_labels_coco/
    │   │   ├── img_txt/
    │   │   ├── train/
    │   │   └── val/
    │   ├── annotations/
    │   │   ├── coco_style_sample.json
    │   │   ├── coco_style_sampled_train.json
    │   │   └── coco_style_sampled_validation.json
    │   ├── images/
    │   │   ├── train/
    │   │   └── validation/
    │   └── images_sampled/
    └── scripts/
```

<!-- TOC --><a name="datasets-gaze-direction-estimation"></a>
## Datasets - Gaze Direction Estimation
<!-- TOC --><a name="gafa"></a>
### GAFA

<!-- TOC --><a name="step-1-cloning-the-dynamic-3d-gaze-from-afar-repository-and-setting-up"></a>
#### Step 1: Cloning the Dynamic 3D Gaze From Afar Repository and Setting Up

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar.git
    cd dynamic-3d-gaze-from-afar
    ```

2. **Copy the necessary files to the `data` directory:**

    ```bash
    cp exps/dynamic-3d-gaze-from-afar/data/coco_GAFA.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/dataset-demo.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/preprocess.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/reprocess.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/split_GAFA.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/utils.py data/
    cp exps/dynamic-3d-gaze-from-afar/data/preprocessed/transforms.py data/preprocessed/
    ```

<!-- TOC --><a name="step-2-downloading-the-dataset"></a>
#### Step 2: Downloading the Dataset

Download the data from the [GAFA GitHub](https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar). Note that this will take a very long time due to the large file sizes. Place the downloaded files in the `dynamic-3d-gaze-from-afar/data/raw_data` folder.

<!-- TOC --><a name="step-3-processing-the-dataset"></a>
#### Step 3: Processing the Dataset

**Option 1: Process All Data at Once (Requires Ample Disk Space)**. 
Extract and preprocess each tar.gz file:

```bash
cd dynamic-3d-gaze-from-afar/
names=("living_room" "courtyard" "library" "kitchen" "lab")
for NAME in "${names[@]}"; do
  tar -zxvf "data/raw_data/$NAME.tar.gz"
done

python data/preprocess.py
python data/reprocess.py
```

**Option 2: Process Data One Folder at a Time (For Limited Disk Space)**. 
Unzip, preprocess, and clean up each folder one by one:

```bash
cd dynamic-3d-gaze-from-afar
names=("living_room" "courtyard" "library" "kitchen" "lab")
for NAME in "${names[@]}"; do
  tar -zxvf "data/raw_data/$NAME.tar.gz"
  python data/preprocess.py
  rm -rf "$NAME"
done

python data/reprocess.py
```

<!-- TOC --><a name="step-4-details-of-processing-scripts"></a>
#### Step 4: Details of Processing Scripts

- **`preprocess.py`**:
    - Processes the GAFA dataset to create annotations and pickle files.
    - Steps per video:
        1. For all directions (gaze, body, head), get the corresponding rotation matrices.
        2. Resize the frames so the width is 720 pixels.
        3. Extract head and body bounding boxes from OpenPose 2D annotations.
        4. Save all frames to an `image.pkl` file.
        5. Save all annotations to an `annotations.pkl` file.
    - Note: Long videos are split into three parts to save RAM.

- **`reprocess.py`**:
    - Further processes the preprocessed data.
    - Steps:
        1. Load annotations.
        2. Get consecutive frames (skipping frames where subjects walk out of frame).
        3. Limit outlier widths and heights to a maximum of ±0.75 the standard deviation.
        4. Smooth the x, y locations over 7 frames using Gaussian kernel convolution.

<!-- TOC --><a name="step-5-generating-coco-style-annotations-and-splitting-data"></a>
#### Step 5: Generating COCO Style Annotations and Splitting Data

1. **Run the `coco_GAFA.py` script:**

    ```bash
    python data/coco_GAFA.py
    ```

    - **`coco_GAFA.py`**:
        - Converts GAFA dataset to COCO style annotations.
        - Steps:
            1. Samples every 7th frame of valid frames.
            2. Gets all valid frames for samples and their annotations (bounding boxes, gaze/head/body direction, 6D poses, Euler angles).
            3. Saves image information and annotations to a COCO style dictionary.

2. **Run the `split_GAFA.py` script:**

    ```bash
    python data/split_GAFA.py
    ```

    - **`split_GAFA.py`**:
        - Splits the COCO style annotations into training and validation sets (75% train, 25% validation).

<!-- TOC --><a name="step-5-prepare-labels-for-training-2"></a>
#### Step 5: Prepare Labels for Training
1. **Ensure labels are within the [0, 1] range and write them to the correct files for YOLOv5 integration:**

    ```bash
    cd ../6DDirect_H_B/
    python utils/labels.py --data data/gafa_coco.yaml
    ```

This should give this approximate folder structure:
```bash
└── dynamic-3d-gaze-from-afar/data/preprocessed/
    ├── courtyard/
    ├── H_B_G/
    │   ├── 6D_body_head_yolov5_labels_coco/
    │   │   ├── img_txt/
    │   │   ├── train/
    │   │   └── val/
    │   ├── annotations/
    │   │   ├── GAFA_coco_style.json
    │   │   ├── GAFA_coco_style_train.json
    │   │   └── GAFA_coco_style_validation.json
    │   ├── images/
    │   │   ├── train/
    │   │   └── validation/
    ├── kitchen/
    ├── lab/
    ├── library/
    └── living_room/
```

Here's the corrected README for setting up and running the baselines for head pose estimation:

<!-- TOC --><a name="baselines-for-head-pose-estimation"></a>
## Baselines for Head Pose Estimation

<!-- TOC --><a name="6drepnet"></a>
### 6DRepNet

1. **Clone the 6DRepNet repository:**

    ```bash
    git clone https://github.com/thohemp/6DRepNet.git
    cd 6DRepNet/sixdrepnet
    ```

2. **Create the output directory and download the pre-trained weights:**

    ```bash
    mkdir output
    wget https://huggingface.co/HoyerChou/DirectMHP/resolve/main/SixDRepNet_AGORA_bs256_e100_epoch_last.pth -P output/
    wget https://huggingface.co/HoyerChou/DirectMHP/resolve/main/SixDRepNet_CMU_bs256_e100_epoch_last.pth -P output/
    ```

3. **Copy the necessary files from the `exps/sixdrepnet` directory:**

    ```bash
    cp ../../exps/sixdrepnet/gen_dataset_full_AGORA_CMU.py ./
    cp ../../exps/sixdrepnet/test.py ./
    cp ../../exps/sixdrepnet/datasets.py ./
    cp ../../exps/sixdrepnet/model.py ./
    cp ../../exps/sixdrepnet/utils.py ./
    ```

<!-- TOC --><a name="overview-of-files-and-changes"></a>
#### Overview of Files and Changes:
Most of the files are originally from [DirectMHP](https://github.com/hnuzhy/DirectMHP) which we tweaked.
- **`gen_dataset_full_AGORA_CMU.py`**: 
    - Tweaked to fix file paths and skip annotations that are not heads, as the dataset now includes heads and bodies.
- **`test.py`**: 
    - Adjusted to add the geodesic distance metric.
- **`datasets.py`, `model.py`, `utils.py`**: 
    - Sourced from the DirectMHP GitHub.

<!-- TOC --><a name="generating-head-crops-and-testing"></a>
#### Generating Head Crops and Testing:

1. **Generate head crops for AGORA:**

    ```bash
    python gen_dataset_full_AGORA_CMU.py --db ../../AGORA/agora_evaluation/HPE/ --img_size 256 --root_dir ./datasets/ --data_type val --filename files_val.txt
    ```

2. **Test on AGORA:**

    ```bash
    python test.py --dataset AGORA --data_dir ./datasets/AGORA/val --filename_list ./datasets/AGORA/files_val.txt --snapshot output/SixDRepNet_AGORA_bs256_e100_epoch_last.pth --gpu 0 --batch_size 1
    ```

3. **Generate head crops for CMU:**

    ```bash
    python gen_dataset_full_AGORA_CMU.py --db ../../CMU/HPE/ --img_size 256 --root_dir ./datasets/ --data_type val --filename files_val.txt
    ```

4. **Test on CMU:**

    ```bash
    python test.py --dataset CMU --data_dir ./datasets/CMU/val --filename_list ./datasets/CMU/files_val.txt --snapshot output/SixDRepNet_CMU_bs256_e100_epoch_last.pth --gpu 0 --batch_size 1
    ```

<!-- TOC --><a name="directmhp"></a>
### DirectMHP

1. **Clone the DirectMHP repository:**

    ```bash
    git clone https://github.com/hnuzhy/DirectMHP.git
    cd DirectMHP
    ```

2. **Copy the necessary files:**

    ```bash
    cp ../exps/DirectMHP/val.py ./
    cp ../exps/DirectMHP/data/agora_coco.yaml ./data/
    cp ../exps/DirectMHP/utils/mae.py ./utils/
    cp -r ../exps/DirectMHP/visualize/ ./
    ```

<!-- TOC --><a name="overview-of-files-and-changes-1"></a>
#### Overview of Files and Changes:
- **`val.py`**: 
    - Added calculation for geodesic distance.
- **`utils/mae.py`**: 
    - Added geodesic loss class, geodesic error as return, and skipped classifications that are not heads.
- **`data/agora_coco.yaml`**: 
    - Updated paths to data.
- **`visualize/*`**: 
    - New code for comparisons between our method and DirectMHP.

<!-- TOC --><a name="downloading-pre-trained-weights-and-evaluating"></a>
#### Downloading Pre-Trained Weights and Evaluating:

1. **Download the pre-trained weights:**

    ```bash
    wget https://huggingface.co/HoyerChou/DirectMHP/resolve/main/agora_m_1280_e300_t40_lw010_best.pt -P weights/
    ```

2. **Evaluate on the AGORA dataset:**

    ```bash
    python val.py --rect --data data/agora_coco.yaml --img 1280 --weights weights/agora_m_1280_e300_t40_lw010_best.pt --batch-size 8 --device 0
    ```

By following these instructions, you will be able to set up and run the baselines for head pose estimation using 6DRepNet and DirectMHP.