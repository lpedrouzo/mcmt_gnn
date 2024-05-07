# Multi-Object Multi-Camera tracking with Graph Neural Networks
This repository contains the code for a multi-object multi-camera tracking system in an offline setting using Graph Neural Networks for tracklet association and Connected Components to retrieve the global trajectories.

![My Movie 1 (2)](https://github.com/hector6298/mcmt_gnn/assets/41920808/b8f71a5a-c243-4ebb-a524-cdb7dc250649)

Take a look at te system diagram:
![sys_diag (2)](https://github.com/hector6298/mcmt_gnn/assets/41920808/ea62393e-e73b-4163-8a1c-be3e84c693a6)

And this is how the graphs are generated:
![avg (1)](https://github.com/hector6298/mcmt_gnn/assets/41920808/43c0f634-186c-45e1-abdf-d6f4da09551d)

For a more in-depth description, see my thesis document here.

#### Note
For an overview of how this repository is organized, and what the folders and scripts mean, please see [repo_organization.md](https://github.com/hector6298/mcmt_gnn/blob/main/repo_organization.md).

## Setting up the workspace

IMPORTANT. Before setting up the repository, make sure you have access to the [Nvidia AI City Challenge](https://www.aicitychallenge.org/2021-track3-download/) (AIC) dataset.

Clone the repository and change the directory:

```bash
git clone https://github.com/hector6298/mcmt_gnn.git
cd mcmt_gnn
```

Move the raw AIC dataset you just downloaded into a folder inside datasets/raw folder with root on this repo:

```bash
mkdir datasets
mkdir datasets/raw
mv <AIC_original_path> datasets/raw
```

### Creating conda environment via yml file


Create the environment with all the required packages:
```bash
conda env create -f environment.yml
```
This will create an environment named `mcmt_env`, which you then need to activate:
```bash
conda activate mcmt_env
```

### Creating conda environment manually
Alternatively, you can create the environment and install the dependencies manually.

Create a conda environment:

```bash
conda create --name mcmt_env python=3.9
conda activate mcmt_env
```

Install pyTorch with GPU through conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia   # check your cuda version
```

Install pyTorch Geometrics through conda:
```bash
conda install pyg -c pyg
``` 

Install the following python packages through conda:
```bash
conda install -c conda-forge mlflow networkx gdown mmengine optuna optuna-dashboard motmetrics pytorch-scatter seaborn opencv

```

### ReID pre-trained model

Download ResNet101 - BN REID model for vehicles (see [LCFractal's repo](https://github.com/LCFractal/AIC21-MTMC) as they authors are the owners of this model):

```bash
gdown 105guaZrBzOF92-gmo0q7yWxHJKu1bXcv -O models/reid/resnet101_ibn_a_2.pth
```



And now we should be all set to start preprocessing the data!

## Preprocessing the dataset
Now we need to make this dataset our own! (won't be literally ours, we still need to stick to their policies).
For this, take a look at `config/preprocessing.yml` which is the configuration file for all of the preprocessing tasks. Note that every set of parameters in this file is wrapped by a key that corresponds to a file inside `processing/dataset_preparation`.

### 1. Prepare videos and annotations
The first step moves all videos, bitmaps of the regions of interest, and annotations into a structure that works with this code. We need to make sure that the key `original_aic_dataset_path` is properly set by pasting the path to your raw dataset:

```yaml
01_prep_videos_annotations:
  original_aic_dataset_path: "datasets/raw/AIC22" # For Example
  # Partitions to process
  dataset_partitions:
    - train
    - validation
  # Relative paths to annotations
  preds_path: "mtsc/mtsc_tc_yolo3.txt"
  annotations_path: "gt/gt.txt"
  roi_filename: 'roi.jpg'
  video_filename: 'vdo.avi'
```
Now notice that inside this file there is a key called `common_params`, it contains values that are shared among the tasks. It has another key called `sequence_path` which corresponds to the path relative to the root of this repo where the dataset will be placed after executing 01_prep_videos_annotations. You do not need to change this.

Now, execute the first task:

```
python processing/dataset_preparation/AIC20/01_prep_videos_annotations.py
```

This will create two initial folders:
- videos:  The dataset videos per camera.
- annotations: The annotation files per camera. Notice that there will be a ground truth and annotations from single camera tracks from existing systems.
- roi: Bitmaps of the regions of interest.

Take your time to take a look at how the files are organized by sequences and cameras, like so:

```
|- datasets
| |- AIC20 # for instance
| |   |- videos
| |   |   |- <sequence_name> # e.g: "S01"
| |   |   |   |- <camera_folder_1> # e.g: "c001"
| |   |   |   |  |- vdo.avi
| |   |   |   |- <camera_folder_2>
| |   |   |   |  |- vdo.avi
| |   |- annotations
| |   |   |- <sequence_name> # e.g: "S01"
| |   |   |   |- <camera_folder_1> # e.g: "c001"
| |   |   |   |  |- gt.txt
| |   |   |   |- <camera_folder_2>
| |   |   |   |  |- gt.txt
```

### 2. Extract frames from videos
Now we are going to add an additional `frames` folder under `datasets` which will store all of the frames for every single video, in every camera, for every sequence. Take a look at the configurations inside `config/preprocessing.yml`, but you can leave them as they are.

``` yaml
02_extract_frames:
  sequences_to_process: ["S01", "S02", "S03", "S04"] # Input sequences as needed
  video_filename: 'vdo.avi'
  video_format: '.avi'
```

Now, execute:

```
python processing/dataset_preparation/AIC20/02_extract_frames.py
```

Make sure that the `sequence_path` parameter matches the actual path of where your videos and annotations are, before extracting the frames.

### 3. Preprocessing annotations

This step is about including the header names in the annotation files and including new columns that will be used on subsequent tasks. You can leave the configurations as they are, except for `sc_preds_filename`. You can include a single camera predictions file from `datasets/raw/AIC20/train/S03/c011/mtsc`. Every camera has a set of single-camera tracking predictions with the same filename.

``` yaml
03_preprocess_annotations:
  sequences_to_process: ["S01", "S02", "S03", "S04"]
  sort_column_name: 'frame'
  gt_filename: gt.txt # Type off without quotes if no gt processing needed
  sc_preds_filename: mtsc_tc_yolo3.txt # Type off if not needed
```

Execute:

```
python processing/dataset_preparation/AIC20/03_preprocess_annotations.py
```

After a successfull execution, the annotations will have the following columns (in that order):

- frame
- id
- xmin
- ymin
- width
- height
- lost
- occluded
- generated
- label
- xmax
- ymax
- camera
- sequence_name

### 4. Filtering Single-Camera tracking files
This task filters-out detections from the single-camera tracking estimations. It mainly performs:

- Region-of-Interest filtering
- Duplicate detection removal
- Removing detections with area less than `min_bb_area`

This is done for the single-camera tracking estimations that will be used  on the validation set. Look at the task keys:

``` yaml
04_filter_sc_tracking:
  validation_partition: 'S02'
  in_sc_preds_filename:  mtsc_tc_yolo3.txt
  out_sc_preds_filename: 'mtsc_tc_yolo3_roi_filtered.txt'
  min_bb_area : 750
  filter_frame_bounds: false
  filter_roi: true
```

You should only change `in_sc_preds_filename` by placing the tracking file from the previous task and `out_sc_preds_filename` with a name of your choice that represents the filtered tracking file.

Execute:

```
python processing/dataset_preparation/AIC20/04_filter_sc_tracking.py
```

Now you also need to do this for the ground truth:

``` yaml
04_filter_sc_tracking:
  validation_partition: 'S02'
  in_sc_preds_filename:  'gt.txt'
  out_sc_preds_filename: 'gt_roi_filtered.txt'
  min_bb_area : 750
  filter_frame_bounds: false
  filter_roi: true
```

Execute:

```
python processing/dataset_preparation/AIC20/04_filter_sc_tracking.py
```

### 5. Extract and store ReID embeddings
This task will use the Re-ID model weights from Fractal's repository to compute appearance embedding for every single detection. Then, it will store individual embeddings on the filesystem, grouped by frames, cameras, and sequences.

Let's look at the keys for an execution with the single-camera tracking estimations:

``` yaml
05_extract_reid_embeddings:
  max_detections_per_df: 5000
  model_type: 'resnet'
  model_path: 'models/reid/resnet101_ibn_a_2.pth'
  annotations_filename: 'mtsc_tc_yolo3_roi_filtered.txt'
  train_sequences: ["S01", "S03", "S04"]
  test_sequences: ["S02"]
  cnn_img_size: [384, 384]
  img_batch_size: 300
  augmentation: false
  add_detection_id: true
```

See the train sequences are S01, S02, S03 and the test sequence is S02. Please keep the split in that way. Moreover, write the model path, relative to the root of the repository. Optionally, you can play with the following parameters to fit the available resources of your machine:

- `max_detections_per_df`: How many detections to process at once. The more detections, the faster, but requires more memory.
- `cnn_img_size`: (height, width) The resulting resolution of the image after reshaping the original detections.
- `img_batch_size`: This task uses a Pytorch dataloader to load images, so more images more memory, and faster.

Execute:

```
python processing/dataset_preparation/AIC20/05_extract_reid_embeddings.py
```

Now we also need to do this for the ground truth files. Check out the keys:

``` yaml
05_extract_reid_embeddings:
  max_detections_per_df: 5000
  model_type: 'resnet'
  model_path: 'models/reid/resnet101_ibn_a_2.pth'
  annotations_filename: 'gt_roi_filtered.txt'
  train_sequences: ["S01", "S03", "S04"]
  test_sequences: ["S02"]
  cnn_img_size: [384, 384]
  img_batch_size: 300
  augmentation: false
  add_detection_id: true
```

Execute:

```
python processing/dataset_preparation/AIC20/05_extract_reid_embeddings.py
```

### 5a. Store average trajectory embeddings

``` yaml
05a_extract_trajectory_embeddings:
  annotations_filename: 'gt_roi_filtered.txt'
  train_sequences: ["S01", "S03", "S04"]
  test_sequences: ["S02"]
```

### 5a. Store Gallery embeddings

``` yaml
05b_extract_galleries:
  annotations_filename: 'gt.txt'
  train_sequences: ["S01", "S03", "S04"]
  test_sequences: ["S02"]
  frames_per_gallery: 20
```


### 6 Execute training iterations for average embeddings

``` yaml
06_hyperparameter_tuning:
  experiment_params:
    num_trials: 3
    experiment_name: "example_gallery"
  dataset_params:
    sequence_path: 'C:/Users/mejia/Documents/tfm/mcmt_gnn/datasets/AIC20'
    test_sequence: 'S02'
    gt_filename: "gt_roi_filtered.txt"
    sct_filename: "mtsc_tnt_ssd512_roi_filtered.txt"
    eval_metric: macro_f1_score
  search_space:
    # Dataset related
    ratio_neg_links_graph: [1] # 1 means original ratio
    num_ids_per_graph: [100]
    # Training related
    lr: [0.01]
    epochs: [20] 
    warmup_duration: [5] # 0 is no warmup
    optimizer_momentum: [0.9] # 0 is no momentum
    lr_scheduler_step_size: [50] # 0 is scheduler off
    # GNN architecture
    message_passing_steps: [1]
    node_enc_fc_layers: [1,3,5] # Feat num decrease /2 every layer
    edge_enc_fc_layers: [1,2,3] # Feat num increase x2 every layer
    node_update_fc_layers: [1,2,3] # Channels increase x2 every layer
    edge_update_fc_layers: [1,2,3] # Channels increase x2 every layer
    edge_update_units: [2, 4, 8, 12, 20, 30, 42, 64]
    # Gallery combinator
    input_format: ['gallery']
    gallery_combination: ['linear']
    # Postprocessing toggles
    pruning: [True]
    spliting: [True]
```