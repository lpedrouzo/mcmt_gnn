# Multi-Object Multi-Camera tracking with Graph Neural Networks
This repo is currently under active development. I know documentation needs a little bit of housekeeping. Just hang tight, we will get there
once the solution is good enough. I need to graduate first, then we can take care of making the documentation pretty.

# Setting up the workspace

## Dataset

IMPORTANT. Before setting up the repository, make sure you have access to the [Nvidia AI City Challenge](https://www.aicitychallenge.org/2021-track3-download/) (AIC) dataset.

## Env, pip packages, REID models
Clone the repository and change directory:

```bash
git clone https://github.com/hector6298/mcmt_gnn.git
cd mcmt_gnn
```

Create a conda environment:

```bash
conda create --name mcmt_env python=3.9
conda activate mcmt_env
```

Install python dependencies:

```bash
pip install -r requirements.txt
```

Download the SOLIDER [CITE] REID model:

```bash
gdown 'https://drive.google.com/uc?id=12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8&export=download' -O models/reid/st_reid_weights.pth
```

Download ResNet101 - BN REID model for vehicles (see [LCFractal's repo](https://github.com/LCFractal/AIC21-MTMC) as they authors are the owners of this model):

```bash
gdown https://drive.google.com/file/d/105guaZrBzOF92-gmo0q7yWxHJKu1bXcv/view?usp=sharing -O models/reid/resnet101_ibn_a_2.pth
```

Move the raw AIC dataset you just downloaded into a folder inside datasets/raw folder with root on this repo:

```bash
mkdir datasets
mkdir datasets/raw
mv <AIC_original_path> datasets/raw
```

And now we should be all set to start preprocessing the data!

# Preprocessing the dataset
Now we need to make this dataset our own! (won't be literally ours, we still need to stick to their policies).
For this take a look at `config/preprocessing.yml` which is the configuration file for all of the preprocessing tasks.
We need to make sure that the key `original_aic_dataset_path` is properly set by pasting the path to your raw dataset:

```yaml
01_prep_videos_annotations:
  original_aic_dataset_path: "datasets/raw/AIC22" # For Example
  # Partitions to process
  dataset_partitions:
    - train
    - validation
  # Relative paths to annotations with respect to <sequence_path>/<camera_name>/
  sc_train_preds_path: "mtsc/mtsc_deepsort_ssd512.txt"
  sc_test_preds_path: "mtsc/mtsc_tc_mask_rcnn.txt"
  annotations_path: "gt/gt.txt"
```

Now, let's take a quick look into the configuration file `config/preprocessing.yml`

```yaml
common_params:
  annotations_filename: 'gt.txt' # common name for all ground truth annotations
  video_filename: 'vdo.avi' # The common name for all videos
  video_format: '.avi' # The video extension
  sequence_path: "datasets/AIC22/" # The path relative to the repo where the dataset is
  sequence_names: ["S01", "S02", "S03", "S04"] # THe list of sequences to include (both train and test)
  sc_preds_filename: mtsc_deepsort_ssd512.txt # Single camera predictions
  remove_past_epoch_embeddings: false # Should we delete embeddings on every pass?
```
The common params, are parameters that are used by multiple preprocessing tasks:
- **annotations_filename:** The name of the ground truth annotations. There is a different folder name for every camera in every sequence, but this name stays consistent. Which is why it is included as parameter.
- **video_filename:** The name of the videos. There is a different folder name for every camera in every sequence, but this name stays consistent. Which is why it is included as parameter.
- **sequence_path:** This will be the path prefix for our processed dataset. There will be folders like `frames, embeddings, annotations, logs, graphs, videos` and they will be stored inside `sequence_path`. Take into account that this prefix path is **relative to the root of the repository!**
- **sequence_names:** A list with all of the sequences included in the experiments. This list will include sequences intended for training and testing, and the separation between train and test will be defined in the configuration keys in further tasks. This key is mainly used in preprocessing, not in training.
- **sc_preds_filename:** This file corresponds to the single camera tracking outputs from a tracking run previous to this experiment. This files are issued by the AIC dataset. The name of each file gives a hint of the detectors and trackers used. There is a different folder name for every camera in every sequence, but this name stays consistent. Which is why it is included as parameter.

Then, there are also specific configurations for each task, and those configurations will be encapsulated by a key that matches with the name of the script. For instance:

```yaml
03_preprocess_annotations: # This corresponds to 03_preprocess_annotations.py
  sort_column_name: 'frame'
```

## 1. Moving videos and annotations
Execute the first task which is meant to setup basic folders in our `dataset` folder:

```
python processing/dataset_preparation/01_prep_videos_annotations.py
```

This will create two initial folders:
- videos:  The dataset videos per camera.
- annotations: The annotation files per camera. Notice that there will be a ground truth and annotations from single camera tracks from existing systems.

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

## 2. Extracting frames from videos
Now we are going to add an additional `frames` folder under `datasets` which will store all of the frames for every single video, in every camera, for every sequence.
For this execute:

```
python processing/dataset_preparation/AIC20/02_extract_frames.py
```

Make sure that the `sequence_path` parameter matches the actual path of where your videos and annotations are, before extracting the frames.

## 3. Preprocessing annotations
This step is about including the header names in the annotation files and including new columns that will be used on subsequent tasks. Execute:

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

#### Further steps to be documented...

# Organization of the repository

## modules 
There are several tools created for the realization for this project. These tools are held inside the folder `modules` which is organized as a python package (with an `__init__.py` file)
You will find REAMDE.md files inside each folder, describing the tools and utility functions created in more depth.

- **data_processor:** Python classes that will aid the preprocessing steps.
    - annotations_processor
    - embeddings_processor
    - video_processor
    - utils

- **torch_dataset:** Definitions of custom pytorch datasets, including Pytorch Geometric datasets to feed the Graph Neural Network training
    - bounding_box_dataset
    - object_graph_dataset
    - sequence_graph_dataset

- **torch_trainer:** Definitions for training engines, meant to simplify the training process, including metrics, LR schedulers, logging, and tensorboard. Also, custom losses are define in this sub-module:
    - custom_loss
    - trainer_abstract
    - trainer

## models
A centralized section of the repository where the model architectures and weights are stored.

- **layers:** This folder holds custom layers or nn.Modules 
    - MLP
- **mcmt:** The center of it all, here lies the definitions of the Graph Neural Networks to tackle MCMT.
    - rgnn
- **reid:** This folder holds Convolutional Neural Network architecture definitions and weights specialized in vehicle re-identification (REID).
    - resnet
    - swin_transformer

## processing
This section holds the scripts that are actually executed, for preprocessing, training, inference, and evaluation. They are numbered, so that the user can know in which order they are supposed to be executed. There are some scripts that have a letter next to the number. That means that they are on the same level of execution, but they differ on the underlying logic. This is thought to support variations in the process.

- **dataset_preparation:** Scripts to preprocess the raw datasets. These scripts will transform the dataset before training and will use the following modules:
    - modules.data_processor
- **training:** Tasks with trainning strategies. These tasks use:
    - modules.torch_dataset
    - modules.torch_dataloader
    - modules.torch_trainer
- **evaluation:** Tasks made for inference and computation of performance metrics for MCMT tracking (Not link prediction. Link prediction metrics are extracted on the training phase as indicators of direct model performance):
    - modules.inference
    - modules.result_extraction

## config
Configuration files for processing, training, and evaluation.

## visualization
Tools created to visualize the videos.

## tests
These are just notebooks included in the repository. They don't take part in the actual process, they just demo how to use different modules.
- **clusters.ipynb:** This is an exploratory for a qualitative performance assessment of the quality of REID embeddings and to know the nature of the dataset.
- **gnn_model_test.ipynb:** This notebook instantiates and tests the GNN model.
- **graph_dataset_test.ipynb:** This notebooks instantiates and tests the graph dataset objects that were made with Pytorch Geometric (PyG).
- **training_engine_test.ipynb:** This notebook gives a quick demo of how to use the training engine to train and test a simple model using a simple dataset.
