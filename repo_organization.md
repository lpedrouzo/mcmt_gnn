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
