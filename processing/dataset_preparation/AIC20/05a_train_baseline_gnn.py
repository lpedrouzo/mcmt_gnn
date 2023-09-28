import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))
import yaml
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_trainer.trainer import TrainingEngineRGNNBinary, TrainingEngineRGNNMulticlass
from torch_trainer.custom_loss import focal_loss, CECustom
from torch_dataset.sequence_graph_dataset import SequenceGraphDataset
from torch_dataloader.custom_loader import NodeSamplingDataLoader, ObjectSamplingDataLoader
from models.mcmt.rgnn import MOTMPNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialization
with open("config/rgcnn.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    gnn_arch = config["gnn_arch"]
    training_config = config["training_config"]
    optimizer_config = config["training_config"]["optimizer"]
    lr_scheduler_config = config["training_config"]["lr_scheduler"]
    loss_config = config["training_config"]["loss"]
    dataset_config = config["training_config"]['dataset']
print("1. Success Opening configuration file.")

# 1 - Initializing data loaders using the YAML configuration parameters
train_dataset = SequenceGraphDataset(sequence_path_prefix=dataset_config['sequence_path_prefix'], 
                                     sequence_names=dataset_config['sequences_train'], 
                                     annotations_filename=dataset_config['annotations_filename'],
                                     transform=T.ToUndirected())
val_dataset = SequenceGraphDataset(sequence_path_prefix=dataset_config['sequence_path_prefix'], 
                                   sequence_names=dataset_config['sequences_val'], 
                                   annotations_filename=dataset_config['annotations_filename'],
                                   transform=T.ToUndirected())
print("2. Success Defining torch datasets.")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)#, num_objects_per_graph=10)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)#, num_objects_per_graph=10)
print("3. Success creating torch dataloaders.")

# 2 - Initializing Graph Neural Network
gnn = MOTMPNet(gnn_arch).to(device)
print("4. GNN defined.")

# 3 - Initialing optimizer
optimizer = SGD(gnn.parameters(),
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                dampening=optimizer_config["dampening"])
print("5. Success defining optimizer.")


# 4a - Binary Cross Entropy loss
if loss_config['loss_type'] == 'bce':
    criterion = BCEWithLogitsLoss(reduction='mean', 
                                  pos_weight=torch.tensor(loss_config['loss_weight']) 
                                        if 'loss_weight' in loss_config else None)
    TrainingEngine = TrainingEngineRGNNBinary

# 4b - Cross Entropy loss with optional fixed weights
elif loss_config["loss_type"] == 'ce':
    weights = (torch.tensor(loss_config['loss_weight']).to(device)
                 if 'loss_weight' in loss_config else None)
    criterion = CrossEntropyLoss(weight=weights, reduction='mean')
    TrainingEngine = TrainingEngineRGNNMulticlass

# 4c - Focal Loss
elif loss_config["loss_type"] == 'focal':
    criterion = focal_loss(loss_config['alpha'], loss_config['gamma'])
    TrainingEngine = TrainingEngineRGNNMulticlass

# 4d - Default is custom CrossEntropy Loss
else:
    criterion = CECustom()
    TrainingEngine = TrainingEngineRGNNMulticlass
print(f"6. Success creating loss function for {loss_config['loss_type']}")


# 5 - Learning rate decay scheduler
if lr_scheduler_config["scheduler_type"] == 'step':
    lr_scheduler = StepLR(optimizer, 
                         lr_scheduler_config["step_size"], 
                         lr_scheduler_config["gamma"])
    
elif lr_scheduler_config["scheduler_type"] == 'cosine':
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=lr_scheduler_config["tmax"])
else:
    raise Exception(f'Learning rate scheduler {lr_scheduler_config["scheduler_type"]} is not supported.')
print(f"7. Success creating learning rate scheduler: {lr_scheduler_config['scheduler_type']}")

# 6 - Definition  of training engine using our custom object
training_engine = TrainingEngine(train_dataloader, val_dataloader, gnn, training_config["checkpoint_path_prefix"], device)
training_engine.setup_engine(
    optimizer=optimizer,
    lr=optimizer_config["lr"],
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    warmup_duration=lr_scheduler_config["warmup_duration"],
    show_training_progress=True,
    metrics=["Accuracy", "Precision", "Recall"],
    resume_training_flag=training_config["load_previous_train_state"],
    use_tensorboard=True,
)

# Check if there are previous checkpoints. Resume training if there are
if os.path.exists(training_config["checkpoint_path_prefix"]):
    if len(os.listdir(osp.join(training_config["checkpoint_path_prefix"], "checkpoints"))) and training_config["load_previous_train_state"]:
        training_engine.load_train_state()

# Run or resume training
training_engine.run_training(max_epochs=training_config["epochs"])