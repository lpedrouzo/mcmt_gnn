import sys
import torch
import os
import os.path as osp
import pandas as pd
from ignite.metrics import Precision, Recall, Accuracy, Loss
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, create_lr_scheduler_with_warmup, Checkpoint
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers import ProgressBar
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from functools import partial

class TrainingEngineAbstract(object):
    """TrainingEngine is a class for managing the training process 
    of a GNN model using Pytorch Ignite.

    Parameters:
    ===========
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    gnn_model : nn.Module
        The GNN model to be trained.
    checkpoint_path_prefix: str
        The directory prefix where engine checkpoints will be stored
    device : torch.device
        The device on which the model and data should be placed.

    Attributes:
    ===========
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    gnn_model : nn.Module
        The GNN model to be trained.
    device : torch.device
        The device on which the model and data should be placed.
    optimizer : Optimizer
        The optimizer used for training.
    loss_fn : callable
        The loss function used for training.
    metrics : dict
        A dictionary of metrics used for evaluation.
    tb_logger : TensorboardLogger
        TensorboardLogger for logging training progress.
    train_engine : Engine
        Ignite Engine for training.
    train_evaluator : Engine
        Ignite Engine for training evaluation.
    val_evaluator : Engine
        Ignite Engine for validation evaluation.
    checkpoint_path_prefix: str
        The directory prefix where the results will be stored
    ignite_lr_scheduler: ignite.handlers.param_scheduler.ConcatScheduler
        The learning rate scheduler from pytorch ignite
    current_epoch: int
        This variable will be 0 for an initial training session,
        or if a past session exists and the engine is set to resume training,
        it will get the last training epoch.
    """

    def __init__(self, train_loader, val_loader, gnn_model, results_path_prefix, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gnn_model = gnn_model
        self.device = device
        self.current_epoch = 0
        self.is_resumed_training = False #Will be true if load_train_state() is called

        # These parameters will be initialized in the setup functions
        self.optimizer = None
        self.loss_fn = None
        self.metrics = {}
        self.tb_logger = None
        self.ignite_lr_scheduler = None

        # These will be Ignite Engines
        self.train_engine = None
        self.train_evaluator = None
        self.val_evaluator = None

        # Results paths
        self.results_path_prefix = results_path_prefix 
        self.checkpoint_path = osp.join(self.results_path_prefix, "checkpoints")
        self.train_metrics_logs_path = osp.join(self.results_path_prefix, "metrics", "train_results.csv")
        self.val_metrics_logs_path = osp.join(self.results_path_prefix, "metrics", "val_results.csv")
        self.tensorboard_path = osp.join(self.results_path_prefix, "tensorboard")

    def setup_trainer(self, optimizer, criterion, progress_bar=True):
        """Set up the trainer for training.

        Parameters:
        ===========
        optimizer : Optimizer
            The optimizer to be used.
        criterion : callable
            The loss function.
        progress_bar : bool, optional
            Whether to use a progress bar during training. Default is True.
        """
        self.optimizer = optimizer
        self.loss_fn = criterion
        train_step = self.setup_train_step()
        self.train_engine = Engine(train_step)

        if progress_bar:
            ProgressBar().attach(self.train_engine)


    def setup_validation(self, metrics):
        """Set up the validation process.

        Parameters:
        ===========
        metrics : list
            List of metric names to be used for evaluation.
        """
        if self.train_engine is None:
            raise Exception("You should call setup_trainer() or pass a valid training engine on set_trainer() first!")

        # Setting up evaluator engines
        validation_step = self.setup_validation_step()
        self.train_evaluator = Engine(validation_step)
        self.val_evaluator = Engine(validation_step)
        self.setup_metrics(metrics)


    def setup_metrics(self, metrics):
        """Set up evaluation metrics.
        This function uses the exact strings to instantiate the metric class.
        Valid values are ['Precision', 'Recall', 'Accuracy'] case sensitive.

        Parameters:
        ===========
        metrics : list
            List of metric names to be used for evaluation.
        """
        if self.loss_fn is None:
            raise Exception("You should assign a loss function using setup_trainer() function first!")

        # Converting strings into ignite.metrics objects and storing in dictionary
        for metric in metrics:
            self.metrics[metric] = getattr(sys.modules[__name__], metric)()
        self.metrics["loss"] = Loss(self.loss_fn)

        # Attaching metrics to the validators
        for name, metric in self.metrics.items():
            metric.attach(self.train_evaluator, name)

        for name, metric in self.metrics.items():
            metric.attach(self.val_evaluator, name)

        # Preparing empty dataframes for logging
        self.train_metrics_df = pd.DataFrame(columns=["epoch", *metrics])
        self.val_metrics_df = pd.DataFrame(columns=["epoch", *metrics])



    def setup_loggers(self, log_training=True, log_interval=100):
        """Set up loggers for training and validation.

        Parameters:
        ===========
        log_training : bool, optional
            Whether to log training progress. Default is True.
        log_interval : int, optional
            The interval for logging training loss. Default is 100.
        """

        os.makedirs(osp.join(self.results_path_prefix, "metrics"), exist_ok=True)

        # Logger for training loss every log_interval iterations
        @self.train_engine.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

        if log_training:
            # Logger for training metrics
            @self.train_engine.on(Events.EPOCH_COMPLETED)
            def log_training_results(trainer):
                self.train_evaluator.run(self.train_loader)
                metrics = self.train_evaluator.state.metrics

                # Log current epoch
                row = {"epoch": trainer.state.epoch}
                msg = f"Train Results - Epoch[{trainer.state.epoch}] - "

                # Log current metrics
                for name, _ in self.metrics.items():
                    msg += f"{name}: {metrics[name]} - "
                    row[name] = metrics[name]
                print(msg)

                # This result logs will besaved to csv once trainer.run() is finished
                row_df = pd.DataFrame([row], columns=self.train_metrics_df.columns)
                self.train_metrics_df = pd.concat([self.train_metrics_df, row_df], ignore_index=True)         
                
        # Logger for validation metrics
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.val_evaluator.run(self.val_loader)
            metrics = self.val_evaluator.state.metrics

            # Log current epoch
            row = {"epoch": trainer.state.epoch}
            msg = f"Validation Results - Epoch[{trainer.state.epoch}] - "

            # Log current metrics
            for name, _ in self.metrics.items():
                msg += f"{name}: {metrics[name]} - "
                row[name] = metrics[name]
            print(msg)

            # This result logs will besaved to csv once trainer.run() is finished
            row_df = pd.DataFrame([row], columns=self.val_metrics_df.columns)
            self.val_metrics_df = pd.concat([self.val_metrics_df, row_df], ignore_index=True)         


    def setup_checkpointer(self, checkpoint_score_fn=None, resume_training_flag=False):
        """Set up model checkpointing.
        The checkpoint_score_fn function should be a function that
        receives a 'metrics' parameter (a dictionary str:float) and 
        defines a computation for the score.
        
        Example:
        ===========
        (2*metrics['Precision']*metrics['Recall'])/(metrics['Precision'] + metrics['Recall'])
        
        Valid keys are only the ones that were passed as parameters in the function
        setup_validation().

        Parameters:
        ===========
        checkpoint_score_fn : callable optional
            Function to determine the score for checkpointing.
        resume_training_flag: bool
            Whether the engine will resume training from last session or not.
        """
        def score_function(engine):
            return checkpoint_score_fn(engine.state.metrics)
        
        to_save = {
            "model": self.gnn_model,
            "optimizer": self.optimizer, 
            "trainer": self.train_engine,
        }
        if self.ignite_lr_scheduler is not None:
            to_save["lr_scheduler"] = self.ignite_lr_scheduler       

        # Checkpoint to store n_saved best models wrt score function
        model_checkpoint = ModelCheckpoint(
            self.checkpoint_path,
            score_function=score_function if checkpoint_score_fn is not None else None,
            score_name="model_score" if checkpoint_score_fn is not None else None,
            global_step_transform=global_step_from_engine(self.train_engine),  
            require_empty=(not resume_training_flag)
        )

        # Save the model after every epoch of val_evaluator is completed
        self.val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, to_save)


    def load_train_state(self):
        """ Loads the state of the training engine
        (model, optimizer, training_engine, lr_scheduler)
        using the last, previously saved checkpoint.

        Checkpoint files must have the following convention

        > checkpoint_1.pt, checkpoint_2.pt, ...
        """
        if not osp.exists(self.train_metrics_logs_path):
            raise Exception("There are no results paths to restore session. Please run a new training.")
        
        print("Resuming training.")
        to_load = {
            "model": self.gnn_model,
            "optimizer": self.optimizer, 
            "trainer": self.train_engine,
        }
        if self.ignite_lr_scheduler is not None:
            to_load["lr_scheduler"] = self.ignite_lr_scheduler   

        # Get the last checkpoint
        checkpoint_last = sorted(os.listdir(self.checkpoint_path), 
                                 key=lambda item: int(item.replace('.pt', '').split('_')[-1]))[-1]
        checkpoint_last = osp.join(self.checkpoint_path, checkpoint_last)

        # Load checkpoint and assign the objects to the trainer
        checkpoint = torch.load(checkpoint_last, map_location=self.device) 
        ModelCheckpoint.load_objects(
            to_load=to_load, 
            checkpoint=checkpoint
        ) 
        self.current_epoch = self.train_engine.state.epoch

        # Restore past metrics
        self.train_metrics_df =  pd.read_csv(self.train_metrics_logs_path)
        self.val_metrics_df =  pd.read_csv(self.val_metrics_logs_path)

        print(self.train_engine.state)


    def setup_lr_scheduler_with_warmup(self, lr_scheduler,
                                       warmup_start_value=None,
                                       warmup_end_value=None,
                                       warmup_duration=None):
        """Set up a learning rate scheduler with warm-up.

        Parameters:
        ===========
        lr_scheduler : LR scheduler
            The learning rate scheduler to be used.
        warmup_start_value : float, optional
            The initial learning rate value for warm-up. If not provided, it defaults to the current learning rate.
        warmup_end_value : float, optional
            The final learning rate value for warm-up. If not provided, it defaults to the current learning rate.
        warmup_duration : int, optional
            The number of iterations for warm-up. If not provided, it defaults to 0.

        Notes:
        ======
        The `ignite_lr_scheduler` is created with warm-up settings and added to the training engine.
        """
        if warmup_start_value is None and warmup_duration is None:
            warmup_duration = 0
            warmup_start_value = self.lr
            
        self.ignite_lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                              warmup_start_value=warmup_start_value,
                                                              warmup_end_value=warmup_end_value,
                                                              warmup_duration=warmup_duration,
                                                              save_history=True)
        
        self.train_engine.add_event_handler(Events.ITERATION_STARTED, self.ignite_lr_scheduler)
    

    def setup_tensorboard(self):
        """Set up Tensorboard logging.
        """
        # Define a Tensorboard logger
        tb_logger = TensorboardLogger(log_dir=self.tensorboard_path)

        # Attach handler to plot trainer's loss every 100 iterations
        tb_logger.attach_output_handler(
            self.train_engine,
            event_name=Events.ITERATION_COMPLETED(every=100),
            tag="training",
            output_transform=lambda loss: {"batch_loss": loss},
        )

        # Attach handler for plotting both evaluators' metrics after every epoch completes
        for tag, evaluator in [("training", self.train_evaluator), ("validation", self.val_evaluator)]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(self.train_engine),
            )


    def setup_engine(self, 
                     optimizer, 
                     lr, 
                     criterion, 
                     lr_scheduler=None, 
                     warmup_duration=None, 
                     show_training_progress=True, 
                     metrics=["Accuracy", "Precision", "Recall"], 
                     log_training=True, 
                     log_interval=100, 
                     checkpoint_score_fn=None, 
                     resume_training_flag=False,
                     use_tensorboard=True):
        counter = 1

        self.setup_trainer(optimizer=optimizer, 
                                    criterion=criterion,
                                    progress_bar=show_training_progress)
        print(f"{counter}. Added training function to TrainingEngine.")
        counter += 1
        
        self.setup_validation(metrics=metrics)
        print(f"{counter}. Added metrics to TrainingEngine.")
        counter += 1

        self.setup_loggers(log_training=log_training, log_interval=log_interval)
        print(f"{counter}. Added Loggers to TrainingEngine.")
        counter += 1

        if lr_scheduler is not None:
            self.setup_lr_scheduler_with_warmup(lr_scheduler, warmup_start_value=lr/10, warmup_end_value=lr, warmup_duration=warmup_duration)
            print(f"{counter}. Added learning rate scheduler to TrainingEngine with warmup {'enabled' if warmup_duration else 'disabled'}.")
            counter += 1

        self.setup_checkpointer(checkpoint_score_fn, resume_training_flag)
        print(f"{counter}. Added Checkpointer to TrainingEngine.")
        counter += 1

        if use_tensorboard:
            self.setup_tensorboard()
            print(f"{counter}. Tensorboard enabled.")


    def run_training(self, max_epochs):
        """Run the training process.

        Parameters:
        ===========
        max_epochs : int
            The maximum number of training epochs.
        """
        self.train_engine.run(self.train_loader, self.current_epoch + max_epochs)
        print("Training has finished successfully!")

        if self.tb_logger is not None:
            self.tb_logger.close()
        
        # Save metrics logs
        self.train_metrics_df.to_csv(self.train_metrics_logs_path)
        self.val_metrics_df.to_csv(self.val_metrics_logs_path)

        print(self.train_engine.state)


    def get_training_engine(self):
        """Get the training engine.

        Returns:
        ========
        Engine
            The Ignite training engine.
        """
        return self.train_engine


    def get_train_evaluator(self):
        """Get the training evaluator engine.

        Returns:
        ========
        Engine
            The Ignite training evaluator engine.
        """
        return self.train_evaluator


    def get_val_evaluator(self):
        """Get the validation evaluator engine.

        Returns:
        ========
        Engine
            The Ignite validation evaluator engine.
        """
        return self.val_evaluator


    def get_model(self):
        """Get the GNN model.

        Returns:
        ========
        nn.Module
            The GNN model.
        """
        return self.gnn_model


    def set_training_engine(self, train_engine, optimizer, loss_fn):
        """Set the training engine, optimizer, and loss function.

        Parameters:
        ===========
        train_engine : Engine
            The Ignite training engine.
        optimizer : Optimizer
            The optimizer used for training.
        loss_fn : callable
            The loss function used for training.

        Raises:
        ======-
        Exception
            If the training engine is already set.
        """
        if self.train_engine is not None:
            raise Exception("Error. Training engine already set!")

        self.train_engine = train_engine
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def setup_train_step(self):
        """ Abstract function that must be overriden.
        Inside this function, the user must create another function called
        train_step(engine, batch) and its behavior should be the core functionality
        of a single training iteration.
        Finally, return this function.

        Example:
        =========
        def setup_train_step(self)
            def train_step(engine, batch):
                model.train()
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                return loss.item()
            return train_step
        """
        raise NotImplementedError("You must override this function!")
    

    def setup_validation_step(self):
        """ Abstract function that must be overriden.
        Inside this function, the user must create another function called
        validation_step(engine, batch) and its behavior should be the core functionality
        of a single validation iteration.
        Finally, return this function.

        Example:
        =========
        def setup_validation_step(self)
            def validation_step(engine, batch):
                model.eval()
                with torch.no_grad():
                    x, y = batch
                    y_pred = model(x)
                return y_pred, y
            return validation_step
        """
        raise NotImplementedError("You must override this function!")

        