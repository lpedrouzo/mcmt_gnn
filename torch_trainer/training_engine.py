import sys
from ignite.metrics import Precision, Recall, Accuracy, Loss
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, create_lr_scheduler_with_warmup
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers import ProgressBar
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from functools import partial

class TrainingEngine(object):
    def __init__(self, train_loader, val_loader, gnn_model, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gnn_model = gnn_model
        self.device = device

        # These parameters will be initialized in the setup functions
        self.optimizer = None
        self.loss_fn = None
        self.metrics = {}
        self.tb_logger = None

        # These will be ignite Engines
        self.train_engine = None
        self.train_evaluator = None
        self.val_evaluator = None
        

    def setup_trainer(self, train_step_fn, optimizer, criterion, progress_bar=True):
        self.optimizer = optimizer
        self.loss_fn = criterion
        train_step = partial(train_step_fn, 
                             model=self.gnn_model, 
                             optimizer=self.optimizer, 
                             criterion=self.loss_fn)
        self.train_engine = Engine(train_step)

        if progress_bar:
            ProgressBar().attach(self.train_engine)


    def setup_validation(self, val_step_fn, metrics):
        if self.train_engine is None:
            raise Exception("You should call setup_trainer() or pass a valid training engine on set_trainer() first!")
        
        # Setting up evaluator engines
        validation_step = partial(val_step_fn,
                                  model=self.gnn_model)
        self.train_evaluator = Engine(validation_step)
        self.val_evaluator = Engine(validation_step)
        self.setup_metrics(metrics)


    def setup_metrics(self, metrics):
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

        
    def setup_loggers(self, log_training=True, log_interval=100):

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

                msg = f"Validation Results - Epoch[{trainer.state.epoch}] - "
                for name, _ in self.metrics.items():
                    msg += f"{name}: {metrics['name']:.2f} - "
                print(msg)

        # Logger for validation metrics
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.val_evaluator.run(self.val_loader)
            metrics = self.val_evaluator.state.metrics

            msg = f"Validation Results - Epoch[{trainer.state.epoch}] - "
            for name, _ in self.metrics.items():
                msg += f"{name}: {metrics['name']:.2f} - "
            print(msg)


    def setup_checkpointer(self, checkpoint_score_fn, n_save=3):

        def score_function(engine):
            return checkpoint_score_fn(engine.state.metrics)
        
        # Checkpoint to store n_saved best models wrt score function
        model_checkpoint = ModelCheckpoint(
            "checkpoint",
            n_saved=n_save,
            filename_prefix="best",
            score_function=score_function,
            score_name="model_score",
            global_step_transform=global_step_from_engine(self.train_engine), # helps fetch the trainer's state
        )
        
        # Save the model after every epoch of val_evaluator is completed
        self.val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": self.gnn_model})


    def setup_lr_scheduler_with_warmup(self, lr_scheduler, 
                                       warmup_start_value=None, 
                                       warmup_end_value=None, 
                                       warmup_duration=None):
        if warmup_start_value is None and warmup_duration is None:
            warmup_duration = 0
            warmup_start_value = self.lr
            
        ignite_lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                              warmup_start_value=warmup_start_value,
                                                              warmup_end_value=warmup_end_value,
                                                              warmup_duration=warmup_duration,
                                                              save_history=True)
        
        self.train_engine.add_event_handler(Events.ITERATION_STARTED, ignite_lr_scheduler)
        

    def setup_tensorboard(self):

        # Define a Tensorboard logger
        tb_logger = TensorboardLogger(log_dir="tb-logger")

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


    def run_training(self, max_epochs):
        self.train_engine.run(self.train_loader, max_epochs)
        print("Training has finished successfully!")

        if self.tb_logger is not None:
            self.tb_logger.close()


    def get_training_engine(self):
        return self.train_engine
    
    def get_train_evaluator(self):
        return self.train_evaluator
    
    def get_val_evaluator(self):
        return self.val_evaluator
    
    def get_model(self):
        return self.gnn_model
    
    def set_training_engine(self, train_engine, optimizer, loss_fn):
        if self.train_engine is not None:
            raise Exception("Error. Training engine already set!")
        
        self.train_engine = train_engine
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    