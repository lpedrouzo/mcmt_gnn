import mlflow

def update_gnn_config(gnn_config, search_space_config):
    """ This function receives the search space with the hyperparameters to update for
    the Graph Neural Network. Then, it overrides the default values from the YAML
    configuration file.

    Parameters
    ==========
    gnn_config: dict
        The default configuration for the Graph Neural Network
        that comes from the YAML file.
    search_space_config: dict
        The new values from the tuning trial. This values will override
        the gnn_config values.

    Returns
    dict
        The new configuration for the GNN.
    """
    
    # GNN architecture configuration override by the hyperparam tuning
    gnn_config['num_enc_steps'] = search_space_config['message_passing_steps']

    # Node encoder
    gnn_config['encoder_feats_dict']['nodes']['node_fc_dims'] = [
        int(gnn_config['encoder_feats_dict']['nodes']['node_in_dim']//(2**i))\
            for i in range(1, search_space_config['node_enc_fc_layers']+1)
            ]
    node_out_dim = int(gnn_config['encoder_feats_dict']['nodes']['node_fc_dims'][-1]//2)
    gnn_config['encoder_feats_dict']['nodes']['node_out_dim'] = node_out_dim

    # Edge encoder
    gnn_config['encoder_feats_dict']['edges']['edge_fc_dims'] = [
        search_space_config["edge_update_units"]  \
            for i in range(search_space_config['edge_enc_fc_layers'])
            ]
    edge_out_dim = gnn_config['encoder_feats_dict']['edges']['edge_fc_dims'][-1]
    gnn_config['encoder_feats_dict']['edges']['edge_out_dim'] = edge_out_dim

    # Node Update
    gnn_config['node_model_feats_dict']['fc_dims'] = [
        node_out_dim for i in range(search_space_config['node_update_fc_layers'])]
    
    # Edge update
    gnn_config['edge_model_feats_dict']['fc_dims'] = [
        edge_out_dim for i in range(search_space_config['edge_update_fc_layers'])]

    # Classifier update
    gnn_config['classifier_feats_dict']['edge_in_dim'] = gnn_config['edge_model_feats_dict']['fc_dims'][-1]

    return gnn_config


def warmup_lr(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    else:
        return 1.0
    

def get_or_create_experiment(experiment_name, artifact_location=None, tags= None):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters
    ==========
    experiment_name: str
        Name of the MLflow experiment.
    artifact_location: str, optional
        The location to store run artifacts.
        If not provided, the server picks an appropriate default.
    tags: Dict[str, Any], optional
        An optional dictionary of string keys and values to set as
        tags on the experiment.
    Returns
    ===========
    str
        ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, artifact_location, tags)
    
