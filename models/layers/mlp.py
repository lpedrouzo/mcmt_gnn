import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, output_dim=None, dropout_p=0.4, use_batchnorm=False):
        """Multi-Layer Perceptron (MLP) neural network module.

        Parameters
        ===========
        input_dim : int
            Dimension of the input data.
        fc_dims : list or tuple
            List or tuple of integers specifying dimensions
            of the fully connected layers in the MLP.
        output_dim : int
            Dimension of the MLP output.
        dropout_p : float, optional
            Dropout probability for dropout layers. Default is 0.4.
        use_batchnorm : bool, optional
            Whether to use batch normalization after each fully connected layer. Default is False.

        Attributes
        ===========
        mlp : nn.Sequential
            Sequential container for MLP layers.
        """
        super().__init__()

        assert isinstance(fc_dims, (list, tuple)), \
            f'fc_dims must be either a list or a tuple, but got {type(fc_dims)}'
        
        self.mlp = nn.Sequential()
        dims = [input_dim] + fc_dims 

        # Set up input and hidden layers
        for i in range(len(dims)-1):
            self.mlp.add_module(f'layer_{i}', nn.Linear(dims[i], dims[i+1]))

            if use_batchnorm:
                self.mlp.add_module(f'bn_{i}', nn.BatchNorm1d(dims[i+1],track_running_stats=False))

            self.mlp.add_module(f'act_{i}', nn.ReLU(inplace=True))

            if dropout_p is not None:
                self.mlp.add_module(f'dropout_{i}', nn.Dropout(p=dropout_p))
        
        # Set classification layer if applicable
        if output_dim is not None:
            self.mlp.add_module(f'output', nn.Linear(dims[-1], output_dim))

    def forward(self, input):
        """Forward pass of the MLP.

        Parameters
        ===========
        input : Tensor
            Input data.

        Returns
        ===========
        Tensor
            Output of the MLP.
        """
        return self.mlp(input)