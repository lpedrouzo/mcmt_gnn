import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False, is_classifier=False):
        """ Multi-Layer Perceptron (MLP) neural network module.

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
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        if not(is_classifier):
            for dim in fc_dims:
                layers.append(nn.Linear(input_dim, dim))
                if use_batchnorm and dim != 1:
                    layers.append(nn.BatchNorm1d(dim,track_running_stats=False))

                if dim != 1:
                    layers.append(nn.ReLU(inplace=True))

                if dropout_p is not None and dim != 1:
                    layers.append(nn.Dropout(p=dropout_p))

                input_dim = dim
        else:
            for dim in fc_dims:
                layers.append(nn.Linear(input_dim, dim))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        """ Forward pass of the MLP.

        Parameters
        ===========
        input : Tensor
            Input data.

        Returns
        ===========
        Tensor
            Output of the MLP.
        """
        return self.fc_layers(input)
    