import torch
import torch.nn as nn

from .mlp import MLP

class GalleryGRUCombinationLayer(nn.Module):
    """
    Custom PyTorch module for combining gallery embeddings using a GRU layer.

    Parameters
    ----------
    input_size : int
        Size of a single embedding within the gallery.
    hidden_size : int
        Size of the hidden state in the GRU layer.
    num_layers : int, optional
        Number of GRU layers. Default is 1.
    batch_first : bool, optional
        If True, input and output tensors have the batch dimension 
        as the first dimension. Default is True.

    Attributes
    ----------
    gru : torch.nn.GRU
        The GRU layer used for gallery combination.

    Methods
    -------
    forward(galleries)
        Forward pass of the module.

    """

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(GalleryGRUCombinationLayer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, galleries):
        """
        Forward pass of the module.

        Parameters
        ----------
        galleries : torch.Tensor
            Input tensor with shape (N, G, C), where N is the number of objects, 
            G is the size of the gallery, and C is the size of a single 
            embedding within the gallery.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (N, hidden_size), representing the aggregated 
            information from each gallery.
        """
        # Apply GRU layer
        gallery_embeddings, _ = self.gru(galleries)

        return gallery_embeddings[:, -1, :]  # Take the last hidden state as the output


class GalleryLinearCombinationLayer(nn.Module):
    """
    Linear combination layer for combining embeddings within galleries.

    Parameters
    ----------
    input_size : int
        Size of a single embedding within the gallery.
    output_size : int
        Desired size of the output embedding.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer for performing the combination.

    Methods
    -------
    forward(galleries)
        Forward pass of the layer.

    """

    def __init__(self, input_size, output_size):
        super(GalleryLinearCombinationLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, galleries):
        """
        Forward pass of the layer.

        Parameters
        ----------
        galleries : torch.Tensor
            Input tensor with shape (N, G, C), where N is the number of objects,
            G is the size of the gallery, and C is the size of a single
              embedding within the gallery.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (N, G * C), representing the combined 
            embeddings within the galleries.
        """
        # Reshape the galleries tensor to (N, G * C) for linear layer input
        flattened_galleries = galleries.view(galleries.size(0), -1)

        # Apply the linear layer
        gallery_embeddings = self.linear(flattened_galleries)

        return gallery_embeddings
    

class GalleryLinearCombinationLayer(nn.Module):
    """
    Linear combination layer for combining embeddings within galleries.

    Parameters
    ----------
    input_size : int
        Size of a single embedding within the gallery.
    output_size : int
        Desired size of the output embedding.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer for performing the combination.

    Methods
    -------
    forward(galleries)
        Forward pass of the layer.

    """

    def __init__(self, input_size, output_size):
        super(GalleryLinearCombinationLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, galleries):
        """
        Forward pass of the layer.

        Parameters
        ----------
        galleries : torch.Tensor
            Input tensor with shape (N, G, C), where N is the number of objects,
            G is the size of the gallery, and C is the size of a single
              embedding within the gallery.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (N, G * C), representing the combined 
            embeddings within the galleries.
        """
        # Reshape the galleries tensor to (N, G * C) for linear layer input
        flattened_galleries = galleries.view(galleries.size(0), -1)

        # Apply the linear layer
        gallery_embeddings = self.linear(flattened_galleries)

        return gallery_embeddings


class GalleryMLPCombinationLayer(nn.Module):
    """
    Linear combination layer for combining embeddings within galleries.

    Parameters
    ----------
    input_size : int
        Size of a single embedding within the gallery.
    output_size : int
        Desired size of the output embedding.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer for performing the combination.

    Methods
    -------
    forward(galleries)
        Forward pass of the layer.

    """

    def __init__(self, gallery_input_size, num_frames_per_gallery, fc_layers_num, output_size):
        input_size = gallery_input_size*num_frames_per_gallery
        fc_dims = [input_size/(2*i) for i in range(1, fc_layers_num+1)]

        super(GalleryMLPCombinationLayer, self).__init__()

        self.mlp = MLP(input_size, fc_dims + output_size, dropout_p=None)

    def forward(self, galleries):
        """
        Forward pass of the layer.

        Parameters
        ----------
        galleries : torch.Tensor
            Input tensor with shape (N, G, C), where N is the number of objects,
            G is the size of the gallery, and C is the size of a single
              embedding within the gallery.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (N, G * C), representing the combined 
            embeddings within the galleries.
        """
        # Reshape the galleries tensor to (N, G * C) for linear layer input
        flattened_galleries = galleries.view(galleries.size(0), -1)

        # Apply the linear layer
        gallery_embeddings = self.mlp(flattened_galleries)

        return gallery_embeddings