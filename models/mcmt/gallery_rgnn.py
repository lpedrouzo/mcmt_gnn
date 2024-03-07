import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
from ..layers.gallery_combination import GalleryLinearCombinationLayer, GalleryGRUCombinationLayer, GalleryMLPCombinationLayer
from .rgnn import  MOTMPNet

# Set a global random seed for CPU
torch.manual_seed(11)

# Set a global random seed for CUDA (GPU) if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(11)

# Additional CUDA configurations for reproducibility (optional)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GalleryMOTMPNet(MOTMPNet):
    """
    GalleryMOTMPNet class extends the MOTMPNet class to include the concept of gallery
    instead of using mean trajectory embeddings
   
    Parameters
    ==========
    model_params : dict
        A dictionary containing parameters for configuring the model.
        - 'gallery_combinator' (dict): Parameters for configuring the gallery combinator.
            - 'layer' (str): The type of combinator layer ('gru' or other).
            - 'combinator_feature_size' (int): The size of input features to the combinator layer.
            - 'combinator_output_size' (int): The size of the output features from the combinator layer.
            - 'combinator_num_gru_layers' (int): Number of GRU layers if 'layer' is set to 'gru'.
            - 'frames_per_gallery' (int): Number of frames considered for each gallery input.
    """
    def __init__(self, model_params):

        super(GalleryMOTMPNet, self).__init__(model_params)

        self.model_params = model_params

        # Setting up the gallery combinator
        combinator_params = model_params['gallery_combinator']
        if combinator_params['layer'] == 'gru':
            self.combinator = GalleryGRUCombinationLayer(input_size=combinator_params['combinator_feature_size'],
                                                         hidden_size=combinator_params['combinator_output_size'],
                                                         num_layers=combinator_params['combinator_num_layers'])
        elif combinator_params['layer'] == 'mlp':
            self.combinator = GalleryMLPCombinationLayer(combinator_params['combinator_feature_size'],
                                                         combinator_params['frames_per_gallery'],
                                                         combinator_params['combinator_num_layers'],
                                                         combinator_params['combinator_output_size'])
        else:
            input_size = combinator_params['combinator_feature_size']*combinator_params['frames_per_gallery']

            self.combinator = GalleryLinearCombinationLayer(input_size=input_size,
                                                            output_size=combinator_params['combinator_output_size'])


    def forward(self, data):
        """ Forward pass of the net.
        The net recieves the galleries (3D tensors), produces 2D tensors
        using its combination layer, then it uses those embeddings with the normal process
        defined in MOTMPNet

        Parameters
        ==========
        data: torch_geometric.Data
            The graph object. Must contain 
                - data.x with 3D embedding. Shape: (batch_size, sequence_length, feature_size)
                - data.edge_index, the list of edges. Shape (2, number_of_edges)
                - data.edge_labels, shape: (number_of_edges,)

        Notes
        ==========
        This network does not recieves edge attributes as it generates them using the embeddings
        from the galleries.
        """
        x, edge_index = data.x, data.edge_index

        # perform gallery combination and encoding
        node_embeddings = self.combinator(x)

        # Generate edge attributes
        edge_attr = torch.cat((
            F.pairwise_distance(node_embeddings[edge_index[0]], 
                                node_embeddings[edge_index[1]]).view(-1, 1),
            1 - F.cosine_similarity(node_embeddings[edge_index[0]], 
                                    node_embeddings[edge_index[1]]).view(-1, 1)
        ), dim=1)
        
        data.x = node_embeddings
        data.edge_attr = edge_attr
        
        # Perform encoding and message passing using the original function in MOTMPNet
        return super().forward(data)