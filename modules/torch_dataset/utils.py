
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import to_networkx

def remove_unidirectional_edges(edge_idx):
    """ For an array of edge indices,
    remove the indices that are unidirectional

    Parameters
    ==========
    edge_idx: np.array
        The edge indices as numpy arrays
    
    Returns
    ==========
    filtered_edge_idx: np.array
        The filtered edge_idx with only bidirectional 
        edges.
    """
    # Initialize a list to store sub-arrays with their inverses
    indices = np.empty(len(edge_idx), dtype=bool)

    for i, edge in enumerate(edge_idx):
        inverse_sub_arr = np.flip(edge)
        indices[i] = np.any(np.all(edge_idx == inverse_sub_arr, axis=1))

    # Convert the list back to a NumPy array
    return edge_idx[indices], indices


def simple_negative_sampling(edge_df, edge_idx, edge_labels, num_neg_samples=None):
    """ Perform negative sampling over the general set of edges so that they have
    the same proportion of positive-negative edges.

    Take into account that this function ASSUMES that the graph is undirected!
    
    Unlike stratified_negative_sampling, this one does not iterate through every 
    object_id to sample the negative edges, it just takes a sample from the whole set of 
    negatives.

    Parameters
    ==========
    edge_df: pd.DataFrame
        DataFrame with edge information including 
        columns=["src_node_id", "dst_node_id", "src_obj_id", "dst_obj_id"].
    edge_idx: torch.Tensor
        Edge indices as a torch tensor.
    edge_labels: torch.Tensor
        Edge labels (0 or 1) indicating whether source and 
        destination nodes belong to the same object.

    Returns
    =========
    Same variables as in the Parameters, but sampled.
    """
    indices = np.array([])
    num_total_edges = len(edge_labels)
    all_idx = np.arange(num_total_edges)

    # Getting positive edge indices and labels
    edge_idx_pos = edge_idx[edge_labels == 1]
    edge_labels_pos = edge_labels[edge_labels == 1]
    edge_df_pos = edge_df[edge_df.edge_labels == 1]

    num_pos = edge_labels.sum()
    non_match_idx = all_idx[edge_labels == 0]

    # Retrieve indices for a sample of those negative edges (negative sampling)
    indices = np.random.choice(non_match_idx, size=num_neg_samples or num_pos, replace=False) 
    
    # Use the index to retrieve the negative samples
    edge_idx_neg = edge_idx[indices]
    edge_labels_neg = edge_labels[indices]
    edge_df_neg = edge_df.iloc[indices]
    
    # Concatenate positives and negatives and return results
    edge_df = pd.concat([edge_df_pos, edge_df_neg], ignore_index=True)
    edge_idx = np.concatenate([edge_idx_pos, edge_idx_neg], axis=0)
    edge_labels = np.concatenate([edge_labels_pos, edge_labels_neg], axis=0)

    return edge_df, edge_idx, edge_labels


def stratified_negative_sampling(edge_df, edge_idx, edge_labels):
    """ Perform negative sampling per individual object_id so that they have
    the same proportion of positive-negative edges.

    Parameters
    ==========
    edge_df: pd.DataFrame
        DataFrame with edge information including 
        columns=["src_node_id", "dst_node_id", "src_obj_id", "dst_obj_id"].
    edge_idx: torch.Tensor
        Edge indices as a torch tensor.
    edge_labels: torch.Tensor
        Edge labels (0 or 1) indicating whether source and 
        destination nodes belong to the same object.

    Returns
    =========
    Same variables as in the Parameters, but sampled.
    """
    indices = np.array([])
    num_total_edges = len(edge_labels)
    all_idx = np.arange(num_total_edges)

    # Getting positive edge indices and labels
    edge_idx_pos = edge_idx[edge_labels == 1]
    edge_labels_pos = edge_labels[edge_labels == 1]
    edge_df_pos = edge_df[edge_df.edge_labels == 1]

    # Iterate through every object id
    for object_id in edge_df.src_obj_id.unique():

        # Get source nodes for a given object_id and get the negative edges
        num_pos_obj = edge_labels[edge_df.src_obj_id == object_id].sum()
        non_match_idx_obj = all_idx[(edge_df.src_obj_id == object_id) & (edge_labels == 0)]

        # Retrieve indices for a sample of those negative edges (negative sampling)
        sample_idx = np.random.choice(non_match_idx_obj, size=int(num_pos_obj*2), replace=False) 
        indices = np.append(indices, sample_idx).astype(np.int64)
    
    # Use the index to retrieve the negative samples
    edge_idx_neg = edge_idx[indices]
    edge_labels_neg = edge_labels[indices]
    edge_df_neg = edge_df.iloc[indices]
    
    # Also remove the inverses of those negative sampled to leave only undirected edges
    edge_idx_neg, unidirectional_inds = remove_unidirectional_edges(edge_idx_neg)
    edge_labels_neg = edge_labels_neg[unidirectional_inds]
    edge_df_neg = edge_df_neg[unidirectional_inds]

    # Concatenate positives and negatives and return results
    edge_df = pd.concat([edge_df_pos, edge_df_neg], ignore_index=True)
    edge_idx = np.concatenate([edge_idx_pos, edge_idx_neg], axis=0)
    edge_labels = np.concatenate([edge_labels_pos, edge_labels_neg], axis=0)

    return edge_df, edge_idx, edge_labels


def draw_pyg_network(data, class_ids=None, color_nodes=True, layout='circular', undirected=True):
    """ Visualize a PyTorch Geometric graph using NetworkX and Matplotlib.
    This function creates a visual representation of the input graph using NetworkX and Matplotlib. It supports
    optional filtering of edges based on their labels and node coloring based on node classes.

    - If 'class_ids' is provided, only edges with labels in 'class_ids' will be displayed.
    - If 'color_nodes' is set to True, nodes will be colored based on their class labels using a color map.

    Parameters
    ==========
    data : torch_geometric.data.Data
        The PyTorch Geometric data object representing the graph.
    class_ids : list or None, optional
        List of edge labels to include (if specified). Default is None.
    color_nodes : bool, optional
        Whether to color nodes based on their class (if available). Default is True.
    layout : str, optional
        Layout algorithm for node positioning ('circular' or 'spring'). Default is 'circular'.

    Returns
    ==========
    None
    """

    # Convert tensors to numpy
    edges_ind = data.edge_index.T.numpy()
    el = data.edge_labels.numpy()

    # If we need to pull certain edge labels
    if class_ids:
        edges_ind = edges_ind[np.isin(el,class_ids)]
        el = el[np.isin(el, class_ids)]

    G = to_networkx(data, to_undirected=undirected)

    # Define colormap
    cmap=plt.cm.viridis(np.linspace(0,1,G.number_of_edges()))

    # Use the selected layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    else:
        pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_color=data.y.tolist() if color_nodes else None, 
                           cmap=plt.cm.gist_ncar)
    nx.draw_networkx_labels(G, pos)

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_ind,
        width=1,
        alpha=0.5,
        edge_color=el,
        edge_cmap=plt.cm.brg
    )
    plt.show()