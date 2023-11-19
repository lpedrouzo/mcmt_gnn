import pandas as pd
import numpy as np

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


def simple_negative_sampling(edge_df, edge_idx, edge_labels, negative_ratio=0.5):
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
    negative_ratio: float, optional
        A number between 0 and 1, representing the ratio of negative
        edges ver the total set of edges.
    Returns
    =========
    Same variables as in the Parameters, but sampled.
    """
    indices = np.array([])
    num_total_edges = len(edge_labels)
    num_neg_edges = negative_ratio*num_total_edges
    all_idx = np.arange(num_total_edges)

    # Getting positive edge indices and labels
    edge_idx_pos = edge_idx[edge_labels == 1]
    edge_labels_pos = edge_labels[edge_labels == 1]
    edge_df_pos = edge_df[edge_df.edge_labels == 1]

    num_pos = edge_labels.sum()
    non_match_idx = all_idx[edge_labels == 0]

    # Retrieve indices for a sample of those negative edges (negative sampling)
    indices = np.random.choice(non_match_idx, size=num_neg_edges, replace=False) 
    
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


def get_n_rows_per_group(group, size):
    """ For a pandas DataFrame group, 
    Pick 'size' number of rows using np.linspace
    (evenly distributed)

    Parameters
    ==========
    group: pd.DataFrame
        An individual group as a result of df.groupby(group_columns)
    size: int
        How many rows to retrieve
    
    Returns
    =========
    pd.DataFrame   
        The group with 'size' number of rows.
    """
    if len(group) > size:
        return group.loc[group.index[np.linspace(0, len(group)-1, size, dtype=int)]]
    else:
        return group
    

def precompute_samples_for_graph_id(initial_id_list, num_ids_per_graph, num_iterations):
        """ Computes

        Parameters
        ==========
        None

        Returns 
        ==========
        None
        """
        print("Sampling object ids")
        precomputed_id_samples = []
        remaining_obj_ids = initial_id_list

        for _ in range(num_iterations):
            sampled_ids, remaining_obj_ids = sample_obj_ids(remaining_obj_ids,
                                                            num_ids_per_graph)
            precomputed_id_samples.append(sampled_ids)
        return precomputed_id_samples


def sample_obj_ids(unique_obj_ids, num_ids):
    """ Sample a specified number of unique object IDs from a list of unique IDs.

    Parameters
    ----------
    self : object
        An instance of the class.
    det_df : pd.DataFrame
        A DataFrame containing detection information.
    unique_obj_ids : list
        A list of unique object IDs.
    num_ids : int
        The number of unique object IDs to sample.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the sampled object IDs.
    list
        A list of remaining unique object IDs after sampling.
    """
    if len(unique_obj_ids) > num_ids:
        # Getting unique IDs and taking a sample of them
        sampled_obj_ids = np.random.choice(unique_obj_ids, size=num_ids, replace=False)

        # Getting the remaining IDs for further iterations
        remaining_obj_ids = [id for id in unique_obj_ids if id not in sampled_obj_ids]
    else:
        sampled_obj_ids = unique_obj_ids
        remaining_obj_ids = []
        
    # Return the DataFrame with only the sampled objects and the list of remaining IDs
    return sampled_obj_ids, remaining_obj_ids