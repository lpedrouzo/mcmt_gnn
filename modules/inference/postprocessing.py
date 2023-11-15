import time
import torch
import numpy as np
import networkx as nx
import pandas as pd
from torch_scatter import scatter_add
from .utils import intersect, torch_isin


def connected_componnets(G, n_nodes, directed_graph=True):
    """ Computes Strongly Connected Components (SCC) and assigns cluster IDs to nodes in a directed graph.

    Parameters
    ===========
    G: nx.DiGraph
        A directed graph (DiGraph) in NetworkX.
    n_nodes: int
        The total number of nodes in the graph.

    Returns
    ========
    id_pred: torch.tensor
        A PyTorch tensor where each element represents the cluster ID of a node.
    n_components_pred: int
        The number of connected components in the graph.
    """

    # Compute connected components
    if directed_graph:
        scc_sets = [c for c in sorted(nx.strongly_connected_components(G), key=len, reverse=False)]
    else:
        scc_sets = [c for c in sorted(nx.connected_components(G), key=len, reverse=False)]

    # Initialize a list of connected components
    all_sets = list(scc_sets)

    # Add independent nodes to the list of connected components
    for i in range(n_nodes):
        is_included = any(i in s for s in all_sets)
        if not is_included:
            all_sets.append({i})

    # Initialize id_pred as a tensor of zeros
    id_pred = torch.zeros(n_nodes, dtype=int)
    
    # Assign cluster IDs
    cluster = 0
    for s in all_sets:
        for node in s:
            id_pred[node] = cluster
        cluster += 1

    # Calculate the number of components
    n_components_pred = len(all_sets)
    
    return id_pred, n_components_pred


def splitting(ID_pred, predictions, preds_prob, edge_list, num_nodes, predicted_act_edges, num_cameras, directed_graph):
    """ This function performs edge disjointing in a graph based on specified criteria.
    The criteria include identifying labels with more than 'num_cameras' elements and
    removing edges based on certain conditions such as minimum probability.
    The graph can be directed or undirected based on the 'directed_graph' parameter.

    The function may be called recursively for further edge disjointing.
    
    Parameters
    ----------
    ID_pred : torch.Tensor
        A tensor containing labels associated with nodes in the graph.
    predictions : torch.Tensor
        A binary tensor representing predictions for edges in the graph.
    preds_prob : torch.Tensor
        A tensor containing edge prediction probabilities.
    edge_list : list
        A list of edges in the graph.
    num_nodes : int
        The number of nodes in the graph.
    predicted_act_edges : list
        A list of currently active edges in the graph.
    num_cameras : int
        The threshold for the number of cameras in a label.
    directed_graph : bool
        A boolean indicating whether the graph is directed or undirected.

    Returns
    -------
    torch.Tensor
        A modified binary tensor of predictions after edge disjointing.

    """
    # Step 1: Identify labels (ID_pred) with more than 'num_cameras' elements
    label_ID_to_disjoint = torch.where(torch.bincount(ID_pred) > num_cameras)[0]

    # If there are labels to disjoint
    if len(label_ID_to_disjoint) >= 1:
        
        # Select the first label that meets the criteria
        l = label_ID_to_disjoint[0]
        flag_need_disjoint = True

        while flag_need_disjoint:
            # Step 2: Identify edges to be disjointed
            global_idx_new_predicted_active_edges = (predictions == 1).nonzero(as_tuple=True)[0]
            nodes_to_disjoint = torch.where(ID_pred == l)[0]

            # Identify active edges to disjoint
            idx_active_edges_to_disjoint = [pos for pos, n in enumerate(predicted_act_edges) if np.any(np.in1d(nodes_to_disjoint, n))]

            # Step 4: Remove edges based on minimum probability
            global_idx_edges_disjoint = global_idx_new_predicted_active_edges[torch.tensor(idx_active_edges_to_disjoint)]
            min_prob = torch.min(preds_prob[global_idx_edges_disjoint])
            global_idx_min_prob = torch.where(preds_prob == min_prob)[0]
            predictions[global_idx_min_prob] = 0

            # Step 5: Check if further disjointing is needed
            predicted_act_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos in torch.where(predictions == 1)[0]]
            G = nx.DiGraph(predicted_act_edges) if directed_graph else nx.Graph(predicted_act_edges)
            ID_pred, _ = connected_componnets(G, num_nodes, directed_graph)

            if np.bincount(ID_pred)[l] > num_cameras:
                flag_need_disjoint = True
            else:
                flag_need_disjoint = False

                # Step 6: Recursively call 'splitting' to further disjoint
                splitting(ID_pred, 
                          predictions, 
                          preds_prob, 
                          edge_list, 
                          num_nodes, 
                          predicted_act_edges,
                          num_cameras,
                          directed_graph)
    return predictions


def pruning(data, edges_out, probs, predicted_active_edges, num_cameras, directed_graph):
    """ Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        data: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        directed_graph: determines whether each edge in data.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    edge_ixs = data.edge_index
    if not directed_graph:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. 
    else:
        sorted = edge_ixs 
        div_factor = 1. 

    flag_rounding_needed = False
    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=data.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=data.num_nodes) / div_factor

    nodes_flow_out = torch.where(flow_out > (num_cameras-1))
    nodes_flow_in = torch.where(flow_in > (num_cameras-1))


    if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
        flag_rounding_needed = True
        new_predictions = edges_out.clone()
    else:
        new_predictions = []

    while flag_rounding_needed:
        edges_to_remove = []

        t = time.time()
        for n in nodes_flow_out[0]:

            pos = intersect(torch.where(edge_ixs[0] == n)[0], torch.where(new_predictions == 1)[0])
            remove_edge = pos[torch.argmin(probs[pos], dim=0)]

            edges_to_remove.append(remove_edge)

        for n in nodes_flow_in[0]:

            pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
            remove_edge = pos[torch.argmin(probs[pos], dim=0)]
            edges_to_remove.append(remove_edge)

        edges_to_remove = torch.stack(edges_to_remove)

        if edges_to_remove.shape[0] >= 1:
            new_predictions[edges_to_remove] = 0
        else:
            new_predictions = []

        flow_out = scatter_add(new_predictions, sorted[0], dim_size=data.num_nodes) / div_factor
        flow_in = scatter_add(new_predictions, sorted[1], dim_size=data.num_nodes) / div_factor

        nodes_flow_out = np.where(flow_out.cpu().numpy() > (num_cameras-1))
        nodes_flow_in = np.where(flow_in.cpu().numpy() > (num_cameras-1))

        if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
            flag_rounding_needed = True
        else:
            flag_rounding_needed = False

    return new_predictions


def remove_edges_single_direction(active_edges, predictions, edge_list):
    """ Remove edges in a single direction from active edges based on predictions.

    Parameters
    ----------
    active_edges : list of tuple
        List of active edges represented as tuples (node1, node2).
    predictions : torch.Tensor
        Binary predictions indicating whether each edge should be removed (1) or not (0).
    edge_list : tuple of np.ndarray
        Tuple containing two numpy arrays representing edges as (source_nodes, target_nodes).

    Returns
    -------
    torch.Tensor
        Updated predictions after removing edges.
    list of tuple
        List of predicted active edges after removing edges.
    """
    # Find indices of active edges to remove
    idx_active_edges_to_remove = [pos for pos, n in enumerate(active_edges) if (n[::-1] not in active_edges)]
    
    if not idx_active_edges_to_remove:
        # No edges to remove, return original predictions and active edges
        return predictions.clone(), active_edges

    # Find global indices of edges to remove
    predicted_active_edges_global_pos = [pos for pos, p in enumerate(predictions) if p == 1]
    global_idx_edges_to_remove = np.asarray(predicted_active_edges_global_pos)[np.asarray(idx_active_edges_to_remove)]

    # Clone predictions and set values to 0 for removed edges
    new_predictions = predictions.clone()
    new_predictions[global_idx_edges_to_remove] = 0

    # Create a list of new predicted active edges
    new_predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) 
                                  for pos, p in enumerate(new_predictions) if p == 1]

    return new_predictions, new_predicted_active_edges


def get_active_edges(edge_list, whole_edges_prediction):
    """ Get form the whole set of edges, the ones that are predicted 
    as active by the Graph Neural Network and deactivates the predicted active edges
    that are not bidirectional, since the graph should be undirected.

    Parameters
    ==========
    edge_list: torch.tensor
        The list of edges (u,v) where u and v are node ids.
    whole_edges_prediction: torch.tensor
        The predictions from the GNN. Should be 0, or 1
        and must have the same number of elements along the first axis as 
        edge_list.
    
    Returns
    ==========
    whole_edges_prediction: torch.tensor
        The new set of predictions after removing non-bidirectional edges
    pred_active_edges: torch.tensor
        The set of active edges after refining non-bidirectional ones
    """
    # Get initial set of predicted active edges
    pred_active_edges = [(edge_list[0][pos], edge_list[1][pos]) 
                               for pos, p in enumerate(whole_edges_prediction) if p == 1]
    
    return whole_edges_prediction, pred_active_edges


def postprocessing(num_cameras, 
                   preds_prob, 
                   whole_edges_prediction, 
                   edge_list, 
                   data,
                   directed_graph=True,
                   remove_unidirectional=False):
    """ Perform postprocessing procedure on the set of edge predictions.
    The procedures follow (pruning -> connected_components -> splitting -> connected components)
    """
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, whole_edges_prediction)

    if directed_graph and remove_unidirectional:
        whole_edges_prediction, pred_active_edges = remove_edges_single_direction(pred_active_edges,
                                                                                  whole_edges_prediction, 
                                                                                  edge_list)
    
    # Pruning edges that violates num_camera contraints
    predictions_r = pruning(data, 
                            whole_edges_prediction.view(-1), 
                            preds_prob[:,1], 
                            pred_active_edges, 
                            num_cameras, directed_graph)

    if len(predictions_r):
        whole_edges_prediction = predictions_r
    
    # Get set of predicted active edges that go both ways (no single direction)
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, 
                                                                 whole_edges_prediction)
    
    if directed_graph and remove_unidirectional:
        whole_edges_prediction, pred_active_edges = remove_edges_single_direction(pred_active_edges,
                                                                                  whole_edges_prediction, 
                                                                                  edge_list)
    
    # Get clusters of active edges. Each cluster represents an object id
    G = nx.DiGraph(pred_active_edges) if directed_graph else nx.Graph(pred_active_edges)
    id_pred, _ = connected_componnets(G, data.num_nodes, directed_graph)

    # Perform splitting for conencted components that present bridges
    whole_edges_prediction = splitting(id_pred, 
                                       whole_edges_prediction, 
                                       preds_prob[:,1], 
                                       edge_list, 
                                       data.num_nodes, 
                                       pred_active_edges,
                                       num_cameras, 
                                       directed_graph)

    # Get initial set of predicted active edges
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, 
                                                                 whole_edges_prediction)
    
    # Get clusters of active edges. Each cluster represents an object id
    G = nx.DiGraph(pred_active_edges) if directed_graph else nx.Graph(pred_active_edges)
    id_pred, _ = connected_componnets(G, data.num_nodes, directed_graph)

    return id_pred, torch.tensor(whole_edges_prediction)


def fix_annotation_frames(gt_df, pred_df):
    """ For a ground truth dataframe and a predictions dataframe
    both resulting form the concatenation of multiple dataframes from
    various cameras, change the frame index every time the camera changes
    to avoid colisions in the computation of performance metrics.

    Parameters
    ===========
    gt_df: pd.DataFrame
        The ground truth annotations (detections) for multiple-cameras
    pred_df: pd.DataFrame
        The predicitons dataframe.
    
    Returns
    ===========
    gt_df: pd.DataFrame
        The ground truth annotations (detections) for multiple-cameras
        with fixed frame numbers.
    pred_df: pd.DataFrame
        The predicitons dataframe with fixed frame numbers.
    """
    gt_cameras = gt_df.camera.unique()
    pred_cameras = pred_df.camera.unique()

    gt_camera_dfs = []
    pred_camera_dfs = []
    max_frame = 0 

    # sum frame_number every time a camera changes to avoid frame id colision
    for camera in gt_cameras:
        gt_camera = gt_df[gt_df['camera'] == camera]
        max_frame_cam = gt_camera['frame'].max()
        gt_camera.loc[:, 'frame'] += max_frame
        gt_camera_dfs.append(gt_camera)

        if camera in pred_cameras:
            pred_camera = pred_df[pred_df['camera'] == camera]
            max_frame_cam = pred_camera['frame'].max()
            pred_camera.loc[:, 'frame'] += max_frame
            pred_camera_dfs.append(pred_camera)

        max_frame += max_frame_cam

    return pd.concat(gt_camera_dfs), pd.concat(pred_camera_dfs)

def remove_dets_with_one_camera(df):
    """Remove objects that appear in only one camera.
    
    parameters
    ==========
    df : pd.DataFrame
        Data that should be filtered

    Returns
    ==========
    df : pd.DataFrame
        Filtered data with only objects that appear in 2 or more cameras.
    """
    # get unique CameraId/Id combinations, then count by Id
    cnt = df[['camera','id']].drop_duplicates()[['id']].groupby(['id']).size()
    # keep only those Ids with a camera count > 1
    keep = cnt[cnt > 1]
    
    # retrict the data to kept ids
    return df.loc[df['id'].isin(keep.index)].reset_index(drop=True)