import time
import torch
import numpy as np
import networkx as nx
import os.path as osp
import cv2
import pandas as pd
from torch_scatter import scatter_add


def intersect(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


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


def split_clusters(G, ID_pred, predictions, preds_prob, num_cameras, num_nodes, directed_graph=True):
    """ Split clusters of nodes in a graph based on a specified condition.

    Parameters
    ==========
    G : networkx.DiGraph
        An updated directed graph.
    ID_pred : torch.Tensor
        A tensor containing cluster IDs for nodes.
    predictions : torch.Tensor
        A tensor representing edge predictions.
    preds_prob : torch.Tensor
        A tensor containing edge prediction probabilities.
    num_cameras : int
        The threshold number of nodes in a cluster to trigger splitting.

    Returns
    ========
    torch.Tensor
        The modified edge predictions after splitting clusters.
    """

    # Find cluster IDs with more than num_cameras nodes
    cluster_sizes = torch.bincount(ID_pred)
    clusters_to_disjoint = torch.nonzero(cluster_sizes > num_cameras).flatten()

    for cluster_id in clusters_to_disjoint:
        # Find nodes in the cluster
        nodes_to_disjoint = torch.where(ID_pred == cluster_id)[0]

        # Find active edges involving nodes in the cluster
        active_edges_to_disjoint = torch.nonzero(
            torch.any(torch.isin(torch.tensor(list(G.edges())), nodes_to_disjoint), dim=1)
        ).flatten()

        if len(active_edges_to_disjoint) > 0:
            # Find edge with minimum probability among active edges
            min_prob_edge = torch.argmin(preds_prob[active_edges_to_disjoint])

            # Set the prediction for the edge with minimum probability to 0
            predictions[active_edges_to_disjoint[min_prob_edge]] = 0

            # Update the graph G based on remaining active edges
            remaining_active_edges = [
                (edge[0], edge[1]) for edge in G.edges()
            ]
            G = nx.DiGraph(remaining_active_edges)

            # Recursively split clusters
            ID_pred, _ = connected_componnets(G, num_nodes, directed_graph)
    
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
            remove_edge = pos[torch.argmin(probs[pos])]

            edges_to_remove.append(remove_edge)

        for n in nodes_flow_in[0]:

            pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
            remove_edge = pos[torch.argmin(probs[pos])]
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


def get_active_edges(edge_list, whole_edges_prediction, directed_graph=True):
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
    
    if directed_graph:
        whole_edges_prediction, pred_active_edges = remove_edges_single_direction(pred_active_edges,
                                                                                whole_edges_prediction, 
                                                                                edge_list)
    
    return whole_edges_prediction, pred_active_edges


def postprocessing(num_cameras, 
                   preds_prob, 
                   whole_edges_prediction, 
                   edge_list, 
                   data,
                   directed_graph=True):
    """ Perform postprocessing procedure on the set of edge predictions.
    The procedures follow (pruning -> connected_components -> splitting -> connected components)
    """
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, whole_edges_prediction, directed_graph)
    
    # Pruning edges that violates num_camera contraints
    predictions_r = pruning(data, whole_edges_prediction.view(-1), preds_prob[:,1], pred_active_edges, num_cameras)

    if len(predictions_r):
        whole_edges_prediction = predictions_r
    
    # Get set of predicted active edges that go both ways (no single direction)
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, whole_edges_prediction, directed_graph)
    
    # Get clusters of active edges. Each cluster represents an object id
    G = nx.DiGraph(pred_active_edges)
    id_pred, _ = connected_componnets(G, data.num_nodes, directed_graph)

    # Perform splitting for conencted components that present bridges
    whole_edges_prediction = split_clusters(G, id_pred, whole_edges_prediction, preds_prob[:,1], num_cameras, data.num_nodes)

    # Get initial set of predicted active edges
    whole_edges_prediction, pred_active_edges = get_active_edges(edge_list, whole_edges_prediction, directed_graph)
    
    # Get clusters of active edges. Each cluster represents an object id
    G = nx.DiGraph(pred_active_edges)
    id_pred, _ = connected_componnets(G, data.num_nodes, directed_graph)

    return id_pred, torch.tensor(whole_edges_prediction)



def load_roi(sequence_path, camera_name, sequence_name=None):
    """ Load Region of Interest binary maps for each
    camera.

    Parameters
    ==========
    sequence_path: str
        The path to the processed dataset.
        Example: mcmt_gnn/datasets/AIC22
    camera_name: str
        The camera name. Ex: c001
    sequence_name: str
        The name of the sequence. Ex: S01
    """
    # If sequence_name is not provided, we assume that it is embedded in sequence_path
    if sequence_name:
        roi_dir = osp.join(sequence_path, "roi", sequence_name, camera_name, 'roi.jpg')
    else:
        roi_dir = osp.join(sequence_path, camera_name, 'roi.jpg')

    if not osp.exists(roi_dir):
        raise ValueError(f"Missing ROI image for camera {camera_name}")
    
    # Load grayscale image and convert to binary image
    binary_image = cv2.imread(roi_dir, cv2.IMREAD_GRAYSCALE)
    binary_image = (binary_image == 255)

    # Transpose if scene is in portrait
    if binary_image.shape[0] > binary_image.shape[1]:
        binary_image = binary_image.T

    return binary_image
    

def is_outside_roi(row, roi):
    """ Check if detection is inside region of interest.
    Return False if not.

    Parameter
    =========
    row: 
        The dataframe row that represents a detection and must have
        (xmin, xmax, ymin, ymax) columns.
    roi: np.array (w,h)
        The region of interest image as a binary map.
    
    Returns
    =========
    boolean
        True if detection is inside region of interest.
        False otherwise.
    """
    height = roi.shape[0]
    width = roi.shape[1]

    # If out of bounds
    if (row['xmin'] > width or row['ymin'] > height or
        row['xmin'] < 0 or row['ymin'] < 0):
        return True
    
    # roi[row['ymin'], row['xmin']] == true means it is inside roi 
    return not roi[row['ymin'], row['xmin']]


def remove_non_roi(sequence_path, data_df):
    """ Remove detections that correspond to areas
    outside of the Region of Interest (RoI).

    Parameters
    ==========
    data_df: pd.DataFrame
        The predicitons dataframe.

    Returns
    ==========
    pd.DataFrame
        The predictions dataframe with non-roi detections
        eliminated.
    """
    # Initialize outlier column as all false
    data_df['outlier'] = False

    # Check if detections are outlier and mark them
    for i, row in data_df.iterrows():
        roi = load_roi(sequence_path, row['camera'], row['sequence_name'])
        if is_outside_roi(row, roi):
            data_df.loc[i, 'outlier'] = True

    # Remove outlier detections
    return data_df[~data_df['outlier']].drop(columns=['outlier'])


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


def filter_dets_outside_frame_bounds(det_df, frame_width, frame_height, boundary_percentage=0.05):
    """Filter bounding boxes outside frame boundaries.
    This function filters the rows in the input DataFrame (det_df) based on whether
    the bounding boxes are within the specified frame boundaries. Bounding boxes that
    fall outside the frame boundaries (with a given margin) are removed from the output.

    Parameters
    ==========
    det_df : pandas.DataFrame
        DataFrame containing bounding box information, including 'xmin', 'ymin',
        'width', and 'height' columns.

    frame_width : int
        The width of the frame.

    frame_height : int
        The height of the frame.

    boundary_percentage : float, optional
        Margin as a percentage of the frame dimensions (default is 0.05, representing
        a 5% margin).

    Returns
    ==========
    pandas.DataFrame
        A filtered DataFrame with only the rows containing bounding boxes within the
        specified frame boundaries.

    Example
    ==========
    >>> filtered_df = filter_dets_outside_frame_bounds(bounding_boxes_df, 1920, 1080)
    """
    # Check if bounding box x-axis is inside the frame with a margin
    cond1 = (frame_width * boundary_percentage <= det_df['xmin'] + (det_df['width'] / 2))
    cond2 = det_df['xmin'] + (det_df['width'] / 2) <= frame_width - (frame_width * boundary_percentage)
    idx_x = np.logical_and(cond1, cond2)

    # Check if bounding box y-axis is inside the frame with a margin
    cond1 = (frame_height * boundary_percentage <= det_df['ymin'] + det_df['height'])
    cond2 = det_df['ymin'] + det_df['height'] <= frame_height - frame_height * boundary_percentage
    idx_h = np.logical_and(cond1, cond2)

    # Combine both conditions to filter the DataFrame
    idx = np.logical_and(idx_x, idx_h)

    return det_df[idx]