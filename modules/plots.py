import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils.convert import to_networkx
from sklearn.manifold import TSNE

def draw_pyg_network(data, 
                     fig_title,
                     node_labels=None,
                     edge_labels=None,
                     class_ids=None, 
                     color_nodes=True, 
                     layout='circular', 
                     undirected=True,
                     save_path:str=None,
                     figsize=(15,10),
                     show=True):
    """ Visualize a PyTorch Geometric graph using NetworkX and Matplotlib.
    This function creates a visual representation of the input graph using NetworkX and Matplotlib. It supports
    optional filtering of edges based on their labels and node coloring based on node classes.

    - If 'class_ids' is provided, only edges with labels in 'class_ids' will be displayed.
    - If 'color_nodes' is set to True, nodes will be colored based on their class labels using a color map.

    Parameters
    ==========
    data : torch_geometric.data.Data
        The PyTorch Geometric data object representing the graph.
    node_labels: np.ndarray, optional
        A 1D numpy array with node labels.
    edge_labels: np.ndarray, optional
        A 1D numpy array with edge labels.
    class_ids : list or None, optional
        List of edge labels to include (if specified). Default is None.
    color_nodes : bool, optional
        Whether to color nodes based on their class (if available). Default is True.
    layout : str, optional
        Layout algorithm for node positioning ('circular' or 'spring'). Default is 'circular'.
    save_path: string, optional
        If not None, then the figure will be saved on that directory.
    Returns
    ==========
    None
    """
    fig = plt.figure(figsize=figsize)

    # Nodes definitions
    node_labels = node_labels if node_labels is not None else data.y.tolist()

    # Edge labels and indices definition
    edges_ind = data.edge_index.T.numpy()
    el = edge_labels if edge_labels is not None else data.edge_labels.numpy()

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
                           node_color=node_labels if color_nodes else None, 
                           cmap=plt.cm.gist_ncar,
                           label=node_labels)
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

    plt.title(fig_title)

    if save_path:
        plt.savefig(save_path)

    if show: plt.show()

    plt.close(fig)
    return fig


def tsne2d_scatterplot(embeddings, labels, fig_title, save_path:str=None, show=True):
    """ Computes a 2D embedding using T-SNE
    and saves the corresponding scatterplot.

    Parameters
    ==========
    embeddings: torch.tensor
        An embedings matrix
    labels: torch.tensor
        A labels vector
    save_path: string, optional
        The path where the figure will be saved.
    """
    fig = plt.figure(figsize=(15, 10))
    x_2 = TSNE().fit_transform(embeddings)
    plt.scatter(x_2[:,0], x_2[:,1], c=labels)

    plt.title(fig_title)

    if save_path:
        plt.savefig(save_path)
    if show: plt.show()

    plt.close(fig)
    return fig


def plot_tsne_samples(graph, node_feats, num_subjects=[10, 20, 30, 40, 50]):
    """ Perform multiple plots of nodes embeddings at varying number of IDs.

    Parameters 
    ==========
    graph: pytorch_geometric.Dataset
        The graph object
    node_feats: torch.tensor
        Node embeddings
    num_subjects: list[int]
        A list with different number of IDs. This function will produce as much figures
        as elements in this list
    
    Returns
    ==========
    list[tuple[plt.Figure, int]]
        A list of figures representing the node embeddings on a low dimensional space.
    """
    figures = []
    for num_subject in num_subjects:
        tsne = tsne2d_scatterplot(
            node_feats[torch.isin(graph.y, torch.unique(graph.y)[:num_subject])].cpu(), 
            graph.y[torch.isin(graph.y, torch.unique(graph.y)[:num_subject])].cpu(), 
            fig_title=f"TSNE 2D Plot of Node Embeddings (sample of {num_subject} IDs)",
            save_path=None, show=False
            )
        figures.append((tsne, num_subject))
    
    return figures


def plot_histogram_wrapper(vector, 
                           xlabel,
                           ylabel, 
                           title, 
                           bins=10, 
                           save_path=None, 
                           show=True):
    """ Given a torch.tensor
    compute the histogram and kernel density
    estimation.

    Parameters
    ==========
    vector : torch.Tensor
        Input tensor for which the histogram and kernel 
        density estimate will be computed.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    bins : int, optional
        Number of bins for the histogram. Default is 10.
    save_path : str, optional
        If provided, the plot will be saved to the specified file path.
    show : bool, optional
        If True, the plot will be displayed. 
        If False, the plot will not be displayed. Default is True.

    Returns
    ==========
    matplotlib.figure.Figure
        The matplotlib Figure object representing the generated plot.

    """
    vector = vector.cpu().numpy()

    fig = plt.figure(figsize=(15, 10))

    # Plot histogram
    plt.hist(vector, bins=bins, 
             density=True, alpha=0.5, 
             color='b', edgecolor='black')

    # Plot kernel density estimate using seaborn
    sns.kdeplot(vector, color='r')

    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    if show: plt.show()

    plt.close(fig)
    return fig