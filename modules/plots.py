import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from sklearn.manifold import TSNE

def draw_pyg_network(data, 
                     class_ids=None, 
                     color_nodes=True, 
                     layout='circular', 
                     undirected=True,
                     save_path:str=None):
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
    save_path: string, optional
        If not None, then the figure will be saved on that directory.
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
                           cmap=plt.cm.gist_ncar,
                           label=data.y.tolist())
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

    if save_path:
        plt.savefig(save_path)

    plt.show()


def tsne2d_scatterplot(embeddings, labels, save_path:str=None):
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
    x_2 = TSNE().fit_transform(embeddings)
    plt.scatter(x_2[:,0], x_2[:,1], c=labels)

    if save_path:
        plt.savefig(save_path)
    plt.plot()