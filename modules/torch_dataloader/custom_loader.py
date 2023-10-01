from torch_geometric.data import DataLoader
import torch

from torch_geometric.data import DataLoader
import torch

class NodeSamplingDataLoader(DataLoader):
    """
    DataLoader for sampling nodes from graphs in a dataset.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        The dataset to sample from.
    batch_size : int, optional
        Batch size for data loading. (default: 1)
    shuffle : bool, optional
        Whether to shuffle the data. (default: True)
    num_samples_per_node : int, optional
        Number of nodes to sample per graph. (default: 10)
    **kwargs
        Additional keyword arguments passed to DataLoader constructor.

    Attributes
    ----------
    num_samples_per_node : int
        Number of nodes to sample per graph.

    Notes
    -----
    This DataLoader is designed for sampling nodes from graphs within a given dataset.
    It allows you to control the number of nodes to sample per graph and supports sampling
    without replacement.

    Example
    -------
    >>> dataset = YourCustomDataset()
    >>> loader = NodeSamplingDataLoader(dataset, batch_size=32, num_samples_per_node=20)
    >>> for batch in loader:
    >>>     # Process the sampled nodes in each batch.

    """
    def __init__(self, dataset, batch_size=1, shuffle=True, num_samples_per_node=10, **kwargs):
        self.num_samples_per_node = num_samples_per_node
        super().__init__(dataset, batch_size, shuffle, **kwargs)

    def collate(self, data_list):
        """
        Collates and samples nodes from graphs in the batch.

        Parameters
        ----------
        data_list : List[torch_geometric.data.Data]
            A list of Data objects representing graphs.

        Returns
        -------
        List[torch_geometric.data.Data]
            A list of sampled Data objects.

        """
        batch = []
        for data in data_list:
            num_nodes = data.num_nodes
            if num_nodes <= self.num_samples_per_node:
                # If the graph has fewer nodes than requested samples,
                # sample all nodes without replacement.
                sampled_nodes = torch.randperm(num_nodes)
            else:
                # Sample nodes without replacement.
                sampled_nodes = torch.randperm(num_nodes)[:self.num_samples_per_node]

            # Create a subgraph with the sampled nodes
            sampled_data = data.subgraph(sampled_nodes)
            batch.append(sampled_data)

        return batch

    

class ObjectSamplingDataLoader(DataLoader):
    """DataLoader for sampling objects from graphs in a dataset.

    Parameters
    ==========
    dataset : torch_geometric.data.Dataset
        The dataset to sample from.
    batch_size : int, optional
        Batch size for data loading. (default: 1)
    shuffle : bool, optional
        Whether to shuffle the data. (default: True)
    num_object_ids_per_graph : int, optional
        Number of objects to sample per graph. (default: 100)
    **kwargs
        Additional keyword arguments passed to DataLoader constructor.

    Attributes
    ==========
    num_object_ids_per_graph : int
        Number of objects to sample per graph.

    Notes
    ==========
    This DataLoader is designed for sampling objects (e.g., nodes or entities) from graphs
    within a given dataset. It allows you to control the number of objects to sample per graph.

    Example
    ==========
    >>> dataset = YourCustomDataset()
    >>> loader = ObjectSamplingDataLoader(dataset, batch_size=32, num_object_ids_per_graph=50)
    >>> for batch in loader:
    >>>     # Process the sampled objects in each batch.

    """

    def __init__(self, dataset, batch_size=1, shuffle=True, num_object_ids_per_graph=100, **kwargs):
        self.num_object_ids_per_graph = num_object_ids_per_graph
        super().__init__(dataset, batch_size, shuffle, **kwargs)

    def collate(self, data_list):
        """Collates and samples objects from graphs in the batch.
        This method samples a specified number of objects (entities using the objext_ids, not nodes) 
        from each graph in the batch and returns a list of sampled Data objects.

        Parameters
        ===========
        data_list : List[torch_geometric.data.Data]
            A list of Data objects representing graphs.

        Returns
        ===========
        List[torch_geometric.data.Data]
            A list of sampled Data objects.

        """
        batch = []
        for data in data_list:
            object_ids = data.y
            unique_ids = torch.unique(object_ids)

            # Sample only if the number of objects is more than or equal to num_object_ids_per_graph
            if len(unique_ids) >= self.num_object_ids_per_graph:
                sampled_idx = torch.randperm(len(unique_ids))[:self.num_object_ids_per_graph]
                sampled_ids = unique_ids[sampled_idx]
                node_indices = torch.arange(data.num_nodes)
                sampled_nodes = node_indices[torch.isin(object_ids, sampled_ids)]

                # Create a subgraph with the sampled nodes
                sampled_data = data.subgraph(sampled_nodes)
                batch.append(sampled_data)
            
            # Otherwise do nothing
            else:
                batch.append(data)

        return batch
    


class NeighborSamplingDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_samples_per_node=10, **kwargs):
        self.num_samples_per_node = num_samples_per_node
        super(NeighborSamplingDataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)

    def collate(self, data_list):
        # Sample neighbors for each node in the batch
        batch = []
        for data in data_list:
            sampled_nodes = torch.randint(0, data.num_nodes, (self.num_samples_per_node,))
            sampled_subgraph = self.sample_neighbors(data, sampled_nodes)
            batch.append(sampled_subgraph)
        
        return batch

    def sample_neighbors(self, data, nodes):
        # Create a subgraph with sampled neighbors for the specified nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[nodes] = 1
        sampled_edge_index, sampled_edge_attr = self.sample_adjacency(edge_index, edge_attr, mask)
        
        sampled_data = data.__class__(
            x=data.x[nodes],
            edge_index=sampled_edge_index,
            edge_attr=sampled_edge_attr,
            y=data.y,
        )
        
        return sampled_data

    def sample_adjacency(self, edge_index, edge_attr, mask):
        # Sample adjacency for the specified nodes
        mask = mask[edge_index[0]]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        return edge_index, edge_attr