import torch

def intersect(tensor1, tensor2):
    """ Compute the intersection of two PyTorch tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first input tensor.
    tensor2 : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        A tensor containing the intersection of elements 
        between 'tensor1' and 'tensor2'.

    Examples
    --------
    >>> tensor1 = torch.tensor([1, 2, 3, 4, 5])
    >>> tensor2 = torch.tensor([3, 4, 5, 6, 7])
    >>> result = intersect(tensor1, tensor2)
    """
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def torch_isin(input, test_elements):
    """ Check for the presence of elements from 'test_elements' 
    in a PyTorch tensor 'input' along a specified dimension.

    Parameters
    ==========
    input : torch.Tensor
        The input tensor to search for elements.
    test_elements : torch.Tensor
        The elements to check for in 'input'.

    Returns
    ==========
    torch.Tensor
        A boolean tensor indicating if any of the elements in 'test_elements' are present
        along the specified dimension of 'input'. 'True' indicates presence, 'False' indicates absence.

    Examples
    ==========
    >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> test_elements = torch.tensor([3, 5, 7])
    >>> dim_to_check = 1  # Check along the rows
    >>> result = torch_isin(input_tensor, test_elements, dim=dim_to_check)
    """
    # Create a boolean mask of the same shape as 'input'
    mask = torch.zeros_like(input, dtype=torch.bool)

    for element in test_elements:
        mask = mask | (input == element)

    return mask