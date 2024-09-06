import torch


def list_to_tensor(
    tensor_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of arbitrary-length tensors to a single tensor.

    Args:
    ----
        tensor_list: List of tensors to pack into a single tensor.

    Returns:
    -------
    A stacked tensor and the respective tensor sizes along the first dimension.
    """
    tensor_sizes = torch.zeros(len(tensor_list), dtype=torch.int64, device=tensor_list[0].device)
    for i, y in enumerate(tensor_list):
        tensor_sizes[i] = len(y)
    tensor_packed = torch.cat(tensor_list, dim=0)
    return tensor_packed, tensor_sizes


def tensor_to_list(tensor: torch.Tensor, split_sizes: torch.Tensor) -> list[torch.Tensor]:
    """Split a tensor into a list based on given tensor sizes.

    Args:
    ----
        tensor: Tensor to split into parts.
        split_sizes: Sizes of the tensors to split along the first dimension.

    Raises:
    ------
    ValueError: When the sum of `split_sizes` does not match the first dimension of `tensor`.
    """
    if torch.sum(split_sizes) != tensor.shape[0]:
        raise ValueError(
            "The sum of `split_sizes` must match the total number of entries for `tensor`."
        )
    return list(tensor.split(tuple(split_sizes.tolist()), dim=0))
