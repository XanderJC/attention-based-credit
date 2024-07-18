import torch


def get_attention_distribution(
    response: torch.Tensor, query: torch.Tensor, attention: torch.Tensor
) -> torch.Tensor:
    """
    Compute the ABC attention map for a given response and query.

    Args:
        response (torch.Tensor): The response tensor.
        query (torch.Tensor): The query tensor.
        attention (torch.Tensor): The attention tensor.

    Returns:
        torch.Tensor: The attention distribution.

    """
    attention = attention.squeeze().cpu().detach().numpy()
    attention_matrix = attention[
        len(query) : len(query) + len(response),
        len(query) : len(query) + len(response),
    ]
    attention_map = attention_matrix[-1, :]  # / attention_matrix[-1, :].sum()

    out = torch.zeros_like(response, dtype=float)
    if len(attention_map) == len(out):
        out += torch.tensor(attention_map, device=out.device)
    elif len(attention_map) < len(out):
        out[len(out) - len(attention_map) :] += torch.tensor(
            attention_map, device=out.device
        )
    else:
        out += torch.tensor(
            attention_map[len(attention_map) - len(out) :], device=out.device
        )

    return (out / out.sum()).detach()


def get_generator_attention_distribution(
    response: torch.Tensor,
    query: torch.Tensor,
    attention: torch.Tensor,
    last_only: bool = False,
) -> torch.Tensor:
    """
    Compute the ABC-D attention distribution for a generator given a response and query.

    Args:
        response (torch.Tensor): The response tensor.
        query (torch.Tensor): The query tensor.
        attention (torch.Tensor): The attention tensor.
        last_only (bool, optional): Whether to consider only the last tokens attention
        map, otherwise weighted average over generation. Defaults to False.

    Returns:
        torch.Tensor: The attention distribution.

    """
    out = torch.zeros_like(response, dtype=float)

    if last_only:
        attention_map = attention[-1][-1].squeeze().mean(0)[len(query) :]
        attention_map = torch.cat(
            (torch.zeros(1, device=attention_map.device), attention_map), 0
        )

    else:
        attention_matrix = torch.zeros((len(response), len(response)))

        for i, token_att in enumerate(attention[1:]):
            att_map = token_att[-1].squeeze().mean(0)[len(query) :]
            attention_matrix[i + 1, 1 : len(att_map) + 1] = att_map

        weight = torch.nan_to_num(1 / (attention_matrix != 0).sum(axis=0), posinf=0)
        attention_map = (attention_matrix * weight).sum(axis=0)

    if len(attention_map) == len(out):
        out += torch.tensor(attention_map, device=out.device)
    elif len(attention_map) < len(out):
        out[len(out) - len(attention_map) :] += torch.tensor(
            attention_map, device=out.device
        )
    else:
        out += torch.tensor(
            attention_map[len(attention_map) - len(out) :], device=out.device
        )

    return (out / out.sum()).detach()
