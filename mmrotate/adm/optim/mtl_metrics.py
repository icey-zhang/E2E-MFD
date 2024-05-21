import torch


def compute_metrics(G):
    """
    Arguments:
        G (torch.Tensor): Matrix of shape TxN
    Returns:
        svals (list[float], T): Singular values
        cn (float): Condition number
        cos (torch.Tensor, TxT): Pair-wise task gradient cosine distance
        gms (torch.Tensor, TxT): Gradient magnitude similarity
                gms(i, j) = 2 ||g_i||_2 ||g_j||_2 / (||g_i||^2 + ||g_j||^2)
        cbm (torch.Tensor, TxT): MTL Curvature Bounding Measure
                cmb(i, j) = (1 - cos^2 (<g_i, g_j>)) * ||g_i-g_j||^2 / ||g_i + g_j||^2
        gn (torch.Tensor, Tx1): Per-task gradient norms
    """
    assert len(G.shape) == 2
    if G.shape[0] > G.shape[1]:
        G = G.T

    t, n = G.shape[0], G.shape[1]

    result = {}
    S = torch.linalg.svdvals(G)
    result['svals'] = S.cpu().numpy()
    result['cn'] = (S.max() / S[S > 0].min()).item()

    gradnorms = torch.linalg.norm(G, dim=1)  # [T, ]
    Gn = G / gradnorms.unsqueeze(1)  # [T, N]

    cos_angles = Gn.matmul(Gn.T)
    dom_denom = gradnorms.view(-1, 1).pow(2) + gradnorms.view(1, -1).pow(2)
    dominance = 2 * gradnorms.view(-1, 1) * gradnorms.view(1, -1) / dom_denom

    sum_norm2 = torch.linalg.norm(G.view(t, 1, -1) + G.view(1, t, -1), dim=-1).pow(2)
    diff_norm2 = torch.linalg.norm(G.view(t, 1, -1) - G.view(1, t, -1), dim=-1).pow(2)
    curvature = (1 - cos_angles.pow(2)) * diff_norm2 / sum_norm2

    result['cos'] = cos_angles.cpu().numpy()
    result['gms'] = dominance.cpu().numpy()
    result['cbm'] = curvature.cpu().numpy()
    result['gn'] = gradnorms.cpu().numpy()
    result['totalgn'] = torch.linalg.norm(G.sum(0)).item()

    return result
