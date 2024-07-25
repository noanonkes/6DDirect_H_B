"""
This code is based on Dynamic 3D Gaze from Afar:
    https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar
"""

import torch


def get_rotation(vec1, vec2):
    """
    compute the rotation matrix to align vec1 to vec2 (vec1, vec2)
    vec1, vec2: (n_batch x 3)
    return: rotation matrix (n_batch x 3 x 3)
    """

    n_batch = vec1.shape[0]

    v = torch.cross(vec1, vec2, dim=1)
    s = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    c = torch.sum(vec1 * vec2, dim=1, keepdims=True)
    v_skew = v.new_full((n_batch, 3, 3), 0)
    v_skew[:, 0, 1] = -v[:, 2]
    v_skew[:, 0, 2] = v[:, 1]
    v_skew[:, 1, 0] = v[:, 2]
    v_skew[:, 1, 2] = -v[:, 0]
    v_skew[:, 2, 0] = -v[:, 1]
    v_skew[:, 2, 1] = v[:, 0]
    R = torch.eye(3, dtype=v.dtype, device=v.device).repeat(n_batch, 1, 1) + v_skew + \
        (v_skew @ v_skew) * ((1 - c)/s**2).unsqueeze(-1)

    return R


def compute_mae(vec1, vec2):
    """
    vec1, vec2 is torch.Tensor
    """

    vec1 = vec1.reshape(-1, vec1.shape[-1])
    vec2 = vec2.reshape(-1, vec2.shape[-1])

    if vec2.shape[-1] == 2 and vec1.shape[-1] == 3:
        vec1 = vec1[..., :2] / torch.norm(vec1[..., :2], dim=-1, keepdim=True)
    if vec2.shape[-1] == 3 and vec1.shape[-1] == 2:
        vec2 = vec2[..., :2] / torch.norm(vec2[..., :2], dim=-1, keepdim=True)

    cos = torch.sum(vec1 * vec2, dim=1)
    cos[cos > 1] = 1
    cos[cos < -1] = -1
    rad = torch.acos(cos)
    mae = torch.rad2deg(torch.mean(rad).cpu().detach())

    return mae