import numpy as np
import cv2
from numpy import cos, sin

import torch
from pytorch3d.transforms import matrix_to_rotation_6d


def get_rotation(vec1, vec2):
    """
    Compute the rotation matrix to align vec1 to vec2.
    vec1, vec2: (n_batch x 3)
    return: rotation matrix (n_batch x 3 x 3)
    """
    n_batch = vec1.shape[0]

    # Compute cross product
    v = np.cross(vec1, vec2, axis=1)

    # Compute the sine and cosine of the angle between vec1 and vec2
    s = np.linalg.norm(v, axis=1, keepdims=True)
    c = np.sum(vec1 * vec2, axis=1, keepdims=True)

    # Create the skew-symmetric matrix of v
    v_skew = np.zeros((n_batch, 3, 3))
    v_skew[:, 0, 1] = -v[:, 2]
    v_skew[:, 0, 2] = v[:, 1]
    v_skew[:, 1, 0] = v[:, 2]
    v_skew[:, 1, 2] = -v[:, 0]
    v_skew[:, 2, 0] = -v[:, 1]
    v_skew[:, 2, 1] = v[:, 0]

    # Compute the rotation matrix
    eye = np.eye(3).reshape((1, 3, 3))  # Identity matrix
    R = eye + v_skew + (v_skew @ v_skew) * ((1 - c) / (s**2)).reshape((n_batch, 1, 1))

    return R


def vecs_to_matrix(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.

    From https://gist.github.com/aormorningstar/3e5dda91f155d7919ef6256cb057ceee
    """
    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    # dimension of the space and identity
    dim = u.size
    I = np.identity(dim)
    # the cos angle between the vectors
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)


def get_rot_reps(vec):

    # This vector is "looking" to the camera
    reverse_e3 = np.array([[0.0, 0.0, -1.0]])

    vec = np.array([vec])
    vec = vec / np.linalg.norm(vec)

    # Rot mat that transforms [0, 0, -1] to input vec
    R = get_rotation(reverse_e3, vec)[0]

    # Take first 2 rows
    sixD = matrix_to_rotation_6d(torch.tensor(R))

    # Extract euler angles from R
    _, [pitch, yaw, roll] = select_euler(
        np.rad2deg(inverse_rotate_zyx(R)), pred=True  # inverse rotation in order of ZYX
    )
    yaw = -yaw
    roll = -roll

    return sixD.tolist(), [pitch, yaw, roll]


def normalize_angle(angle):
    """Normalize angle to be within the range (-180, 180] degrees."""
    if angle > 180.:
        return angle - 360.
    elif angle <= -180.:
        return angle + 360.
    return angle


def select_euler(two_sets, pred=False):
    """
    By accepting two sets of Euler angles, the function provides 
    flexibility in handling cases where gimbal lock might occur. 
    It can then decide which set of Euler angles to use based on 
    additional criteria (not exceeding +90 degrees).
    """
    pitch, yaw, roll = two_sets[0]
    pitch2, yaw2, roll2 = two_sets[1]

    yaw = normalize_angle(yaw)
    yaw2 = normalize_angle(yaw2)
    pitch = normalize_angle(pitch)
    pitch2 = normalize_angle(pitch2)
    roll = normalize_angle(roll)
    roll2 = normalize_angle(roll2)
    
    if abs(roll) < 90 and abs(pitch) < 90:
        return True, [pitch, yaw, roll]
    elif abs(roll2) < 90 and abs(pitch2) < 90:
        return True, [pitch2, yaw2, roll2]
    elif not pred:
        return False, [-999, -999, -999]
    else:
        if abs(roll2) < abs(roll) and abs(pitch2) < abs(pitch):
            return True, [pitch2, yaw2, roll2]
        else:
            return True, [pitch, yaw, roll]


def inverse_rotate_zyx(M):
    if np.linalg.norm(M[:3, :3].T @ M[:3, :3] - np.eye(3)) > 1e-1:
        print(M)
        raise ValueError('Matrix is not a rotation')

    if np.abs(M[0, 2]) > 0.9999999:
        # gimbal lock
        z = 0.0
        # M[1,0] =  cz*sx*sy
        # M[2,0] =  cx*cz*sy
        if M[0, 2] > 0:
            y = -np.pi / 2
            x = np.arctan2(-M[1, 0], -M[2, 0])
        else:
            y = np.pi / 2
            x = np.arctan2(M[1, 0], M[2, 0])
        return np.array((x, y, z)), np.array((x, y, z))
    else:
        # no gimbal lock
        y0 = np.arcsin(-M[0, 2])
        y1 = np.pi - y0
        cy0 = np.cos(y0)
        cy1 = np.cos(y1)

        x0 = np.arctan2(M[1, 2] / cy0, M[2, 2] / cy0)
        x1 = np.arctan2(M[1, 2] / cy1, M[2, 2] / cy1)

        z0 = np.arctan2(M[0, 1] / cy0, M[0, 0] / cy0)
        z1 = np.arctan2(M[0, 1] / cy1, M[0, 0] / cy1)
        return np.array((x0, y0, z0)), np.array((x1, y1, z1))