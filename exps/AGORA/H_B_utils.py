
import math
import cv2
import numpy as np
from scipy.spatial import Delaunay
from math import cos, sin, pi


def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
    # Y-Axis pointing to down. drawn in green    
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)

    return img


def plot_cuboid_Zaxis_by_euler_angles(img, yaw, pitch, roll, tdx=None, tdy=None, size=100., thickness=3, looking_away=False):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        if abs(yaw) < 90:
            face_x = tdx - 0.5 * size
            face_y = tdy - 0.5 * size
        else:
            face_x = tdx + 0.5 * size
            face_y = tdy - 0.5 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    # X-Axis (pointing to left) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    if looking_away:
        # Draw top in green
        cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),thickness)
        cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),thickness)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),thickness)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),thickness)
        
        # Draw pillars in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness-1)
        cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),thickness-1)
        cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),thickness-1)
        cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),thickness-1)
        
        # Draw base in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)), (0,0,255), thickness-1)
        cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)), (0,0,255), thickness-1)
        cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)), (0,0,255), thickness-1)
        cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)), (0,0,255), thickness-1)
    else:
        # Draw base in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)), (0,0,255), thickness)
        cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)), (0,0,255), thickness)
        cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)), (0,0,255), thickness)
        cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)), (0,0,255), thickness)
        
        # Draw pillars in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness-1)
        cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),thickness-1)
        cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),thickness-1)
        cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),thickness-1)
        
        # Draw top in green
        cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),thickness-1)
        cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),thickness-1)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),thickness-1)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),thickness-1)

    return img


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    # This step applies the extrinsic transformation (rotation R and translation t)
    # to the 3D points X. It results in the coordinates of the points in the camera
    # coordinate system
    x = np.asarray(R * X + t)

    # Normalization the coordinates by dividing by the third (homogeneous) coordinate
    x[0:2, :] = x[0:2, :] / x[2, :]

    # Radial distortion correction
    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    # Calculates the radial distance of each point from the image center
    # And corrects for radial distortion using the distortion parameters Kd
    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
        r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
        r + 2 * x[1, :] * x[1, :])

    # Applies the intrinsic transformation using the camera matrix K to convert the normalized
    # image coordinates to pixel coordinates
    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    # Returns projected 2D points
    return x


def align_3d(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    https://github.com/raulmur/evaluate_ate_scale

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)


    rot: The rotation matrix (3x3) that represents the transformation needed to align the model points with the data points.

    trans: The translation vector (3x1) that represents the shift needed to align the model points with the data points.

    trans_error: Translational error per point (1xn), indicating how much each point is shifted after the alignment.

    s: A scaling factor that represents the scale change applied during the alignment process.
    """
    np.set_printoptions(precision=3, suppress=True)

    # Center the trajectories by subtracting their means
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    # Calculate the covariance matrix W
    # This corresponds to the computation of
    # the similarity transformation matrix Mc mentioned in the text.
    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column],
                      data_zerocentered[:, column])

    U, d, Vh = np.linalg.linalg.svd(W.transpose())

    # Ensure a right-handed coordinate system
    # The closed-form solution in the paper
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered

    # Compute the scaling factor 's' to align the trajectories
    dots = 0.0
    norms = 0.0
    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,
                       column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    # Compute the translation vector 'trans'
    trans = data.mean(1) - s * rot * model.mean(1)

    # Apply the scaling, rotation, and translation to the original model
    model_aligned = s * rot * model + trans

    # Calculate the alignment error and translational error
    alignment_error = model_aligned - data
    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error, s


def reference_head(scale=0.01, pyr=(10., 0.0, 0.0)):
    kps = np.asarray([[-7.308957, 0.913869, 0.000000], [-6.775290, -0.730814, -0.012799],
                      [-5.665918, -3.286078, 1.022951], [-5.011779, -4.876396, 1.047961],
                      [-4.056931, -5.947019, 1.636229], [-1.833492, -7.056977, 4.061275],
                      [0.000000, -7.415691, 4.070434], [1.833492, -7.056977, 4.061275],
                      [4.056931, -5.947019, 1.636229], [5.011779, -4.876396, 1.047961],
                      [5.665918, -3.286078, 1.022951],
                      [6.775290, -0.730814, -0.012799], [7.308957, 0.913869, 0.000000],
                      [5.311432, 5.485328, 3.987654], [
                          4.461908, 6.189018, 5.594410],
                      [3.550622, 6.185143, 5.712299], [
                          2.542231, 5.862829, 4.687939],
                      [1.789930, 5.393625, 4.413414], [
                          2.693583, 5.018237, 5.072837],
                      [3.530191, 4.981603, 4.937805], [
                          4.490323, 5.186498, 4.694397],
                      [-5.311432, 5.485328, 3.987654], [-4.461908, 6.189018, 5.594410],
                      [-3.550622, 6.185143, 5.712299], [-2.542231, 5.862829, 4.687939],
                      [-1.789930, 5.393625, 4.413414], [-2.693583, 5.018237, 5.072837],
                      [-3.530191, 4.981603, 4.937805], [-4.490323, 5.186498, 4.694397],
                      [1.330353, 7.122144, 6.903745], [
                          2.533424, 7.878085, 7.451034],
                      [4.861131, 7.878672, 6.601275], [
                          6.137002, 7.271266, 5.200823],
                      [6.825897, 6.760612, 4.402142], [-1.330353, 7.122144, 6.903745],
                      [-2.533424, 7.878085, 7.451034], [-4.861131, 7.878672, 6.601275],
                      [-6.137002, 7.271266, 5.200823], [-6.825897, 6.760612, 4.402142],
                      [-2.774015, -2.080775, 5.048531], [-0.509714, -1.571179, 6.566167],
                      [0.000000, -1.646444, 6.704956], [0.509714, -1.571179, 6.566167],
                      [2.774015, -2.080775, 5.048531], [0.589441, -2.958597, 6.109526],
                      [0.000000, -3.116408, 6.097667], [-0.589441, -2.958597, 6.109526],
                      [-0.981972, 4.554081, 6.301271], [-0.973987, 1.916389, 7.654050],
                      [-2.005628, 1.409845, 6.165652], [-1.930245, 0.424351, 5.914376],
                      [-0.746313, 0.348381, 6.263227], [0.000000, 0.000000, 6.763430],
                      [0.746313, 0.348381, 6.263227], [
                          1.930245, 0.424351, 5.914376],
                      [2.005628, 1.409845, 6.165652], [
                          0.973987, 1.916389, 7.654050],
                      [0.981972, 4.554081, 6.301271]]).T  # 58 3D points
    R = rotate_zyx(np.deg2rad(pyr))
    kps = transform(R, kps*scale)
    tris = Delaunay(kps[:2].T).simplices.copy()
    return kps, tris


def reference_body():
    kps = np.array([[3.1233e-03, -3.5141e-01,  1.2037e-02],
                    [6.1313e-02, -4.4417e-01, -1.3965e-02],
                    [-6.0144e-02, -4.5532e-01, -9.2138e-03],
                    [3.6056e-04, -2.4152e-01, -1.5581e-02],
                    [1.1601e-01, -8.2292e-01, -2.3361e-02],
                    [-1.0435e-01, -8.1770e-01, -2.6038e-02],
                    [9.8083e-03, -1.0966e-01, -2.1521e-02],
                    [7.2555e-02, -1.2260e+00, -5.5237e-02],
                    [-8.8937e-02, -1.2284e+00, -4.6230e-02],
                    [-1.5222e-03, -5.7428e-02,  6.9258e-03],
                    [1.1981e-01, -1.2840e+00,  6.2980e-02],
                    [-1.2775e-01, -1.2868e+00,  7.2819e-02],
                    [-1.3687e-02,  1.0774e-01, -2.4690e-02],
                    [4.4842e-02,  2.7515e-02, -2.9465e-04],
                    [-4.9217e-02,  2.6910e-02, -6.4741e-03],
                    [1.1097e-02,  2.6819e-01, -3.9522e-03],
                    [1.6408e-01,  8.5243e-02, -1.5756e-02],
                    [-1.5179e-01,  8.0435e-02, -1.9143e-02],
                    [4.1820e-01,  1.3093e-02, -5.8214e-02],
                    [-4.2294e-01,  4.3942e-02, -4.5610e-02],
                    [6.7019e-01,  3.6314e-02, -6.0687e-02],
                    [-6.7221e-01,  3.9410e-02, -6.0935e-02],
                    [-4.6678e-03,  2.6767e-01, -9.5914e-03],
                    [3.1599e-02,  3.1083e-01,  6.2195e-02],
                    [-3.1600e-02,  3.1083e-01,  6.2194e-02],
                    [7.7209e-01,  2.7626e-02, -4.1335e-02],
                    [8.0224e-01,  1.6472e-02, -4.0185e-02],
                    [8.0875e-01, -5.1052e-03, -4.2618e-02],
                    [7.7959e-01,  2.9986e-02, -6.4667e-02],
                    [8.0606e-01,  1.4768e-02, -6.9308e-02],
                    [8.1108e-01, -8.5781e-03, -7.1220e-02],
                    [7.5424e-01,  2.1775e-02, -1.0444e-01],
                    [7.6636e-01,  7.3269e-03, -1.0981e-01],
                    [7.7016e-01, -1.1526e-02, -1.0964e-01],
                    [7.6763e-01,  2.7046e-02, -8.8031e-02],
                    [7.9120e-01,  1.0989e-02, -8.9926e-02],
                    [7.9545e-01, -1.2559e-02, -8.9112e-02],
                    [7.1083e-01,  1.8337e-02, -3.5076e-02],
                    [7.3246e-01,  6.9800e-04, -2.3452e-02],
                    [7.5646e-01, -7.6581e-03, -1.6904e-02],
                    [-7.7209e-01,  2.7627e-02, -4.1335e-02],
                    [-8.0224e-01,  1.6473e-02, -4.0184e-02],
                    [-8.0875e-01, -5.1044e-03, -4.2616e-02],
                    [-7.7959e-01,  2.9988e-02, -6.4669e-02],
                    [-8.0606e-01,  1.4770e-02, -6.9310e-02],
                    [-8.1108e-01, -8.5770e-03, -7.1221e-02],
                    [-7.5424e-01,  2.1775e-02, -1.0444e-01],
                    [-7.6636e-01,  7.3276e-03, -1.0981e-01],
                    [-7.7016e-01, -1.1525e-02, -1.0964e-01],
                    [-7.6764e-01,  2.7048e-02, -8.8034e-02],
                    [-7.9120e-01,  1.0991e-02, -8.9928e-02],
                    [-7.9545e-01, -1.2557e-02, -8.9113e-02],
                    [-7.1082e-01,  1.8335e-02, -3.5074e-02],
                    [-7.3246e-01,  6.9654e-04, -2.3450e-02],
                    [-7.5647e-01, -7.6567e-03, -1.6904e-02],
                    [8.1727e-07,  2.7588e-01,  1.1294e-01],
                    [-3.3018e-02,  3.1129e-01,  7.5301e-02],
                    [3.4586e-02,  3.1127e-01,  7.4859e-02],
                    [-7.1341e-02,  2.7677e-01, -1.4943e-02],
                    [7.2024e-02,  2.8323e-01, -1.4660e-02],
                    [8.7891e-02, -1.2863e+00,  1.3817e-01],
                    [1.5520e-01, -1.2939e+00,  9.0668e-02],
                    [9.6269e-02, -1.2658e+00, -1.1773e-01],
                    [-7.9235e-02, -1.2879e+00,  1.3619e-01],
                    [-1.5521e-01, -1.2939e+00,  9.0667e-02],
                    [-9.8138e-02, -1.2824e+00, -1.1574e-01],
                    [7.8187e-01, -2.8468e-02, -1.3414e-02],
                    [8.1442e-01, -2.9543e-02, -4.3518e-02],
                    [8.1413e-01, -3.6012e-02, -7.0888e-02],
                    [7.9637e-01, -3.7088e-02, -8.3112e-02],
                    [7.7425e-01, -3.3062e-02, -1.0473e-01],
                    [-7.8023e-01, -2.8799e-02, -9.9889e-03],
                    [-8.1442e-01, -2.9542e-02, -4.3515e-02],
                    [-8.1394e-01, -3.6286e-02, -7.0859e-02],
                    [-7.9624e-01, -3.7267e-02, -8.3044e-02],
                    [-7.7425e-01, -3.3059e-02, -1.0473e-01],
                    [-6.1207e-02,  3.1478e-01,  4.8969e-02],
                    [-5.2276e-02,  3.3204e-01,  6.3757e-02],
                    [-3.7966e-02,  3.3650e-01,  7.6953e-02],
                    [-2.2457e-02,  3.3535e-01,  8.4111e-02],
                    [-9.2774e-03,  3.3089e-01,  8.6310e-02],
                    [9.2767e-03,  3.3089e-01,  8.6311e-02],
                    [2.2456e-02,  3.3535e-01,  8.4114e-02],
                    [3.7965e-02,  3.3651e-01,  7.6956e-02],
                    [5.2274e-02,  3.3205e-01,  6.3760e-02],
                    [6.1205e-02,  3.1478e-01,  4.8973e-02],
                    [7.8541e-07,  3.1507e-01,  8.9552e-02],
                    [7.4400e-05,  3.0411e-01,  9.8290e-02],
                    [1.8244e-05,  2.9431e-01,  1.0655e-01],
                    [2.9447e-05,  2.8426e-01,  1.1401e-01],
                    [-1.2236e-02,  2.7020e-01,  9.3512e-02],
                    [-6.2909e-03,  2.6837e-01,  9.7300e-02],
                    [1.2929e-05,  2.6653e-01,  9.9285e-02],
                    [6.2914e-03,  2.6837e-01,  9.7301e-02],
                    [1.2236e-02,  2.7021e-01,  9.3513e-02],
                    [-4.5269e-02,  3.0985e-01,  6.5911e-02],
                    [-3.6538e-02,  3.1596e-01,  7.4935e-02],
                    [-2.6869e-02,  3.1550e-01,  7.5687e-02],
                    [-1.8548e-02,  3.0942e-01,  7.3419e-02],
                    [-2.6488e-02,  3.0691e-01,  7.4988e-02],
                    [-3.6002e-02,  3.0639e-01,  7.3990e-02],
                    [1.8547e-02,  3.0942e-01,  7.3422e-02],
                    [2.6868e-02,  3.1550e-01,  7.5691e-02],
                    [3.6537e-02,  3.1596e-01,  7.4939e-02],
                    [4.5268e-02,  3.0986e-01,  6.5914e-02],
                    [3.6001e-02,  3.0640e-01,  7.3993e-02],
                    [2.6487e-02,  3.0691e-01,  7.4992e-02],
                    [-2.3740e-02,  2.4298e-01,  8.4656e-02],
                    [-1.6264e-02,  2.4871e-01,  9.4408e-02],
                    [-6.3308e-03,  2.5140e-01,  9.9746e-02],
                    [2.5156e-06,  2.5072e-01,  1.0048e-01],
                    [6.3327e-03,  2.5140e-01,  9.9747e-02],
                    [1.6266e-02,  2.4871e-01,  9.4409e-02],
                    [2.3741e-02,  2.4298e-01,  8.4657e-02],
                    [1.6855e-02,  2.3921e-01,  9.2247e-02],
                    [6.6958e-03,  2.3801e-01,  9.8480e-02],
                    [1.8619e-06,  2.3774e-01,  9.9167e-02],
                    [-6.6934e-03,  2.3801e-01,  9.8479e-02],
                    [-1.6853e-02,  2.3921e-01,  9.2246e-02],
                    [-2.3127e-02,  2.4307e-01,  8.4205e-02],
                    [-6.3188e-03,  2.4484e-01,  9.4825e-02],
                    [9.8153e-07,  2.4475e-01,  9.5298e-02],
                    [6.3209e-03,  2.4484e-01,  9.4825e-02],
                    [2.3481e-02,  2.4305e-01,  8.4598e-02],
                    [6.8651e-03,  2.4460e-01,  9.4979e-02],
                    [7.6591e-07,  2.4477e-01,  9.5699e-02],
                    [-6.8633e-03,  2.4460e-01,  9.4978e-02]]).T
    return kps


def rotate_zyx(theta):
    sx, sy, sz = np.sin(theta)
    cx, cy, cz = np.cos(theta)
    return np.array([
        [cy * cz, cy * sz, -sy, 0],
        [-cx * sz + cz * sx * sy, cx * cz + sx * sy * sz, cy * sx, 0],
        [cx * cz * sy + sx * sz, cx * sy * sz - cz * sx, cx * cy, 0],
        [0, 0, 0, 1]], dtype=float)


def transform(E, p):
    p = np.array(p)
    if p.ndim > 1:
        return E[:3, :3]@p + E[:3, 3, None]
    return E[:3, :3]@p + E[:3, 3]


def get_sphere(theta, phi, radius):
    # convert degrees to radians
    theta = theta / 180. * pi
    phi = phi / 180. * pi

    # position in 3d is given by two angles (+ radius)
    # x = r * cos(s) * sin(t)
    # y = r * sin(s) * sin(t)
    # z = r * cos(t)

    x = radius * cos(theta) * sin(phi)
    y = radius * sin(theta) * sin(phi)
    z = radius * cos(phi)
    return x, y, z


def get_ellipse(theta, phi, radius_x, radius_y, radius_z):
    # convert degrees to radians
    theta = theta / 180. * pi
    phi = phi / 180. * pi
    # position in 3d is given by two angles (+ radius)
    # x = r * cos(s) * sin(t)
    # y = r * sin(s) * sin(t)
    # z = r * cos(t)
    x = radius_x * cos(theta) * sin(phi)
    y = radius_y * sin(theta) * sin(phi)
    z = radius_z * cos(phi)
    return x, y, z


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
    """
    The inverse_rotate_zyx function returns two sets of Euler angles 
    because it handles a specific case known as gimbal lock. 
    Gimbal lock occurs when the rotation representation loses one degree 
    of freedom due to the alignment of two of the three rotational axes. 
    In this scenario, there are multiple valid sets of Euler angles that 
    can represent the same orientation.
    """

    if np.linalg.norm(M[:3, :3].T @ M[:3, :3] - np.eye(3)) > 1e-5:
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