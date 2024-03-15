import numpy as np


def unit(v):
    return v / np.linalg.norm(v)


def rot_mat_from_bases(b1, b2, b3):
    A = np.eye(3)
    A[:, 0] = b1
    A[:, 1] = b2
    A[:, 2] = b3
    return A.T


def rot_from_a_to_b(a: np.ndarray, b: np.ndarray):
    cross_1_2 = np.cross(a, b)
    skew_symm_cross_1_2 = np.array([
        [0, -cross_1_2[2], cross_1_2[1]],
        [cross_1_2[2], 0, -cross_1_2[0]],
        [-cross_1_2[1], cross_1_2[0], 0],
    ])
    cos = np.dot(a, b)
    R = (np.identity(3) + skew_symm_cross_1_2 +
         np.dot(skew_symm_cross_1_2, skew_symm_cross_1_2) * 1 / (1 + cos + 1e-15))
    return R


def three_pts_to_rot_mat(p1, p2, p3, neg_x=False):
    xaxis = unit(p2 - p1)
    if neg_x:
        xaxis *= -1
    v_another = unit(p3 - p1)
    zaxis = -unit(np.cross(xaxis, v_another))
    yaxis = unit(np.cross(zaxis, xaxis))
    return get_rot_mat_from_basis(xaxis, yaxis, zaxis)


def project_axis_to_plane(plane_normal, axis_to_project):
    plane_normal = unit(plane_normal)
    axis_to_project /= np.linalg.norm(axis_to_project)
    proj_to_plane_from_axis = (np.dot(plane_normal, axis_to_project) /
                               np.linalg.norm(plane_normal) * plane_normal)
    return unit(axis_to_project - proj_to_plane_from_axis)


def rodrigues_rotation_matrix(axis, angle):
    axis = unit(axis)  # Ensure it's a unit vector
    K = skew_symmetric(axis)
    angle_rad = np.radians(angle)
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    return R


def rot_about_orthogonal_axes(v, theta_deg, phi_deg):

    def skew_symmetric(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # axes
    a = np.array([1, 0, 0], dtype=np.float64)
    b = np.array([0, 1, 0], dtype=np.float64)
    a = project_axis_to_plane(v, a)
    b = np.cross(v, a)

    # Compute rotation matrices
    R_a = rodrigues_rotation_matrix(a, theta_deg)
    R_b = rodrigues_rotation_matrix(b, phi_deg)

    # Combined rotation
    R = R_b @ R_a  # Order depends on which rotation you want to apply first

    return R


if __name__ == "__main__":
    assert np.allclose(
        rot_about_orthogonal_axes(np.random.rand(3).astype(np.float64), 0, 0),
        np.eye(3),
    )
