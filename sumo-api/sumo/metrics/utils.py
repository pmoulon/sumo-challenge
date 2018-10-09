"""
    Metrics helper functions
"""

import math
import numpy as np

from sumo.geometry.rot3 import Rot3

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pymesh
from pymesh.meshutils import remove_duplicated_vertices_raw
import pyny3d.geoms as pyny
from numpy import linalg as LA
from sympy.geometry import Point, Segment
from sklearn.neighbors import BallTree
from collections import defaultdict
from scipy import stats

#from sumo.metrics.voxelize_cpu import voxelize

def matrix_to_quat(M):
    """
    Transform a 3X3 Rotation matrix to a unit quaternion.

    Inputs:
    M (numpy 3x3 array of float) - rotation matrix

    Return:
    q (numpy vector of float) -  quaternion in (w,i,j,k) order

    Source:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    www.ee.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf
    """
    m00, m01, m02 = M[0, :]
    m10, m11, m12 = M[1, :]
    m20, m21, m22 = M[2, :]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    D = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return np.array([qw/D, qx/D, qy/D, qz/D])


def quat_to_matrix(quat):
    """
    Convert quaternion to rotation matrix

    Inputs:
    quat (numpy vector of float) - quaternion in (w,i,j,k) order

    Return
    M (numpy 3x3 array of float) - rotation matrix

    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    q = math.sqrt(2.0) * quat
    q = np.outer(q, q)
    return np.array([[1 - q[2,2] - q[3,3], q[1,2] - q[3,0], q[1,3] + q[2,0]],
                        [q[1,2] + q[3,0], 1.0 - q[1,1] - q[3,3], q[2,3] - q[1,0]],
                        [q[1,3] - q[2,0], q[2,3] + q[1,0], 1.0 - q[1,1] - q[2,2]]])
    
def quat_to_euler(q):
    """
    Convert quaternion to static ZYX Euler angle representation (i.e., R = R(z)*R(y)*R(x))
    
    Inputs:
    q (numpy vector of float) - quaternion in (w,i,j,k) order
    
    Return:
    numpy vector of float - Euler angles in radians (z, y, x)
    
    Source: matlab quat2eul
    eul = [ atan2( 2*(qx.*qy+qw.*qz), qw.^2 + qx.^2 - qy.^2 - qz.^2 ), ...
    asin( -2*(qx.*qz-qw.*qy) ), ...
    atan2( 2*(qy.*qz+qw.*qx), qw.^2 - qx.^2 - qy.^2 + qz.^2 )];
    
    """
    return np.array([math.atan2(2 * (q[1] * q[2] + q[0] * q[3]),
                                q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]),
                     math.asin(-2 * (q[1] * q[3] - q[0] * q[2])),
                     math.atan2(2 * (q[2] * q[3] + q[0] * q[1]),
                                q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])])


def euler_to_quat(e):
    """
    Convert ZYX euler angles to quaternion

    Inputs:
    e (numpy vector of float) - Euler angles in radians (z, y, x)

    Return:
    q (numpy vector of float) - quaternion in (w,i,j,k) order
    """
    cz = math.cos(0.5 * e[0])
    sz = math.sin(0.5 * e[0])
    cy = math.cos(0.5 * e[1])
    sy = math.sin(0.5 * e[1])
    cx = math.cos(0.5 * e[2])
    sx = math.sin(0.5 * e[2])

    return np.array([cz * cy * cx + sz * sy * sx,
		     cz * cy * sx - sz * sy * cx,
		     cz * sy * cx + sz * cy * sx,
                     sz * cy * cx - cz * sy * sx])
    

def euler_to_matrix(e):
    """
    Convert ZYX Euler angles to 3x3 rotation matrix.

    Inputs:
    e (numpy 3-vector of float) - ZYX Euler angles (radians)

    Return:
    matrix (3x3 numpy array2 of float) - rotation matrix

    TODO: This could be optimized somewhat by using the direct
    equations for the final matrix rather than multiplying out the 
    matrices.
    """
    return (Rot3.Rz(e[0]) * Rot3.Ry(e[1]) * Rot3.Rx(e[2])).R

def matrix_to_euler(matrix):
    """
    Convert 3x3 matrix to ZYX Euler angles.
    
    Inputs:
    matrix (numpy 3x3 numpy array2 of float) - rotation matrix

    Return:
    numpy 3-vector of float - ZYX Euler angles (radians)

    TODO:
    This could be written more efficiently going directly between
    matrix and Euler angles, but there are singularities and issues
    with numerical stability to be considered.
    """
    return quat_to_euler(matrix_to_quat(matrix))
    

def compute_pr(det_matches, det_scores, n_gt, recall_samples=None, interp=False):
    """
    Compute the precision-recall curve.

    Inputs:
    det_matches (numpy vector of N ints) - Each non-zero entry is a
      correct detection.  Zeroes are false positives.
      det_scores (numpy vector of N floats) - The detection scores for
      the corresponding matches.  Higher is better.
    n_gt (int) - The number of ground truth entities in the task.
    recall_samples (numpy vector of float) - If set, compute precision at
      these sample locations.  Values must be between 0 and 1
      inclusive. 
    interp (Boolean) - If true, the interpolated PR curve will be
      generated (as described in :::cite pascal voc paper)

    Return:
    (precision, recall)
    precision (numpy vector of float) - precision values at corresponding <recall> points
      recall (numpy vector of float) - recall locations.  If
      <recall_samples> is not set, it is the locations where precision 
      changes.  Otherwise it is set to <recall_samples>.
    """

    # sort input based on score
    indices = np.argsort(-det_scores)
    sorted_matches = det_matches[indices]

    # split out true positives and false positives
    tps = np.not_equal(sorted_matches, 0)
    fps = np.equal(sorted_matches, 0)

#    print(tps)
#    print(fps)
#    print(sorted_matches)
    # compute basic PR curve
    tp_sum = np.cumsum(tps)
    fp_sum = np.cumsum(fps)

    # use epsilon to prevent divide by 0 special case
    epsilon = np.spacing(1)  

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / n_gt

    # compute interpolated PR curve
    if (interp):
        for i in range(len(precision)-1, 0, -1):
            if precision[i] > precision[i-1]:
                precision[i-1] = precision[i]
                
    # compute at recall sample points 
    # Note: This is what MS Coco does.  Not sure if it is correct,
    # but it should be sufficient if the number of samples used to
    # create the PR curve is large enough.  
    # This assigns the precision value for a given recall_sample to 
    # the nearest value on the right.  Anything greater than the last
    # computed recall value will be set to zero.
    if recall_samples is not None:
        n_precision = len(precision)
        precision2 = np.zeros(len(recall_samples))  # default is 0
        
        indices2 = np.searchsorted(recall, recall_samples, side='left')
        for recall_index, precision_index in enumerate(indices2):
            if (precision_index < n_precision):
                precision2[recall_index] = precision[precision_index]
        precision = precision2
        recall = recall_samples

#    print("precision = {}".format(str(precision)))
#    print("recall = {}".format(str(recall)))
    
        
    return (precision, recall)
    
    
def plot_pr(precision, recall):
    """
    Creates a new figure and generates a plot of a precision recall curve.

    Inputs:
    precision (numpy vector of N floats) - precision values at corresponding recall
    recall (numpy vector of N floats) - recall values
    
    Return:
    Figure - matplotlib Figure object for the plot.
    
    Notes:
    does not call plt.show()
    """
    fig = plt.figure()
    plt.plot(recall, precision, 'r-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1.1])    

    return fig


def voxel_or_points_iou(pv1, pv2, config):
    """
    Compute voxel or points intersection over union for two objects.
    To compare the similarity between the mesh (resp voxel centers) of
    object i and j, we define a point(resp voxel centers) IoU metric as
    the ratio of overlapping mesh points (resp voxel centers) over the
    total number of mesh points (resp voxel centers). An overlapping point
    (resp voxel center) in element i is a point (resp voxel center) whose
    center is within a small distance threshold of at least one other
    point (resp voxel center) in element j

    Args:
    pv1, pv2: (N X 7) points or voxel centers for ProjectObject
    instances whose shapes to compare
    config.voxel_th: distance threshold expressed as number of voxels
    config.voxel_size: Voxel Size
    config.voxel_th * config.voxel_size is the distance
    threshold for 2 points or voxel centers to be considered "the same"

    Returns:
    iou: points (resp voxel centers) intersection over union -
    inter/union - where inter  = (n1 + n2) where n1/n2 is the number
    of points (resp voxel centers of obj1/obj2 that are within the
    threshold distance of another point(resp voxel center) of obj2/obj1
    union is the total number of points(resp voxels centers) in obj1
    and obj2
    """
    id1, id2, dist1, dist2 = nearest_neighbor(pv1, pv2)
    inter1 = np.sum(dist1 <= config.voxel_size * config.voxel_th)
    inter2 = np.sum(dist2 <= config.voxel_size * config.voxel_th)
    inter = inter1 + inter2
    union = id1.shape[0] + id2.shape[0]

    return inter/union


def nearest_neighbor(pv1, pv2):
    """
        Compute nearest neighbor of points from one set of points to the other
        Args:
            pv1, pv2: (N X 7) points or voxel centers
        Returns:
            id1, id2: (N,) id of nearest neighbor in other point set
            dist1, dist2: (N,) corresponding distance to nearest neigbor for
            each point

    """
    tree1 = BallTree(pv1[:, 0:3])
    tree2 = BallTree(pv2[:, 0:3])
    dist1, ind1 = tree2.query(pv1[:, 0:3])
    dist2, ind2 = tree1.query(pv2[:, 0:3])
    ind1 = ind1.flatten()
    ind2 = ind2.flatten()
    return ind1, ind2, dist1, dist2

def sample_mesh(faces, density=5e2):
    """ 
    Sample points from a mesh surface using barycentric coordinates
    
    Inputs:
    faces (np array - 3*N x 6) -  matrix representing vertices and faces with
      X, Y, Z, R, G, B faces[0:3, :] is the first face. N is the number of faces
    density (float) - Number of points per square meter

    Return:
    points (np array - N X 6 matrix of sampled points
    """
    A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
    cross = np.cross(A[:, 0:3] - C[:, 0:3] , B[:, 0:3] - C[:, 0:3])
    areas = 0.5*(np.sqrt(np.sum(cross**2, axis=1)))

    # ::: set minimum of 1 sample
    Nsamples_per_face = (density*areas).astype(int)
    N = np.sum(Nsamples_per_face)
    #TODO: Remove after mesh reading bug is fixed
    if N == 0:
        return np.empty((0, 3))
    face_ids = np.zeros((N,), dtype=int)
    
    count = 0
    for i, n in enumerate(Nsamples_per_face):
        face_ids[count:count + Nsamples_per_face[i]] = i
        count += Nsamples_per_face[i]

    A = A[face_ids, :]; B = B[face_ids, :]; C = C[face_ids, :]
    r = np.random.uniform(0, 1, (N, 2))
    sqrt_r1 = np.sqrt(r[:, 0:1])
    points = (1 - sqrt_r1)*A + sqrt_r1*(1 - r[:, 1:])*B + sqrt_r1*r[:, 1:]*C
    return points









# :::not used
def category_filter_matched(matched, ground_truth, submission, category):
    """
        Filter matches by keeping only dictionary elements for which
        either the key or value elements are of a given category
        Args:
            matched: dictionary of values gt_id:sub_id representing a
            match between a ground-truth instance and a submitted instance
            ground_truth: ground-truth - list of projectobject instances
            submission: submission - list of projectobject instances
            category: category used for filtering
        Returns:
            filtered: dictionary of values gt_id:sub_id representing a
            match between a ground-truth instance and a submitted instance
            where either the ground_truth or the submitted instance or both,
            are belong to category
    """
    filtered = {}
    for gt_id, sub_id in matched.items():
        gt = ground_truth[gt_id]
        sub = submission[sub_id]
        if gt.category == category or sub.category == category:
            filtered[gt_id] = sub_id
    return filtered


def category_filter_list(item_ids, object_list, category):
    """
        Filter list of objects by keeping only elements that belong to the
        given category
        Args:
            item_ids: IDs of select objects in object_list
            object_list: list of projectobject instances
            category: category used for filtering
        Returns:
            filtered: subset of item_ids corresponding to objects that belong
            to category
    """

    filtered = []
    for id_ in item_ids:
        item = object_list[id_]
        if item.category == category:
            filtered.append(id_)
    return filtered


def AP(evaluator, ground_truth, submission):
    """
        Compute average precision by averaging precision values across
        IoU thresholds in config.IoU_start:config.IoU_step:config.IoU_step
        range
        Args:
            evaluator: Evaluator object
            ground_truth: ground-truth - list of projectobject instances
            submission: submission - list of projectobject instances
        Returns:
            AP: average precision
    """
    APs = []
    config = evaluator.config
    for IoU_th in range(config.IoU_start, config.IoU_end, config.IoU_step):
        matched, missed, extra = evaluator.association_cache[IoU_th]
        P, R = compute_PR(matched, missed, extra)
        APs.append(P)  # these are precisions, not average precisons :::
    return np.mean(APs)


def count_category_instances(items, category):
    """
        Count the number of objects in items that belong to category
        Args:
            ground_truth: ground-truth - list of projectobject instances
            category: category of interest
        Returns:
            count: count of elements in items belonging to category
    """
    count = 0
    for item in items:
        if item.category == category:
            count += 1
    return count


def mAP(evaluator, ground_truth, submission):
    """
        Compute mean Average precision by averaging precision across categories
        and across shape similarity thresholds
        Args:
            evaluator: Evaluator object
            ground_truth: ground-truth - list of projectobject instances
            submission: submission - list of projectobject instances
        Returns:
            mAP: mean Average Precision
        References:
            http://cocodataset.org/#detections-eval
    """

    APs = []
    config = evaluator.config
    for IoU_th in range(config.IoU_start, config.IoU_end, config.IoU_step):
        classAP = []
        matched, missed, extra = evaluator.association_cache[IoU_th]
        for category in evaluator.config.categories:
            if count_category_instances(ground_truth, category) + \
                    count_category_instances(submission, category) == 0:
                continue
            fmatched = category_filter_matched(matched, ground_truth, submission, category)
            fmissed = category_filter_list(missed, ground_truth, category)
            fextra = category_filter_list(extra, submission, category)
            P, R = compute_PR(fmatched, fmissed, fextra)
            classAP.append(P)
        APs.append(np.mean(classAP))
    return np.mean(APs)



def oriented_box2mesh(obj):
    """
        Represents the 3D bounding box of an object as a mesh object
        Args:
            obj: ProjectObject instance
        Returns:
            boxmesh: pymesh.Mesh object representation of obj's oriented
            3D bounding box
    """
    box = obj._bounds
    pose = obj.pose
    i1, i2, i3, i4, i5, i6, i7, i8 = range(8)
    faces = [[i1, i2, i3], [i1, i3, i4], [i4, i3, i8], [i4, i8, i5],
             [i5, i8, i7], [i5, i7, i6], [i6, i7, i2], [i6, i2, i1],
             [i1, i4, i5], [i1, i5, i6], [i2, i7, i8], [i2, i8, i3]]

    vertices = box.corners().T
    for i in range(len(vertices)):
            vertices[i] = np.dot(vertices[i], pose.R.R) + pose.t

    boxmesh = pymesh.form_mesh(np.array(vertices), np.array(faces))

    return boxmesh


def voxelize_points(points, config, min_xyz, max_xyz):
    """ Voxelize point cloud

    ¦   Args:
    ¦   ¦   points (Nx7 float)   : raw points
    ¦   ¦   config                 : Easy and configuration
            min_xyz: minimum x, y, x coordinates to consider
            max_xyz: maximum x, y, x coordinates to consider
    ¦   Returns:
    ¦   ¦   voxels (kxlxmx5)     : the voxelized cloud
    """
    vox_size = float(config.voxel_size)
    num_categories = int(len(config.categories))
    noColor = float(config.empty_color)  # color value for empty voxels
    noCat = float(config.empty_cat)    # category value for empty voxels
    voxels = voxelize(points.astype(np.float32), min_xyz[0],
                      max_xyz[0], min_xyz[1], max_xyz[1],
                      min_xyz[2], max_xyz[2], vox_size, noColor,
                      noCat, num_categories)
    return voxels[:, :, :, 0:5]

def voxel_to_centers(voxels, voxel_size, min_xyz):
    """
        Extract voxel centers from a voxelized volume
        Args:
            voxels [K X L X M X C] voxelized volume
        Returns:
            centers [N X (C + 2)] array of voxel centers
    """
    centers = []
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            for k in range(voxels.shape[2]):
                if voxels[i, j, k, 0] == 1:
                    xyz = voxel_size*np.array([i, j, k]) + min_xyz + voxel_size/2
                    centers.append(xyz[np.newaxis, :])
    if len(centers) == 0:
        centers = np.empty((0, 0))
    else:
        centers = np.concatenate(centers, axis=0)

    return centers


def get_obj_points_and_voxels(obj, config, with_voxels=False):
    """
        Uniformly sample and return object points and optionally
        voxel centers
        Args:
            obj: ProjectObject instance
            config.voxel_size: Voxel Size
            with_voxels: if true, voxelize sampled mesh
        Returns:
            points: N X 7 sampled mesh (X, Y, Z, R, G, B, Cat)
            centers: N X 3 matrix of voxel centers in real-world coordinates
    """
    vertices = obj.mesh.vertices().T
    vertices_xyz_rgb_cat = np.zeros((vertices.shape[0], 7))
    indices = obj.mesh.indices()
    uv_coords = obj.mesh.uv_coords().T
    base_color = obj.mesh.base_color()
    W, H, _ = base_color.shape
    ucoord = np.mod((uv_coords[:, 0] * W).astype(int), W)
    vcoord = np.mod((uv_coords[:, 1] * H).astype(int), H)
    vertices_xyz_rgb_cat[:, 0:3] = vertices
    vertices_xyz_rgb_cat[:, 3:6] = base_color[ucoord, vcoord, :]
    vertices_xyz_rgb_cat[:, 6] = config.categories.index(obj.category)
    # Workaround for Mesh reading bug
    # TODO: Fix bug and remove this
    new_indices = []
    for i in range(0, indices.shape[0], 3):
        if np.sum(indices[i:i+3] < vertices.shape[0]) == 3:
            for j in range(3):
                new_indices.append(indices[i + j])
    indices = np.array(new_indices)
    faces = vertices_xyz_rgb_cat[indices, :]
    points = sample_mesh(faces)
    centers = None
    if points.shape[0] > 0:
        if with_voxels:
            min_xyz = np.min(points[:, 0:3], axis=0)
            max_xyz = np.max(points[:, 0:3], axis=0)
            voxels = voxelize_points(points, config, min_xyz, max_xyz)
            centers = voxel_to_centers(voxels, config.voxel_size, min_xyz)
    obj.points = points
    obj.vcenters = centers
    return obj




def points_rmsd(evaluator, ground_truth, submission, use_voxels):
    """
    For voxel and mesh representation, we use [2] to compare matched scene
    elements by computing a measure of the distance from the points (or voxel
    centers) of one element to the points (or voxel centers) of the other. We
    average this value across thresholds to obtain the average root mean
    squared surface distance measure (RMSD)
    Args:
        evaluator: Evaluator object
        ground_truth: ground-truth - list of projectobject instances
        submission: submission - list of projectobject instances
        use_voxels: if True, use voxel centers, else, use mesh points
    Returns:
        average root mean squared surface distance measure (RMSD)
    References:
        [2]https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
    """
    config = evaluator.config
    rmsd_IoU = []
    for IoU_th in range(config.IoU_start, config.IoU_end, config.IoU_step):
        matched, missed, extra = evaluator.association_cache[IoU_th]
        rmsd = []
        for gt_id, sub_id in list(matched.items()):
            if use_voxels:
                gt_pv = ground_truth[gt_id].vcenters
                sub_pv = submission[sub_id].vcenters
            else:
                gt_pv = ground_truth[gt_id].points
                sub_pv = submission[sub_id].points
            ind1, ind2, dist1, dist2 = nearest_neighbor(gt_pv, sub_pv)
            rmsd.append(np.sqrt((np.sum(dist1**2) +
                         np.sum(dist2**2))/(dist1.shape[0] + dist2.shape[0])))
        rmsd_IoU.append(np.mean(rmsd))
    return np.mean(rmsd_IoU)

def color_rmsd(evaluator, ground_truth, submission, use_voxels):
    """
    For voxel and mesh representation, we use a variant of [2] to compare
    the color of matched scene elements by computing a measure of the distance
    from the points (or voxel centers)'s color of one element to the points
    (or voxel centers)'color of the other. For each point in one element,
    we aggregate the color difference between the point and it's neighbor in
    the other element and derive a root mean square error. We average this
    value across thresholds to obtain the average root mean squared
    surface color distance measure (RMSCD)
    Args:
        evaluator: Evaluator object
        ground_truth: ground-truth - list of projectobject instances
        submission: submission - list of projectobject instances
        use_voxels: if True, use voxel centers, else, use mesh points
    Returns:
        average root mean squared surface color distance measure (RMSCD)
    References:
        [2]https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
    """

    config = evaluator.config
    rmsd_IoU = []
    for IoU_th in range(config.IoU_start, config.IoU_end, config.IoU_step):
        matched, missed, extra = evaluator.association_cache[IoU_th]
        rmsd = []
        for gt_id, sub_id in list(matched.items()):
            if use_voxels:
                gt_pv = ground_truth[gt_id].vcenters
                sub_pv = submission[sub_id].vcenters
            else:
                gt_pv = ground_truth[gt_id].points
                sub_pv = submission[sub_id].points
            ind1, ind2, dist1, dist2 = nearest_neighbor(gt_pv, sub_pv)
            if use_voxels:
                pv1 = ground_truth[gt_id].vcenters
                pv2 = submission[sub_id].vcenters
            else:
                pv1 = ground_truth[gt_id].points
                pv2 = submission[sub_id].points
            dist1 = pv1[:, 3:6] - pv2[ind1, 3:6]
            dist2 = pv2[:, 3:6] - pv1[ind2, 3:6]
            rmsd.append(np.sqrt((np.sum(dist1**2) + np.sum(dist2**2))
                                / (dist1.shape[0] + dist2.shape[0])))
        rmsd_IoU.append(np.mean(rmsd))
    return np.mean(rmsd_IoU)




def to_surface(mesh):
    """
        Convert Mesh object into pyny.Surface object (used for visualization)
        Args:
            mesh: Mesh object
        Returns:
            pyny.Surface object
    """
    vert = mesh.vertices
    faces = mesh.faces
    surface = []
    for i in range(faces.shape[0]):
        points = vert[faces[i], :]
        surface.append(pyny.Polygon(np.array(points)))
    return pyny.Surface(surface)


def visualize_mesh(obj, visualize):
    vertices = obj.mesh.vertices().T.reshape((-1, 3))
    indices = obj.mesh.indices().reshape((-1, 3))
    mesh = pymesh.form_mesh(vertices, indices)
    if visualize:
        to_surface(mesh).plot('r')
        plt.show()
        plt.waitforbuttonpress()
        plt.close()


