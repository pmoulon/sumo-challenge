"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""

from sumo.metrics.utils import color_rmsd
from sumo.metrics.utils import voxel_or_points_iou, points_rmsd
from sumo.metrics.Evaluator import Evaluator


class MeshEvaluator(Evaluator):
    """
    Algorithm to evaluate a submission for the mesh track.
    """

    def __init__(self, submission, ground_truth, settings):
        """
        Constructor.  Computes similarity between all elements in the
        submission and ground_truth and also computes amodal and modal
        data association caches. 

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene
        settings (EasyDict) - configuration for the evaluator.  See
        Evaluator.py for recognized keys and values.   
        """
        super(MeshEvaluator, self).__init__(submission, ground_truth, settings)

    def evaluate_all(self)
        """
        Compute all metrics for the submission

        Return:
        metrics (dict) - Keys/values are:
        "shape_score" : float
        "rotation_error" : float
        "translation_error" : float
        "semantics_score" : float
        "perceptual_score" : float
        """
        metrics = {}

        ::: fixme
        metrics["shape_score"] = self.shape_similarity_score()
        rotation_error, translation_error = self.pose_error()
        metrics["rotation_error"] = rotation_error
        metrics["translation_error"] = translation_error
        metrics["semantics_score"] = self.semantics_score()
        metrics["perceptual_score"] = self.perceptual_score()

        return metrics


    def evaluate_geometry(self, ground_truth, submission):
        """
            Computes shape score, pose score, and points root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                pts_rmsd: root mean squared symmetric surface distance of
                object points. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        shape_score, pose_score = super(
            MeshEvaluator, self).evaluate_geometry(ground_truth, submission)
        pts_rmsd = points_rmsd(self, ground_truth, submission, use_voxels=False)

        return shape_score, pose_score, pts_rmsd

    def evaluate_appearance(self, ground_truth, submission):
        """
            Computes shape score, pose score, and points root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                pts_color_rmsd: root mean squared symmetric surface distance of
                object points. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """

        pts_color_rmsd = color_rmsd(self, ground_truth, submission, use_voxels=False)

        return pts_color_rmsd

    def evaluate_perceptual(self, ground_truth, submission):
        """
            Computes perceptual score for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                perceptual_score: a list of 3 tuples [(s1, s1m), (s2, s2m),
                (s3, s3m)] where s1, s2, s3 are layout, furniture and clutter
                scores respectively and s1m, s2m, s3m are the maximum possible
                scores for layout, furniture, and clutter respectively.
        """
        raise NotImplementedError('Instantiate a child class')

    def evaluate_all(self, ground_truth, submission):
        """
            Computes all metrics for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                dictionary containing geometry, appearance, semantics, and
                perceptual metrics for submission
        """
        metrics = {}
        shape_score, pose_score, rmsd = self.evaluate_geometry(
            ground_truth, submission)
        rmscd = self.evaluate_appearance(ground_truth, submission)
        mAP = self.evaluate_semantics(ground_truth, submission)
        metrics[self.config.metrics.AP[0]] = shape_score
        metrics[self.config.metrics.Rerror[0]] = pose_score[0]
        metrics[self.config.metrics.Terror[0]] = pose_score[1]
        metrics[self.config.metrics.rmsd[0]] = rmsd
        metrics[self.config.metrics.rmscd[0]] = rmscd
        metrics[self.config.metrics.mAP[0]] = mAP

        return metrics


#------------------------
# End of public interface
#------------------------

    def _shape_similarity(self, obj1, obj2):
        """
            Defines a similarity function that compares the shape of 2 object
            meshes
            Args:
                obj1.points, obj2.points: points of ProjectObject instances
                self.config.voxel_th: voxel distance threshold
                self.config.voxel_size: voxel size
                self.config.voxel_th * self.config.voxel_size is the distance
                threshold for 2 points to be considered "the same"
            Returns:
                shape_sim : points distance IoU
        """
        return voxel_or_points_iou(obj1.points, obj2.points, self.config)

    
