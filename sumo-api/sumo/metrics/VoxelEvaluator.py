"""
    Class defining function to evaluate a submission

"""

from sumo.metrics.utils import color_rmsd
from sumo.metrics.utils import voxel_or_points_iou, points_rmsd
from sumo.metrics.Evaluator import Evaluator


class VoxelEvaluator(Evaluator):
    def __init__(self, config, ground_truth, submission):
        """
            Initialize a voxel evaluator for a given submission. Perform and
            cache data association results between ground-truth and submission for
            IoU threshold in config.IoU_start:config.IoU_step:config.IoU_step
            Args:
                config: Easydict or Namespace variable containing evaluator's
                configuration
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
        """

        super(VoxelEvaluator, self).__init__(config, ground_truth, submission)

    def _shape_similarity(self, obj1, obj2):
        """
            Defines a similarity function that compares the shape of 2 objects
            voxel representations
            Args:
                obj1.vcenters, obj2.vcenters: voxel centers for ProjectObject
                instances
                self.config.voxel_th: voxel distance threshold
                self.config.voxel_size: voxel size
            Returns:
                shape_sim : voxel distance IoU
        """
        return voxel_or_points_iou(obj1.vcenters, obj2.vcenters,
                                            self.config)

    def evaluate_geometry(self, ground_truth, submission):
        """
            Computes shape score, pose score, and voxel root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                voxel_rmsd: root mean squared symmetric surface distance of
                voxel centers. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """
        shape_score, pose_score = super(
            VoxelEvaluator, self).evaluate_geometry(ground_truth, submission)
        voxel_rmsd = points_rmsd(self, ground_truth, submission, use_voxels=True)

        return shape_score, pose_score, voxel_rmsd

    def evaluate_appearance(self, ground_truth, submission):
        """
            Computes shape score, pose score, and voxel root mean squared
            symmetric distance for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
                voxel_color_rmsd: root mean squared symmetric surface distance of
                voxel centers. See https://www.cs.ox.ac.uk/files/7732/CS-RR-15-08.pdf
        """

        voxel_color_rmsd = color_rmsd(self, ground_truth, submission, use_voxels=True)

        return voxel_color_rmsd

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
