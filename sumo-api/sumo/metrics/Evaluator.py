#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from easydict import EasyDict as edict
import numpy as np

import sumo.utils as utils

class Evaluator():
    """
    Base class for evaluating a submission.
    
    """

    def __init__(self, submission, ground_truth, settings=None):
        """
        Constructor.

        Computes and caches shape similarity and data association
        computations.

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene
        settings (EasyDict) - configuration for the evaluator.  See top of
          file for recognized keys and values.  
        """

        self._settings = self.default_settings() if settings is None else settings

        # compute similarity between all detections and gt elements
        self._similarity_cache = self._make_similarity_cache(submission, ground_truth)
        
        # compute data association (for all thresholds)
        self._amodal_data_assoc = self._amodal_data_association(
            submission, ground_truth, _settings.thresholds
        )

        #::: save submission and ground_truth
   

    @staticmethod
    def default_settings(self):
        """
        Create and return an EasyDict containing default settings.
        """

        thresholds = np.linspace(0.5, 0.95, 10)
        recall_samples = np.linspace(0, 1, 101)
                                                         
        return edict({"thresholds": thresholds,
                      "recall_samples": recall_samples})

    
    def shape_similarity(self):
        """
        Compute shape similarity score (Equation 6 in SUMO white paper)

        Return:
        float - shape similarity score
        """

        n_gt = len(self._ground_truth.elements)
        
        aps = []  # average precision list
        for t in self._settings.thresholds:

            # construct input needed for PR curve computation
            det_matches = []
            det_scores = []
            for element in self._submission.elements:
                if element.id in self._amodal_data_assoc[t]:
                    det_matches.append(1)  # correct detection
                else:
                    det_matches.append(0)  # false positive
                det_scores.append(element.score)
                
            (precision, _) = utils.compute_pr(
                det_matches, det_scores, n_gt,
                recall_samples = self._settings.recall_samples,
                interp=True
            )

            aps.append(np.mean(precision))  # Equation 4
        return np.mean(aps)   # Equation 6


    def pose_error(self):
        """
        Compute pose error for the submission.  The pose error
        consists of a rotation error, which is the average geodesic
        distance between detected and ground truth rotations, and a
        translation error, which is the average translation difference
        beween detected and ground truth translations.  See SUMO white paper
        for details.

        Return:
        (rotation_error, translation_error) - where
        rotation_error (float) - Equation 7 in SUMO white paper
        translation_error (float) - Equation 9 in sumo paper
        """
        
        rot_errors = []
        trans_errors = []

        for t in self._settings.thresholds:
            rot_errors1, trans_errors1 = [], []
            for corr in self._amodal_data_assoc[t].itervalues():
                det_element = self._submission.elements[corr.det_id]
                gt_element = self._ground_truth.elements[corr.gt_id]

                q_gt = utils.matrix2quaternion(gt_element.pose.R.R)
                q_det = utils.matrix2quaternion(det_element.pose.R.R)
                #  Eq. 8
                rot_errors1.append(np.sqrt(1/2.0) *
                                  LA.norm(np.log(np.dot(q_gt, q_det.T))))
                # Eq. 10
                trans_errors1.append(
                    LA.norm(gt_element.pose.t - det_element.pose.t)
                )
                
            rot_errors.append(np.mean(rot_errors1))
            trans_errors.append(np.mean(trans_errors1))

        # Eqs. 7 and 9    
        return np.mean(rot_errors), np.mean(trans_errors)


#------------------------
# End of public interface
#------------------------

    def _make_similarity_cache(self, submission, ground_truth):
        """
        Compute similarity between each pair of elements in submission
        and ground truth.  

        Inputs:
        submission (ProjectScene) - Submitted scene to be evaluated
        ground_truth (ProjectScene) - The ground truth scene

        Return:
        similarity_cache (dict of dict of Corr) - 
        similarity_cache[det_id][gt_id] = corr, where corr stores the
        similarity and detection score for a putative match between a
        detection and a ground truth element (det_id & gt_id).
        """

        sim_cache = {}
        for det_element in submission.elements:
            det_id = det_element.id
            sim_cache[det_id] = {}
            det_score = det_element.score
            for gt_element in ground_truth.elements:
                corr = Corr(det_id, gt_element.id, det_score,
                            self._shape_similarity(sub_element, gt_element))
                sim_cache[det_id][gt_id] = corr
        return sim_cache

    def _shape_similarity(self, element1, element2):
        """
        Defines a similarity function that compares the shape of
          <element1> and <element2>. 
        The actual similarity function must be defined in
        track-specific child classes.

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject) 

        Return:
        float - shape similarity score
        """
        raise NotImplementedError('Instantiate a child class')

    def _amodal_data_association(self, submission, ground_truth, thresholds):
        """
        Computes amodal (category-independent) data association
        between the elements in <submission> and <ground_truth> for
        each similarity threshold in <thresholds>.

        Inputs:
        thresholds (list of float) - Similarity thresholds to be used.

        Return:
        data_association (dict of dicts of Corr) -
        data_association[thresh][det_id], where thresh is taken from
        <thresholds> and det_id is a detection ID.  If a det_id is not
        in the dict, no correspondance was found.
        
        Note:
        self._similarity_cache must be already computed.

        Algorithm:
        1. Matches with similarity < thresh are eliminated from
        consideration.
        2. The remaining detections are sorted by decreasing detection
        score.
        3. Loop over detections.  The detection with highest score is
        assigned to its corresponding GT, and that detection and GT
        are removed from further consideration.  If a detection has
        multiple possible matches in the GT, the match with highest
        similarity score is used. 
        """

        # make copy of the cache that we can modify
        sim_cache = self._similarity_cache.deep_copy()

        # for storing results
        data_assoc = {}  # key is threshold

        # loop in increasing similarity threshold order
        # Note: This allows us to reuse edits to similarity cache,
        # since any putative matches ruled out for a given threshold
        # will also be ruled out for higher thresholds.
        for thresh in thresholds.sorted():
            
            # remove matches with similarity < thresh
            for det_id in sim_cache.keys():
                for gt_id, corr in sim_cache[det_id].iteritems():
                    if corr.simlarity < thresh:
                        pop(sim_cache[det_id][gt_id])

            # for tracking GT elements that have already been assigned
            # at this threshold
            assigned_gts = {} # key is gt_id, val is det_id
            
            # iterate over detections sorted in descending order of
            # score.
            for det_id, _ in sorted(
                    submission.iteritems(), 
                    key = lambda(id, element): return element.score,
                    reverse = True):

                # make list of possible matches for this det_id and
                # sort by similarity
                possible_corrs = [
                    corr for corr in sim_cache[det_id].values()
                    if corr.gt_id not in assigned_gts]
                sort_corrs_by_similarity(possible_corrs)

                # create match with last corr in list
                if len(possible_corrs) > 0:
                    data_assoc[thresh][det_id] = possible_corrs[-1]
                    assigned_gts[possible_corrs[-1].gt_id] = det_id
                # else no match for this det_id

        return data_assoc




    
class Corr():
    """
    Helper class for storing a correspondence.
    """

    def __init__(self, det_id, gt_id, similarity, det_score):
        self.det_id = det_id
        self.gt_id = gt_id
        self.similarity = similarity
        self.det_score = det_score

    def sort_corrs_by_similarity(corrs):
        """
        Sort a list of correspondences by similarity.  Sort is in place.
        
        Inputs:
        corrs (list of Corr) - correspondences to sort.
        """
        corrs.sort(key = lambda(corr): return corr.similarity)

        
#---------------  old code below here (need to revise)


    
    def association(self, ground_truth, submission, thresh):
        """
        Associate instances in submission with instances in ground-truth
        for evaluation.

        1. Compute the shape similarity between all pairs of
        predicted elements and ground truth elements (within a scene)

        2. Sort the shape similarity scores in a descending order.

        3. Choose the available pair with the largest shape similarity and
        mark the two elements as unavailable.

        4. Repeat step 3 until the shape similarity is lower than a threshold <thresh>.

        Inputs:
        ground_truth (list of ProjectObject) - Ground-truth objects for one scene.
        submission (list of ProjectObject) - Submission objects for one scene.
        thresh (float) -  shape similarity threshold for match (IoU measure)

        Return:
        tuple (matched, missed, extra) where
        matched (dict) - maps ground truth object ids (keys) to
            submission object ids (values) representing a match between a
            ground-truth object and a submitted object.
        missed (list) - object ids of non-matched ground-truth objects
            (i.e., in GT but not in the submission)
        extra (list) - object ids of non-matched detected objects
            (i.e., in submission but not in GT)
        """

        Ngt = len(ground_truth)
        Nsub = len(submission)
        ssimilarity = np.zeros((Ngt, Nsub))
        for i in range(Ngt):
            for j in range(Nsub):
                ssimilarity[i, j] = self._shape_similarity(
                    ground_truth[i], submission[j])
        assigned_gt, assigned_sub, missed, extra = [], [], [], []
        matched = {}
        for i in range(Ngt):
            candidates = np.argsort(-ssimilarity[i, :])
            for j in range(Nsub):
                cand = candidates[j]
                if cand not in assigned_sub and ssimilarity[i, cand] >= thresh:
                    matched[i] = cand
                    assigned_gt.append(i)
                    assigned_sub.append(cand)
                    break
        missed = list(set(range(Ngt)) - set(assigned_gt))
        extra = list(set(range(Nsub)) - set(assigned_sub))

        return matched, missed, extra

    def evaluate_geometry(self, ground_truth, submission):
        """
            Computes shape and pose score for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:  :::fix order and return type
                pose_score: (R_error, t_error) tuple representing average
                rotation matrix geodesic distance, and translation error
                shape_score: Average Precision (class-agnostic) of submission
        """
        shape_score = AP(self, ground_truth, submission)
        pose_score = pose_error(self, ground_truth, submission)

        return shape_score, pose_score

    def evaluate_appearance(self, ground_truth, submission):
        """
            Computes appeareance score for a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                appearance_score: appearance score of submission
        """
        raise NotImplementedError('Instantiate a child class')

    def evaluate_semantics(self, ground_truth, submission):
        """
            Computes semantic score a participant's submission
            Args:
                ground_truth: Ground-truth - list of ProjectObject instances
                submission: Submission - list of ProjectObject instances
            Returns:
                semantic_score: mean Average Precision (mean AP across classes)
                of submission
        """

        semantic_score = mAP(self, ground_truth, submission)

        return semantic_score

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
        raise NotImplementedError('Instantiate a child class')

    
    def _compute_category_agnostic_map(self):
        """
        """

        aps = []  # list of average precisions (one per sim thresh)
        for sim_thresh in range(self._settings.sim_thresh_start,
                                self._settings.sim_thresh_end,
                                self._settings.sim_thresh_step):
            ap = self._compute_category_agnostic_ap(sim_thresh)
            aps.append(ap)
        return np.mean(aps)

    def _compute_category_agnostic_ap(self, sim_thresh):
        """
        Compute the category agnostic average precision (eq. 4)
        for a single similarity threshold (tau).
        """
        
