"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Algorithm class: Evaluate a mesh track submission
"""

from easydict import EasyDict as edict
import numpy as np

import sumo.utils as utils

class Evaluator():
    """
    Base class for evaluating a submission.

    Do not instantiate objects of this class.  Instead, use one of the
    track-specific sub-classes.

    Configuration:
    The algorithm is configured using the settings, which is an
    EasyDict object.  Recognized keys:
    thresholds (numpy vector of float) - values for IoU thresholds (tau) at
      which the similarity will be measured.  Default 0.5 to 0.95 in
      0.05 increments.
    recall_samples (numpy vector of float) - recall values where PR
      curve will be sampled for average precision computation.
      Default 0 to 1 in 0.01 increments.
    categories (list of string) - categories for which the semantic metric
      should be evaluated.  Default: a small subset of the SUMO
      evaluation categories (see default_settings)
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
        self._submission = submission
        self._ground_truth = ground_truth

        # compute similarity between all detections and gt elements
        self._similarity_cache = self._make_similarity_cache(submission, ground_truth)
        
        # compute amodal and modal data association (for all thresholds)
        self._amodal_data_assoc = self._amodal_data_assoc(
            submission.elements, ground_truth.elements,
            _settings.thresholds, self._similarity_cache
        )

        self._modal_data_assoc = self._modal_data_assoc(
            submission.elements, ground_truth.elements,
            _settings.thresholds,
            _settings.categories, self._similarity_cache
        )
        

    @staticmethod
    def default_settings(self):
        """
        Create and return an EasyDict containing default settings.
        """

        thresholds = np.linspace(0.5, 0.95, 10)
        recall_samples = np.linspace(0, 1, 101)
        categories = ["wall", "chair"]
        
        return edict({"thresholds": thresholds,
                      "recall_samples": recall_samples,
                      "categories": categories
        })
    


    def evaluate_all(self):
        """
        Computes all metrics for the submission.

        Return:
        dict with key: value pairs (keys are strings - values are corresponding evaluation 
        metrics.  Exact keys depend on evaluation track.
        """
        raise NotImplementedError('Instantiate a child class')
    

    def shape_similarity_score(self):
        """
        Compute shape similarity score (Equation 6 in SUMO white paper)

        Return:
        float - shape similarity score
        """

        n_gt = len(self._ground_truth.elements)
        
        aps = []  # average precision list
        for t in self._settings.thresholds:

            # construct input needed for PR curve computation
            # det_matches = 1 if correct detection, 0 if false positive
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

    def appearance_score(self):
        """
        Compute appeareance score for the submission.

        Return:
        float - appearance score (Equation 13 in SUMO white paper)
        """
        raise NotImplementedError('Instantiate a child class')


    def semantics_score(self):
        """
        Compute semantic score for the submission.
        Return:
        semantic_score (float) - mean Average Precision (mean AP across classes)
        of submission (Equation 15 in SUMO white paper)
        """

        # Initialize:
        # n_gt[cat] = number of GT elements in that category
        # det_matches[cat] = list of matches (1 for each detection).
        #   Entry is 1 for a correct match, 0 for a false positive
        # det_scores[cat] = list of detection scores (1 for each detection).
        for cat in self._categories:
            n_gt[cat] = 0
            det_matches[cat] = []
            det_scores[cat] = []
        aps = []  # average precision list

        # compute number of GT elements in each category
        for element in self._submission.elements:
            n_gt[element.category] += 1
        
        for t in self._settings.thresholds:

            # build lists of matches per category
            for element in self._submission.elements:
                cat = element.category
                if element.id in self._modal_data_assoc[t][cat]:
                    det_matches[cat].append(1)  # correct detection
                else:
                    det_matches[cat].append(0)  # false positive
                det_scores[cat].append(element.score)

            # compute PR curve per category
            for c in self._settings.categories:
                (precision, _) = utils.compute_pr(
                    det_matches, det_scores, n_gt,
                    recall_samples = self._settings.recall_samples,
                    interp=True)
                aps.append(np.mean(precision))  # Equation 15
        return np.mean(aps)  # Equation 16

    
    def evaluate_perceptual(self):
        """
        ::: need to fix
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
        Similarity function that compares the shape of <element1> and
        <element2>.  
        The actual similarity function must be defined in
        track-specific child classes. 

        Inputs:
        element1 (ProjectObject)
        element2 (ProjectObject) 

        Return:
        float - shape similarity score
        """
        raise NotImplementedError('Instantiate a child class')

    def _amodal_data_assoc(self, sub_elements, gt_elements, thresholds, sim_cache):
        """
        Computes amodal (category-independent) data association
        between the elements in <submission> and <ground_truth> for
        each similarity threshold in <thresholds>.

        Inputs:
        sub_elements (ProjectObjectDict) - submitted scene elements
        gt_elements (ProjectObjectDict) - corresponding ground truth
          scene elements
        thresholds (list of float) - Similarity thresholds to be used.
        sim_cache (dict of dict of Corrs) - similarity cache -
          sim_cache[det_id][gt_id] is similarity between det_id and gt_id.

        Return:
        data_assoc (dict of dicts of Corr) -
        data_assoc[thresh][det_id], where thresh is taken from
        <thresholds> and det_id is a detection ID.  If a det_id is not
        in the dict, no correspondance was found.
        
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
        sim_cache2 = sim_cache.deep_copy()

        # for storing results
        data_assoc = {}  # key is threshold

        # loop in increasing similarity threshold order
        # Note: This allows us to reuse edits to similarity cache,
        # since any putative matches ruled out for a given threshold
        # will also be ruled out for higher thresholds.
        for thresh in thresholds.sorted():
            
            # remove matches with similarity < thresh
            for det_id in sim_cache2.keys():
                for gt_id, corr in sim_cache2[det_id].iteritems():
                    if corr.simlarity < thresh:
                        pop(sim_cache2[det_id][gt_id])

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
                    corr for corr in sim_cache2[det_id].values()
                    if corr.gt_id not in assigned_gts]
                sort_corrs_by_similarity(possible_corrs)

                # create match with last corr in list
                if len(possible_corrs) > 0:
                    data_assoc[thresh][det_id] = possible_corrs[-1]
                    assigned_gts[possible_corrs[-1].gt_id] = det_id
                # else no match for this det_id

        return data_assoc


    def _modal_data_assoc(self, sub_elements, gt_elements,
                          thresholds, categories, sim_cache):
        """
        Computes modal (category-specific) data association
        between the elements in <submission> and <ground_truth> for
        each similarity threshold in <thresholds>.

        Inputs:
        sub_elements (ProjectObjectDict) - submitted scene elements
        gt_elements (ProjectObjectDict) - corresponding ground truth
          scene elements
        thresholds (list of float) - Similarity thresholds to be used.
        sim_cache (dict of dict of Corrs) - similarity cache -
          sim_cache[det_id][gt_id] is similarity between det_id and gt_id.

        Return:
        data_assoc (dict of dicts of dicts of Corr) -
        data_assoc[category][thresh][det_id], where thresh is taken from
        <thresholds> and det_id is a detection ID.  If a det_id is not
        in the dict, it means that no correspondance was found.
        

        Algorithm:
        For each category C in category list (from settings):
        1. Get a list of elements in submission and GT belonging to
        that category;  If both lists are empty, we skip the category.
        2. Construct submission and ground truth ProjectScenes using
        only elements with category C.
        3. Compute amodal data association using these subsets.

        """

        # Split detections and gt elements by category and store in
        # dict of ProjectObjectDicts.  
        dets_by_cat = {}
        gts_by_cat = {}
        for cat in categories:
            dets_by_cat[cat] = ProjectObjectDict()
            gts_by_cat[cat] = ProjectObjectDict()

        for (id, element) in det_elements.items():
            dets_by_cat[element.category][id] = element

        for (id, element) in gt_elements.items():
            gts_by_cat[element.category][id] = element


        data_assoc = {}  # for storing results (key is category)

        for cat in categories:
            dets = dets_by_cat[cat]
            gts = gts_by_cat[cat]
            if (len(dets) + len(gts)) == 0:
                continue

            # build mini sim_cache
            sim_cache_cat = {}
            for det_id in dets.keys():
                sim_cache_cat[det_id] = {}
                for gt_id in gts.keys():
                    sim_cache_cat[det_id][gt_id] = sim_cache[det_id][gt_id]

            # do data association
            data_assoc[cat] = self._amodal_data_assoc(dets, gts, thresholds, sim_cache_cat)

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



    




    
        
