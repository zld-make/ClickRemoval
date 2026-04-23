import gc
import time
from typing import List
import torch
import numpy as np
from torchvision.transforms.functional import gaussian_blur
from src.markov_map import flood_fill_with_min_threshold, create_semantic_markov_map_from_start_state
from src.utils import get_integer_sobel_gradient, collect_segment_border_pixels, matrix_ipf, change_temperature, create_single_point_heatmap, joint_bilateral_upsampling
import kornia.filters as F

def get_thresholds_with_segment_boundary_gradients(distance_map: torch.Tensor):
    # ---------------------
    # Collect gradient info
    # ---------------------
    semantic_gradient_map = get_integer_sobel_gradient(distance_map[None, None])
    res = collect_segment_border_pixels(semantic_gradient_map, distance_map)

    # Calculate average segment boundary gradient
    boundary_semantic_gradient = res['values'][:, :-1] / res['border_lengths'][:-1][None]

    # Return result
    return {
        'thresholds': res['distances'][:-1],
        'segment_boundary_average_semantic_gradients': boundary_semantic_gradient[0]
    }


def get_point_statistics_for_each_threshold(distance_map: torch.Tensor,
                                            thresholds: torch.Tensor,
                                            positive_points: List,
                                            negative_points: List):
    """
    :param distance_map: (H, W)
    :param thresholds: (N,)
    :param positive_points: (num_points, 2)
    :param negative_points: (num_points, 2)
    :return: updated scores in the shape of (batch, N)
    """
    values_positive_points = torch.zeros((len(positive_points)), dtype=distance_map.dtype, device=distance_map.device)
    for i, (x, y) in enumerate(positive_points):
        values_positive_points[i] = distance_map[y, x]

    values_negative_points = torch.ones((len(negative_points) + 1), dtype=distance_map.dtype, device=distance_map.device) * torch.max(distance_map)
    for i, (x, y) in enumerate(negative_points):
        values_negative_points[i] = distance_map[y, x]
    neg_lowest_value = torch.min(values_negative_points)

    return {
        'percentage_of_positive_points_included_in_segment': torch.mean((values_positive_points[:, None] <= thresholds[None, :]).float(), dim=0),
        'no_negative_point_in_segment': (thresholds < neg_lowest_value).float()
    }


class M2N2SegmentationModel(object):
    def __init__(self,
                 attention_aggregator,
                 temperature=0.65,
                 semantic_distance_map_relative_probability_threshold=0.3,
                 use_floodfill=True,
                 jbu_spatial_sigma=1.0,
                 jbu_range_sigma=0.1,
                 max_prompt_point_segment_area=0.4,
                 score_prior=True,
                 score_edge=True,
                 score_pos=True,
                 score_neg=True,
                 cache_size=20):

        super().__init__()
        self.attention_aggregator = attention_aggregator
        self.temperature = temperature
        self.semantic_distance_map_relative_probability_threshold = semantic_distance_map_relative_probability_threshold
        self.use_flood_fill = use_floodfill
        self.jbu_spatial_sigma = jbu_spatial_sigma
        self.jbu_range_sigma = jbu_range_sigma
        self.max_prompt_point_segment_area = max_prompt_point_segment_area
        self.score_prior = score_prior
        self.score_edge = score_edge
        self.score_pos = score_pos
        self.score_neg = score_neg

        # Statistics to keep track of
        self.num_attentions_loaded = 0
        self.num_point_prompts_processed = 0
        self.attention_tensor_preparation_time_total = 0
        self.point_prompt_processing_time_total = 0

        # Caching stuff
        self.cached_distance_maps = dict()
        self.cache_size = cache_size
        self.cached_attention_tensor = None
        self.prev_img = None

    def prepare_cache(self, img: np.ndarray):
        to_delete = []
        for k, v in self.cached_distance_maps.items():
            if v['used'] is False:
                to_delete.append(k)
        for k in to_delete:
            del self.cached_distance_maps[k]

        if self.prev_img is not None and not np.array_equal(self.prev_img, img):
            self.cached_attention_tensor = None
            self.cached_distance_maps = dict()
            gc.collect()
            torch.cuda.empty_cache()
        self.prev_img = img.copy()
        for k, v in self.cached_distance_maps.items():
            v['used'] = False

    def upscale_to_image_resolution(self, x: torch.Tensor, img: np.ndarray):
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float, device=x.device).permute((2, 0, 1))[None]
        x = joint_bilateral_upsampling(x[None, None], img_tensor, self.jbu_spatial_sigma, self.jbu_range_sigma)[0, 0]
        return x

    def get_distance_map(self, img: np.ndarray, attn: torch.Tensor, point):
        bh, bw, h, w = attn.shape

        # Start state
        start_state = create_single_point_heatmap(
            point=point,
            in_shape=(img.shape[:2]),
            out_shape=(h, w),
            device=attn.device,
            dtype=attn.dtype
        )

        # Markov semantic map
        distance_map = create_semantic_markov_map_from_start_state(
            A=attn.reshape(bh * bw, h * w),
            start_state=start_state.reshape(h * w),
            threshold=self.semantic_distance_map_relative_probability_threshold
        ).reshape(h, w)

        # Upscale
        distance_map = self.upscale_to_image_resolution(distance_map, img)
        distance_map = distance_map / torch.max(distance_map)

        distance_map_orig = distance_map
        # Floodfill to get the final Markov-map
        if self.use_flood_fill:
            distance_map = torch.tensor(flood_fill_with_min_threshold(
                heatmap=distance_map.cpu().numpy(),
                start_point=point
            ), dtype=distance_map.dtype, device=distance_map.device)

        # Normalize
        distance_map = distance_map / torch.max(distance_map)
        return distance_map, distance_map_orig

    def calculate_scores_for_distance_map(self,
                                          distance_map: torch.Tensor,
                                          positive_points,
                                          negative_points):

        # ----------------------------------------
        # Collecting statistics for each threshold
        # ----------------------------------------
        res = get_thresholds_with_segment_boundary_gradients(
            distance_map=distance_map
        )
        thresholds = res['thresholds']
        segment_boundary_average_semantic_gradients = res['segment_boundary_average_semantic_gradients']

        res2 = get_point_statistics_for_each_threshold(
            distance_map=distance_map,
            thresholds=thresholds,
            positive_points=positive_points,
            negative_points=negative_points
        )
        percentage_of_positive_points_included_in_segment = res2['percentage_of_positive_points_included_in_segment']
        no_negative_point_in_segment = res2['no_negative_point_in_segment']

        # -------------------------------------
        # Calculate the score of each threshold
        # -------------------------------------
        no_full_image_segments = torch.linspace(0.0, 1.0, len(thresholds), dtype=thresholds.dtype, device=thresholds.device) < self.max_prompt_point_segment_area
        scores = torch.ones_like(segment_boundary_average_semantic_gradients)
        if self.score_prior:
            scores = scores * no_full_image_segments
        if self.score_edge:
            scores = scores * segment_boundary_average_semantic_gradients
        if self.score_pos:
            scores = scores * percentage_of_positive_points_included_in_segment
        if self.score_neg:
            scores = scores * no_negative_point_in_segment

        # Return the thresholds and their scores
        return thresholds, scores

    def get_cached_attention_tensor(self, img):
        if self.cached_attention_tensor is not None:
            return self.cached_attention_tensor

        # Set parameters
        start_attention_preparation_timer = time.time()

        # Get attention tensor
        attn = self.attention_aggregator.extract_attention(img)
        bh, bw, h, w = attn.shape

        # Adjust temperature and make doubly stochastic
        attn = change_temperature(attn.reshape(bh * bw, h * w), temperature=self.temperature).reshape(bh, bw, h, w)
        attn = matrix_ipf(attn.reshape(bh * bw, h * w), iterations=200).reshape(bh, bw, h, w)
        self.cached_attention_tensor = attn

        # Update statistics
        self.attention_tensor_preparation_time_total += time.time() - start_attention_preparation_timer
        self.num_attentions_loaded += 1
        return attn

    def get_cached_distance_map(self, point, img, attn):
        if point in self.cached_distance_maps:
            res = self.cached_distance_maps[point]
        else:
            distance_map_with_ff, distance_map_without_ff = self.get_distance_map(img=img, attn=attn, point=point)
            res = {
                'distance_map_ff': distance_map_with_ff,
                'distance_map_no_ff': distance_map_without_ff
            }
            if len(self.cached_distance_maps) < self.cache_size:
                self.cached_distance_maps[point] = res
        res['used'] = True
        return res['distance_map_ff'], res['distance_map_no_ff']

    def get_segmentation_of_single_point(self,
                                         point_idx,
                                         img: np.ndarray,
                                         attn: torch.Tensor,
                                         all_points: List,
                                         points_in_segment: List,
                                         transition_gradient_threshold=0.05):
        point = all_points[point_idx]
        positive_points = []
        negative_points = []
        for p_idx, (p_point, p_in_segment) in enumerate(zip(all_points, points_in_segment)):
            if p_in_segment:
                positive_points.append(p_point)
            else:
                negative_points.append(p_point)

        # Get a normalized distance map from 0 (closest to point) to 1.0 (furthest to point)
        distance_map_ff, distance_map_no_ff = self.get_cached_distance_map(img=img, attn=attn, point=point)

        # Calculate the scores for each possible threshold
        thresholds, scores = self.calculate_scores_for_distance_map(
            distance_map=distance_map_ff,
            positive_points=positive_points,
            negative_points=negative_points
        )

        # Get the segmentation map with the threshold of the best score
        best_threshold_idx = torch.argmax(scores)
        best_score = scores[best_threshold_idx]
        if self.score_prior is False and self.score_edge is False and self.score_pos is False and self.score_neg is False:
            best_threshold = torch.tensor(0.5, dtype=thresholds.dtype, device=thresholds.device)
        else:
            best_threshold = thresholds[best_threshold_idx]
        # segmentation = (distance_map_ff <= best_threshold).long()
        segmentation = (distance_map_ff <= best_threshold).float()
        adjusted_distance_map = distance_map_ff / (torch.clip(best_threshold, 0.0000001))

        # Return segmentation and the score
        return segmentation, adjusted_distance_map, best_score, distance_map_no_ff

    def segment(self, img: np.ndarray, points: List, points_in_segment: List):
        # Round all points to have integer pixel coordinates
        points = [(int(round(p[0])), int(round(p[1]))) for p in points]

        # ---------------------
        # Getting the attention
        # ---------------------
        self.prepare_cache(img)
        attn = self.get_cached_attention_tensor(img)
        # --------------------------------
        # Estimate segments for each point
        # --------------------------------
        start_prompt_point_timer = time.time()
        segmentations = torch.zeros((len(points), img.shape[0], img.shape[1]), dtype=torch.long, device=attn.device)
        distance_map_ff = torch.zeros((len(points), img.shape[0], img.shape[1]), dtype=torch.float, device=attn.device)
        distance_map_no_ff = torch.zeros((len(points), img.shape[0], img.shape[1]), dtype=torch.float, device=attn.device)
        scores = torch.zeros((len(points),), dtype=torch.long, device=attn.device)
        for point_idx, label in enumerate(points_in_segment):
            segmentations[point_idx], distance_map_ff[point_idx], scores[point_idx], distance_map_no_ff[point_idx] = self.get_segmentation_of_single_point(
                point_idx=point_idx, # distance_map_ff have norm， distance_map_no_ff no have norm
                img=img,
                attn=attn,
                all_points=points,
                points_in_segment=points_in_segment if label is True else [not s for s in points_in_segment]
            )
        # ------------------------------
        # Merging segments of each point
        # ------------------------------
        all_points_signs = torch.tensor(np.array(points_in_segment, dtype=bool), device=attn.device)
        min_indices = torch.argmin(distance_map_ff, dim=0)
        pass_higher_than_negative_points = all_points_signs[min_indices.reshape(-1)].reshape(img.shape[:2])
        pass_in_segment_threshold = torch.min(distance_map_ff, dim=0).values <= 1.0
        # segmentation = (torch.logical_and(pass_higher_than_negative_points, pass_in_segment_threshold)).long()
        segmentation = (torch.logical_and(pass_higher_than_negative_points, pass_in_segment_threshold)).float()

        # Write the prompt point pixel to the output (We assume the prompt pixel to be GT)
        for point, point_in_seg in zip(points, points_in_segment):
            segmentation[point[1], point[0]] = 1 if point_in_seg else 0

        positive_indices = [i for i, is_pos in enumerate(points_in_segment) if is_pos]
        if len(positive_indices) > 0:
            positive_semantic_maps = distance_map_no_ff[positive_indices]
            min_semantic_distances = torch.min(positive_semantic_maps, dim=0).values
            semantic_map = 1.0 - min_semantic_distances

        segmentation = gaussian_blur(segmentation.unsqueeze(0).unsqueeze(0), kernel_size=(57, 57)).squeeze()
        soft_mask = segmentation

        self.point_prompt_processing_time_total += time.time() - start_prompt_point_timer
        self.num_point_prompts_processed += 1
        return segmentation, semantic_map, soft_mask
