# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import print_function
import numpy as np

from mapillary_vistas.evaluation.instance_sizes import AVERAGE_CATEGORY_SIZE

from pprint import pprint


def calculate_instance_specific_pixel_accuracy_from_arrays(prediction_labels, ground_truth, labels):
    """
    Calculates the weighted measures for the iIoU metric.
    This only applies to labels with the 'instances' flag set.
    """

    ground_truth_labels = ground_truth / 256

    instance_information = {}

    for label_id, label in enumerate(labels):
        if not label['evaluate']:
            continue
        if not label['instances']:
            continue

        instance_information[label['name']] = {
            'raw_true_positives': 0,
            'weighted_true_positives': 0,
            'raw_false_negatives': 0,
            'weighted_false_negatives': 0,
        }

        current_ground_truth_indices = ground_truth_labels == label_id

        if np.count_nonzero(current_ground_truth_indices) == 0:
            continue

        current_ground_truth_instances = ground_truth[current_ground_truth_indices] % 256
        instance_count = np.bincount(current_ground_truth_instances)
        for instance_id, instance_size in enumerate(instance_count):
            if instance_size == 0:
                continue

            current_instance_indices = ground_truth == instance_id + label_id * 2**8
            current_true_positives = np.count_nonzero(prediction_labels[current_instance_indices] == label_id)

            current_false_negatives = instance_size - current_true_positives

            factor = AVERAGE_CATEGORY_SIZE.get(label['name'], instance_size) / instance_size

            instance_information[label['name']]['raw_true_positives'] += current_true_positives
            instance_information[label['name']]['weighted_true_positives'] += current_true_positives * factor
            instance_information[label['name']]['raw_false_negatives'] += current_false_negatives
            instance_information[label['name']]['weighted_false_negatives'] += current_false_negatives * factor

    return instance_information
