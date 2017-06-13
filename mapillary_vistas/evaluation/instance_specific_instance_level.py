"""
    This file is responsible for the instance specific evaluation metric.
    It extracts the information from ground truth and prediction masks,
    matches them and calculates the average precision for each label.
"""

from __future__ import print_function

import os
import itertools
import time
from pprint import pprint

import numpy as np
import sklearn.metrics
from PIL import Image

from mapillary_vistas.helpers.progress import progress
from mapillary_vistas.helpers.plot import plot_precision_recall


def calculate_iou_for_indices(ground_truth_indices, prediction_indices, ground_truth_size, prediction_size):
    """
    Helper function to calculate the overlap of a match.
    """

    # optimised version of
    # np.count_nonzero(np.logical_and(ground_truth_indices, prediction_indices))
    intersection = np.count_nonzero(prediction_indices[ground_truth_indices])
    union = prediction_size + ground_truth_size - intersection

    if union == 0:
        iou = float('nan')
    else:
        iou = intersection / float(union)

    return {
        'intersection': intersection,
        'iou': iou,
        'false_positives': prediction_size - intersection,
        'false_negatives': ground_truth_size - intersection,
        'prediction_size': prediction_size,
        'ground_truth_size': ground_truth_size,
    }


def calculate_instance_specific_instance_accuracy_from_arrays(instance_predictions_path, ground_truth, labels):
    """
    Load all prediction masks for the current image, and match with all
    ground truth masks.
    """

    # reduce to label information
    ground_truth_labels = ground_truth / 256

    # caluclate ground truth which is ignored
    # hits on these pixels will not count in evaluation
    # in the same loop, create a dictionary for label lookup by name
    ground_truth_ignore_labels = np.zeros_like(ground_truth_labels, dtype=np.bool)
    label_indices = {}
    for index, label in enumerate(labels):
        label_indices[label['name']] = index
        if not label['evaluate']:
            ground_truth_ignore_labels[ground_truth_labels == index] = True

    try:
        with open(instance_predictions_path) as instance_file:
            instance_prediction_infos = {}
            for line in instance_file.readlines():
                line = line.strip()
                if line.startswith("#"):
                    continue
                mask_file, label, confidence = line.split(" ")

                # enable the loading of label id or label name
                try:
                    label_id = int(label)
                except ValueError:
                    label_id = label_indices[label]

                if label_id not in instance_prediction_infos:
                    instance_prediction_infos[label_id] = []
                instance_prediction_infos[label_id].append({
                    'mask': os.path.join(os.path.dirname(instance_predictions_path), mask_file),
                    'confidence': float(confidence),
                })
    except:
        print("ERROR in {}".format(instance_predictions_path))
        print("Ensure the instance file format to be")
        print("<path to instance mask> (<label id> | <label name>) <confidence>")
        raise

    # initialize result structures
    overlap_information = {}

    # the metric is label specific
    for label_id, label in enumerate(labels):
        if not label['evaluate']:
            continue
        if not label['instances']:
            continue

        if label_id not in instance_prediction_infos:
            instance_prediction_infos[label_id] = []

        # get list of instances of current label in current image
        # note that due to overlaps, the ids might not be sequential
        ground_truth_instance_ids = np.unique(ground_truth[ground_truth_labels == label_id] % 256)

        # cache the ground truth masks/sizes for faster processing
        ground_truth_instance_information = {}
        ground_truths = {}
        for instance_id in ground_truth_instance_ids:
            instance_indices = ground_truth == instance_id + label_id * 256
            ground_truth_instance_information[instance_id] = instance_indices
            ground_truths[instance_id] = {
                'size': np.count_nonzero(instance_indices)
            }

        # prediction ids are sequential, but use the same structure for readability
        prediction_instance_ids = range(len(instance_prediction_infos[label_id]))

        prediction_instance_information = {}
        predictions = {}
        for instance_id in prediction_instance_ids:
            instance_image = Image.open(instance_prediction_infos[label_id][instance_id]['mask'])
            instance_array = np.array(instance_image)
            instance_indices = instance_array != 0
            prediction_size = np.count_nonzero(instance_indices)

            # skip emtpy masks
            if prediction_size == 0:
                continue

            prediction_instance_information[instance_id] = instance_indices

            ignore_pixel_count = np.count_nonzero(np.logical_and(
                instance_indices,
                ground_truth_ignore_labels
            ))

            predictions[instance_id] = {
                'confidence': instance_prediction_infos[label_id][instance_id]['confidence'],
                'ignore_pixel_count': ignore_pixel_count,
                'file': instance_prediction_infos[label_id][instance_id]['mask'],
                'size': prediction_size,
            }

        overlap_information[label_id] = {
            'ground_truths': ground_truths,
            'predictions': predictions,
            'ground_truth_overlaps': {},
            'prediction_overlaps': {},
            'file': instance_predictions_path,
        }

        # test every combination
        iterator = itertools.product(
            ground_truth_instance_information.keys(),
            prediction_instance_information.keys()
        )
        for ground_truth_id, prediction_id in iterator:
            overlap = calculate_iou_for_indices(
                ground_truth_instance_information[ground_truth_id],
                prediction_instance_information[prediction_id],
                ground_truths[ground_truth_id]['size'],
                predictions[prediction_id]['size']
            )

            # this information only needs to be stored once per instance, not per pair
            overlap.pop('ground_truth_size')
            overlap.pop('prediction_size')

            # only store true matches
            if overlap['iou'] > 0:
                if ground_truth_id not in overlap_information[label_id]['ground_truth_overlaps']:
                    overlap_information[label_id]['ground_truth_overlaps'][ground_truth_id] = {}
                if prediction_id not in overlap_information[label_id]['prediction_overlaps']:
                    overlap_information[label_id]['prediction_overlaps'][prediction_id] = {}

                # store the information in both directions
                overlap_information[label_id]['ground_truth_overlaps'][ground_truth_id][prediction_id] = overlap
                overlap_information[label_id]['prediction_overlaps'][prediction_id][ground_truth_id] = overlap

    return overlap_information


def calculate_average_precision(instance_specific_instance_information, labels, args):
    """
    Using the instance specific information, calculate the average precision
    over all images for each label.
    """

    # we ignore ground truths smaller than 100 pixels.
    min_size = 100
    # calculate AP for minimal overlap from 50% to 95% in 5% steps
    thresholds = np.arange(.5, 1, .05).tolist()
    label_ids = []
    for index, label in enumerate(labels):
        if not label['evaluate']:
            continue
        if not label['instances']:
            continue

        label_ids += [index]

    precisions = {}
    precisions_50 = {}

    iterator = itertools.product(thresholds, label_ids)
    # loop count for progress bar
    iteration_count = len(thresholds) * len(label_ids)
    for threshold, label_id in progress(iterator, total=iteration_count):
        # the metric is calculated independently for every label
        # In the inner loop we have to go over each image and check for instances of this label
        # Lastly we need to loop over all instances of the current label in the current image
        # However, to find the best match with the current overlap threshold, we need to iterate
        # over each ground truth - prediction combination, so there are another two loops.

        current_overlap_infos = []
        missed_ground_truths = 0
        found_ground_truths = 0
        found_predictions = 0
        for image_information in instance_specific_instance_information:
            # this image does not contain the current label
            if label_id not in image_information:
                continue

            image_information = image_information[label_id]
            ground_truth_ids = []
            for ground_truth_id in image_information['ground_truths'].keys():
                ground_truth_size = image_information['ground_truths'][ground_truth_id]['size']

                # check if the current ground truth object is big enough to matter
                if ground_truth_size >= min_size:
                    ground_truth_ids += [ground_truth_id]

            prediction_ids = image_information['predictions'].keys()

            # keep track of instance count
            found_ground_truths += len(ground_truth_ids)
            found_predictions += len(image_information['predictions'])

            # keep track of assigned predictions to determine false positives
            assigned_prediction_ids = []
            for ground_truth_id in ground_truth_ids:
                ground_truth_size = image_information['ground_truths'][ground_truth_id]['size']
                # if we do not have any overlaps for the current ground truth,
                # then it's definitely a false negative.
                if ground_truth_id not in image_information['ground_truth_overlaps']:
                    missed_ground_truths += 1
                    continue

                current_result = {}
                # for each gt instance check if any overlapping pred overlaps enough
                for prediction_id in image_information['ground_truth_overlaps'][ground_truth_id].keys():
                    overlap_information = image_information['ground_truth_overlaps'][ground_truth_id][prediction_id]
                    iou = overlap_information['iou']
                    confidence = image_information['predictions'][prediction_id]['confidence']
                    if iou < threshold:
                        continue

                    # keep track of assignments in current_result
                    # assigned stores whether a prediction is 'chosen' for the current ground truth
                    # in case another prediction matches with a higher confidence, assigned is set to False
                    is_best_match = True
                    for matching_prediction_id in current_result.keys():
                        if current_result[matching_prediction_id]['confidence'] < confidence:
                            current_result[matching_prediction_id]['assigned'] = False
                        else:
                            is_best_match = False

                    current_result[prediction_id] = {
                        'assigned': is_best_match,
                        'confidence': confidence,
                    }

                # there are overlaps, but no good ones..
                if len(current_result) == 0:
                    missed_ground_truths += 1

                # accumulate assignments and infos for all ground truth labels in the current image
                for prediction_id in prediction_ids:
                    if prediction_id in current_result:
                        result = current_result[prediction_id]
                        current_overlap_infos.append((result['assigned'], result['confidence']))
                        assigned_prediction_ids.append(prediction_id)

            # analyse the missed predictions if they are missed for a reason
            for prediction_id in prediction_ids:
                if prediction_id not in assigned_prediction_ids:
                    ignored_pixels = image_information['predictions'][prediction_id]['ignore_pixel_count']
                    prediction_size = image_information['predictions'][prediction_id]['size']

                    if prediction_id in image_information['prediction_overlaps']:
                        for ground_truth_id in image_information['prediction_overlaps'][prediction_id].keys():
                            ground_truth_size = image_information['ground_truths'][ground_truth_id]['size']
                            if ground_truth_size < min_size:
                                ignored_pixels += image_information['prediction_overlaps'][prediction_id][ground_truth_id]['intersection']

                    ignored_part = ignored_pixels / float(prediction_size)
                    if ignored_part < threshold:
                        current_overlap_infos.append((False, image_information['predictions'][prediction_id]['confidence']))

        if label_id not in precisions:
            precisions[label_id] = []

        if label_id not in precisions_50:
            precisions_50[label_id] = []

        if len(current_overlap_infos) == 0:
            if found_predictions > 0 or found_ground_truths > 0:
                precisions[label_id] += [0.0]
                if threshold == .5:
                    precisions_50[label_id] += [0.0]
            continue

        # sort matches by confidence
        current_overlap_infos.sort(key=lambda entry: entry[1])

        precision = []
        recall = []

        _, indices = np.unique([entry[1] for entry in current_overlap_infos], return_index=True)
        indices = list(indices)

        positives = sum([int(entry[0]) for entry in current_overlap_infos])

        # incides always points to the first of the unique elements
        # we want the highest (=last) positive count for each unique element,
        # so we shift the whole array by one
        true_positives_up_to = [0] + list(np.cumsum([entry[0] for entry in current_overlap_infos]))

        for index in indices:
            true_positives = positives - true_positives_up_to[index]
            false_positives = len(current_overlap_infos) - index - true_positives
            false_negatives = missed_ground_truths + true_positives_up_to[index]

            if true_positives + false_positives == 0:
                current_precision = 0
            else:
                current_precision = true_positives / float(true_positives + false_positives)

            if true_positives + false_negatives == 0:
                current_recall = 0
            else:
                current_recall = true_positives / float(true_positives + false_negatives)

            precision += [current_precision]
            recall += [current_recall]

        # insert "if you don't do anything, you cannot do it wrong" point
        precision.append(1)
        recall.append(0)

        # recall needs to be sorted ascending for auc
        precision.reverse()
        recall.reverse()

        # AP is the area under curve (auc) of recall-precision
        average_precision = sklearn.metrics.auc(recall, precision)

        if args.plot:
            label_name = labels[label_id]['name']
            plot_precision_recall(
                precision,
                recall,
                average_precision,
                threshold,
                label_name,
                os.path.join(args.plot_dir, '{}_{:d}.{}'.format(label_name, int(threshold * 100), args.plot_extension))
            )

        precisions[label_id] += [average_precision]
        if threshold == .5:
            precisions_50[label_id] += [average_precision]

    for label_id in precisions.keys():
        if len(precisions[label_id]) > 0:
            precisions[label_id] = np.average(precisions[label_id])
        else:
            precisions[label_id] = float('nan')

    for label_id in precisions_50.keys():
        if len(precisions_50[label_id]) > 0:
            precisions_50[label_id] = np.average(precisions_50[label_id])
        else:
            precisions_50[label_id] = float('nan')

    return precisions, precisions_50
