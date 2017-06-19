"""
    This file contains all the confusion matrix handling code.
"""

from __future__ import print_function
import itertools
import numpy as np


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    """
    calculate the confusion matrix for one image pair.
    prediction and ground_truth have to have the same shape.
    """

    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T

    # add up confusion matrix
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    """
    calculate IoU (intersecion over union) for a given confusion matrix.
    """

    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives

        denom = true_positives + false_positives + false_negatives

        # no entries, no iou..
        if denom == 0:
            iou = float('nan')
        else:
            iou = float(true_positives)/denom

        ious.append(iou)

    return ious


def calculate_iiou(labels, confusion_matrix, instance_information):
    """
    Calculates the iIoU (instance specific intersecion over union) and IoU.
    The true positives and false negatives are scaled by the average
    instance size of the category. Scaling factors are applied outside
    of this function.
    """

    ious = calculate_iou(confusion_matrix)
    iious = []
    for index, (label, iou) in enumerate(zip(labels, ious)):
        if not label['instances']:
            iious.append((iou, None, None))
            continue

        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives

        weighted_true_positives = instance_information[label['name']]['weighted_true_positives']
        weighted_false_negatives = instance_information[label['name']]['weighted_false_negatives']

        denom = weighted_true_positives + weighted_false_negatives + false_positives

        if denom == 0:
            weighted_iiou = float('nan')
        else:
            weighted_iiou = weighted_true_positives / denom

        iious.append((iou, weighted_iiou))

    return iious


def add_to_confusion_matrix(prediction, ground_truth, number_labels, confusion_matrix=None):
    """
    This function is a handy shortcut to update a confusion matrix with new
    data. If a confusion matrix is given, the new hits are added to it.
    """
    matrix_update = calculate_confusion_matrix_from_arrays(
        prediction,
        ground_truth,
        number_labels
    )

    if confusion_matrix is None:
        confusion_matrix = matrix_update
    else:
        confusion_matrix += matrix_update

    return confusion_matrix


def reduce_evaluation_to_evaluated_categories(labels, confusion_matrix, instance_specific_information):
    """
    Delete rows/cols in the confusion matrix that are not used for evaluation.
    Also unused instance specific information is removed.
    """

    new_instance_specific_information = dict(instance_specific_information)
    new_labels = []
    eval_indices = []
    for label in labels:
        if not label['evaluate']:
            eval_indices.append(False)
            if label['name'] in new_instance_specific_information:
                new_instance_specific_information.pop(label['name'])
            continue

        eval_indices.append(True)
        new_labels.append(label)

    eval_indices = np.array(eval_indices, dtype=np.bool)

    reduced_confusion_matrix = confusion_matrix[eval_indices, :][:, eval_indices]

    return (new_labels, reduced_confusion_matrix, new_instance_specific_information)


def reduce_evaluation_to_metalevel(labels, confusion_matrix, instance_specific_information, level):
    """
    Combine and add up rows/cols in common meta categories.
    Level specifies how many levels should be left (1 or 2)
    """

    max_depth = 0
    for label in labels:
        depth = len(label['name'].split('--'))
        if depth > max_depth:
            max_depth = depth

    if max_depth < level:
        return (labels, confusion_matrix)

    new_labels = []
    new_instance_specific_information = {}
    mapping = {}
    labels_by_name = {}
    ids_by_name = {}
    for index, label in enumerate(labels):
        old_name = label['name']
        labels_by_name[old_name] = label
        ids_by_name[old_name] = index
        levels = old_name.split('--')
        new_name = "--".join(levels[:level])

        if new_name not in mapping:
            mapping[new_name] = []

        mapping[new_name] += [old_name]

    id_mapping = []
    for new_name, old_names in sorted(mapping.items(), key=lambda entry: entry[0]):
        evaluate = True
        instances = True
        color = None
        readable = None

        new_instance_specific_information[new_name] = {
            'raw_true_positives': 0,
            'weighted_true_positives': 0,
            'raw_false_negatives': 0,
            'weighted_false_negatives': 0,
        }

        current_ids = []
        for old_name in old_names:
            current_ids += [ids_by_name[old_name]]
            old_label = labels_by_name[old_name]
            if not old_label['evaluate']:
                evaluate = False
            if not old_label['instances']:
                instances = False
            if color is None:
                color = old_label['color']
            if readable is None:
                readable = old_label['readable']

            # skip if no instance specific information is available
            if old_name in instance_specific_information:
                new_instance_specific_information[new_name]['raw_true_positives'] += instance_specific_information[old_name]['raw_true_positives']
                new_instance_specific_information[new_name]['weighted_true_positives'] += instance_specific_information[old_name]['weighted_true_positives']
                new_instance_specific_information[new_name]['raw_false_negatives'] += instance_specific_information[old_name]['raw_false_negatives']
                new_instance_specific_information[new_name]['weighted_false_negatives'] += instance_specific_information[old_name]['weighted_false_negatives']

        id_mapping += [current_ids]

        new_labels.append({
            'name': new_name,
            'readable': readable,
            'evaluate': evaluate,
            'color': color,
            'instances': instances,
        })

    nr_labels = len(new_labels)
    reduced_confusion_matrix = np.zeros((nr_labels, nr_labels), dtype=np.uint32)
    for i, j in itertools.product(range(nr_labels), range(nr_labels)):
        reduced_confusion_matrix[i, j] = confusion_matrix[id_mapping[i], :][:, id_mapping[j]].sum()

    return (new_labels, reduced_confusion_matrix, new_instance_specific_information)
