# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import print_function

import math
import numpy as np

from mapillary_vistas.evaluation.confusion_matrix import calculate_iiou


def get_max_length(labels):
    max_length = 0
    for label in labels:
        current_length = len(label['name'])
        if current_length > max_length:
            max_length = current_length

    return max_length + 1


def print_confusion_matrix(labels, confusion_matrix, percent=False):
    max_length = get_max_length(labels)

    print('')
    print('')
    print('')

    header = "{:>{ml}} |".format("", ml=max_length+1)
    for label in labels:
        header += "{:>{ml}} |".format(label["name"], ml=max_length)
    print(header)

    for index, label in enumerate(labels):
        current_line = "{:>{ml}}: |".format(label["name"], ml=max_length)
        divider = float(np.sum(confusion_matrix[index, :]))
        for inner_index, _ in enumerate(labels):
            if percent:
                if divider == 0:
                    metric = float('nan')
                else:
                    metric = confusion_matrix[index, inner_index] / divider
                value = "{:.2%}".format(metric)
            else:
                value = "{:d}".format(confusion_matrix[index, inner_index])
            current_line += "{:>{ml}} |".format(value, ml=max_length)
        print(current_line)

    print()


def print_ious(labels, confusion_matrix, instance_information=None):
    max_length = get_max_length(labels)
    iious = calculate_iiou(labels, confusion_matrix, instance_information)

    print('')
    print('')
    print('')

    header = "{:>{ml}}".format("", ml=max_length+1)
    header += " {:^6} |".format("IoU")
    header += " {:^6} |".format("iIoU")
    print(header)

    iou_values = []
    iiou_values = []
    for label, iiou in zip(labels, iious):
        if not math.isnan(iiou[0]):
            iou_values += [iiou[0]]
        current_line = "{:>{ml}}:".format(label["name"], ml=max_length)
        current_line += " {:>6} |".format("{:.1%}".format(iiou[0]))
        if label['instances']:
            if not math.isnan(iiou[1]):
                iiou_values += [iiou[1]]
            current_line += " {:>6} |".format("{:.1%}".format(iiou[1]))

        print(current_line)

    print('')
    print("{:>{ml}}: {:>6} | {:>6} |".format(
        "Avg",
        "{:.1%}".format(np.average(iou_values)),
        "{:.1%}".format(np.average(iiou_values)),
        ml=max_length
    ))
    print('')
    print('')


def print_precisions(labels, precisions, precisions_50):
    max_length = get_max_length(
        [label for index, label in enumerate(labels) if index in precisions.keys()]
    )

    print('')
    print('')
    print('')

    header = "{:>{ml}}".format("", ml=max_length+1)
    header += " {:^6} |".format("AP")
    print(header)

    iou_values = []
    iiou_values = []
    for label_id, label in enumerate(labels):
        if label_id not in precisions:
            continue
        precision = precisions[label_id]
        current_line = "{:>{ml}}:".format(labels[label_id]["name"], ml=max_length)
        current_line += " {:>6} |".format("{:.1%}".format(precision))
        if label_id in precisions_50:
            current_line += " {:>6} |".format("{:.1%}".format(precisions_50[label_id]))

        print(current_line)

    valid_values = [value for value in precisions.values() if not math.isnan(value)]
    valid_values_50 = [value for value in precisions_50.values() if not math.isnan(value)]

    print('')
    print("{:>{ml}}: {:>6} | {:>6} |".format(
        "Avg",
        "{:.1%}".format(np.average(valid_values)),
        "{:.1%}".format(np.average(valid_values_50)),
        ml=max_length
    ))
    print('')
    print('')
