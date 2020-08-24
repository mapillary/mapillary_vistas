# Copyright (c) Facebook, Inc. and its affiliates.

import os
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mapillary_vistas.evaluation.confusion_matrix import calculate_iou


def plot_confusion_matrix(labels, confusion_matrix, directory, name, extension):
    """
    Plots the normalized confusion matrix with the target names as axis ticks.
    """

    ious = calculate_iou(confusion_matrix)

    size = len(labels)/5+2
    fig, ax = plt.subplots(figsize=(size+2, size))
    plot = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, norm=LogNorm())
    # plot.set_clim(vmin=0, vmax=100)

    ticks_with_iou = []
    ticks_without_iou = []
    tick_marks = np.arange(len(ious))
    ious_for_average = []
    for label, iou in zip(labels, ious):
        if math.isnan(iou):
            iou = 0
        else:
            ious_for_average.append(iou)
        ticks_with_iou.append("{}: {:.2%}".format(label['name'], iou))
        ticks_without_iou.append("{}".format(label['name']))

    avg_iou = np.average(ious_for_average)

    fig.colorbar(plot)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(ticks_without_iou, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(ticks_with_iou)
    ax.set_title("Average IoU: {:.2%}".format(avg_iou))

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()

    fig.savefig(os.path.join(directory, '{}.{}'.format(name, extension)))


def plot_precision_recall(precision, recall, average_precision, threshold, label, filename):
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Label: {}, Threshold: {:.2%}, AP: {:.2%}'.format(label, threshold, average_precision))
    fig.savefig(filename)
    plt.close(fig)
