
from __future__ import print_function
import os
import itertools
import pickle
from multiprocessing.pool import Pool
from pprint import pprint
import warnings
import traceback

import numpy as np
from PIL import Image

from mapillary_vistas.helpers.progress import progress
from mapillary_vistas.helpers.plot import plot_confusion_matrix
from mapillary_vistas.helpers.print import print_confusion_matrix, print_ious, print_precisions
from mapillary_vistas.evaluation.confusion_matrix import calculate_confusion_matrix_from_arrays
from mapillary_vistas.evaluation.confusion_matrix import reduce_evaluation_to_evaluated_categories
from mapillary_vistas.evaluation.confusion_matrix import reduce_evaluation_to_metalevel
from mapillary_vistas.evaluation.instance_specific_pixel_level import calculate_instance_specific_pixel_accuracy_from_arrays
from mapillary_vistas.evaluation.instance_specific_instance_level import calculate_instance_specific_instance_accuracy_from_arrays, calculate_average_precision


def process_image(labels, files):
    """
    Calculate all metrics for the given files, using the given labels.
    """

    confusion_matrix = None
    instance_specific_instance_information = None
    instance_specific_pixel_information = {}
    try:
        ground_truth_path = files['ground_truth']
        prediction_path = files['prediction']
        instance_predictions_path = files['instances']

        ground_truth_image = Image.open(ground_truth_path)
        ground_truth_array = np.array(ground_truth_image)
        ground_truth_instance_array = None
        if ground_truth_array.dtype != np.uint8:
            # split labels and store separately
            ground_truth_label_array = np.array(ground_truth_array / 256, dtype=np.uint8)
            ground_truth_instance_array = ground_truth_array
        else:
            ground_truth_label_array = ground_truth_array
            warn_text = """You specified 8bit label files as ground truth.
It is not possible to derive instance specific metrics from these images.
Specify the 16bit instance files."""
            if instance_predictions_path is not None:
                raise RuntimeError(warn_text)
            else:
                warnings.warn(warn_text, RuntimeWarning)

        if prediction_path is not None:
            prediction_image = Image.open(prediction_path)
            prediction_array = np.array(prediction_image)
            assert prediction_array.dtype == np.uint8

            confusion_matrix = calculate_confusion_matrix_from_arrays(
                prediction_array,
                ground_truth_label_array,
                len(labels)
            )

            if ground_truth_instance_array is not None:
                instance_specific_pixel_information = calculate_instance_specific_pixel_accuracy_from_arrays(
                    prediction_array,
                    ground_truth_instance_array,
                    labels
                )

        if instance_predictions_path is not None and ground_truth_instance_array is not None:
            instance_specific_instance_information = calculate_instance_specific_instance_accuracy_from_arrays(
                instance_predictions_path,
                ground_truth_instance_array,
                labels
            )
    except:
        traceback.print_exc()
        print("problem in processing {}, {}".format(prediction_path, ground_truth_path))
        raise

    return (confusion_matrix, instance_specific_pixel_information, instance_specific_instance_information)


def process_image_unpack_args(args):
    """
    Wrapper to make imap work.
    """
    return process_image(*args)


def add_result(return_value, confusion_matrix, instance_specific_pixel_information, instance_specific_instance_information):
    """
    Add the result of one image pair to the result structures.
    """

    result, pixel_information, instance_information = return_value
    if confusion_matrix is None:
        confusion_matrix = result
    elif result is not None:
        confusion_matrix += result

    for label, values in pixel_information.items():
        instance_specific_pixel_information[label]['raw_true_positives'] += values['raw_true_positives']
        instance_specific_pixel_information[label]['weighted_true_positives'] += values['weighted_true_positives']
        instance_specific_pixel_information[label]['raw_false_negatives'] += values['raw_false_negatives']
        instance_specific_pixel_information[label]['weighted_false_negatives'] += values['weighted_false_negatives']

    if instance_information is not None:
        instance_specific_instance_information += [instance_information]

    return (confusion_matrix, instance_specific_pixel_information, instance_specific_instance_information)


def evaluate_dirs(labels, args):
    """
    Evaluate the given command line parameters and print/plot the results.
    """

    prediction_dir = None
    if args.prediction_labels is not None:
        prediction_dir = os.path.abspath(args.prediction_labels)
    ground_truth_label_dir = os.path.abspath(args.ground_truth_labels)

    instance_dir = None
    if args.instances is not None:
        instance_dir =  os.path.abspath(args.instances)
    jobs = args.jobs

    print("Parsing directories for images...")
    image_tuples = []
    for (path, _, files) in os.walk(ground_truth_label_dir):
        for ground_truth_file in files:

            # ignore non png predictions
            if not ground_truth_file.endswith('.png'):
                continue

            ground_truth_path = os.path.join(path, ground_truth_file)

            prediction_path = None
            if prediction_dir is not None:
                prediction_path = ground_truth_path.replace(
                    ground_truth_label_dir,
                    prediction_dir,
                )

            if instance_dir is not None:
                instance_path = ground_truth_path.replace(
                    ground_truth_label_dir,
                    instance_dir
                )
                instance_path = os.path.splitext(instance_path)[0] + '.txt'
            else:
                instance_path = None

            image_tuples.append({
                'ground_truth': ground_truth_path,
                'prediction': prediction_path,
                'instances': instance_path,
            })

    print("Found {} predictions with ground truth".format(len(image_tuples)))

    # initialize result structures
    confusion_matrix = None
    instance_specific_pixel_information = {}
    instance_specific_instance_information = []
    for label in labels:
        instance_specific_pixel_information[label['name']] = {
            'raw_true_positives': 0,
            'weighted_true_positives': 0,
            'raw_false_negatives': 0,
            'weighted_false_negatives': 0,
        }

    print("Analysing predictions")
    if jobs == 1:
        # if only one job is allowed, use the current process to
        # improve signal handling and backtrace information in case
        # of errors
        for files in progress(image_tuples):
            result = process_image(labels, files)
            confusion_matrix, \
                instance_specific_pixel_information, \
                instance_specific_instance_information = \
                add_result(
                    result,
                    confusion_matrix,
                    instance_specific_pixel_information,
                    instance_specific_instance_information
                )
    else:
        # jobs can be a number or None (in which case all cores will be used)
        pool = Pool(processes=jobs)
        pool_args = zip(itertools.repeat(labels, len(image_tuples)), image_tuples)
        results = pool.imap_unordered(process_image_unpack_args, pool_args)
        for result in progress(results, total=len(image_tuples)):
            confusion_matrix, \
            instance_specific_pixel_information, \
            instance_specific_instance_information = \
                add_result(
                    result,
                    confusion_matrix,
                    instance_specific_pixel_information,
                    instance_specific_instance_information
                )
        pool.close()
        pool.join()

    if len(instance_specific_instance_information) > 0:
        print("Calculating instance specific accuracy")
        precisions, precisions_50 = calculate_average_precision(instance_specific_instance_information, labels, args)
        print_precisions(labels, precisions, precisions_50)

    if confusion_matrix is not None:
        # print the results according to command line parameters
        reduced_labels, reduced_confusion_matrix, reduced_instance_specific_pixel_information = reduce_evaluation_to_evaluated_categories(labels, confusion_matrix, instance_specific_pixel_information)

        if args.print_absolute_confusion_matrix:
            percentage = False
        else:
            percentage = True

        if args.print_full_confusion_matrix:
            labels_for_printing = labels
            confusion_matrix_for_printing = confusion_matrix
            instance_specific_information_for_printing = reduced_instance_specific_pixel_information
        else:
            labels_for_printing = reduced_labels
            confusion_matrix_for_printing = reduced_confusion_matrix
            instance_specific_information_for_printing = instance_specific_pixel_information

        if args.plot:
            plot_confusion_matrix(labels, confusion_matrix, args.plot_dir, "confusion_matrix", args.plot_extension)
        print_confusion_matrix(labels_for_printing, confusion_matrix_for_printing, percent=percentage)
        print_ious(labels_for_printing, confusion_matrix_for_printing, instance_specific_information_for_printing)

        meta_labels, meta_confusion_matrix, meta_instance = reduce_evaluation_to_metalevel(labels_for_printing, confusion_matrix_for_printing, instance_specific_information_for_printing, 2)
        if args.plot:
            plot_confusion_matrix(meta_labels, meta_confusion_matrix, args.plot_dir, "confusion_matrix_meta_2", args.plot_extension)
        print_confusion_matrix(meta_labels, meta_confusion_matrix, percent=percentage)
        print_ious(meta_labels, meta_confusion_matrix, meta_instance)

        meta_labels, meta_confusion_matrix, meta_instance = reduce_evaluation_to_metalevel(labels_for_printing, confusion_matrix_for_printing, instance_specific_information_for_printing, 1)
        if args.plot:
            plot_confusion_matrix(meta_labels, meta_confusion_matrix, args.plot_dir, "confusion_matrix_meta_1", args.plot_extension)
        print_confusion_matrix(meta_labels, meta_confusion_matrix, percent=percentage)
        print_ious(meta_labels, meta_confusion_matrix, meta_instance)
