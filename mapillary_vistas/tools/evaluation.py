
import os
import sys
import json
import argparse
from argparse import RawTextHelpFormatter

# enable absolute mapillary_vistas imports
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from mapillary_vistas.evaluation.evaluation import evaluate_dirs


def main():
    parser = argparse.ArgumentParser(description="Commandline tool for evaluating predictions", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-c',
        '--dataset-config',
        type=argparse.FileType(),
        action='store',
        required=True,
        help="config.json of the dataset. Used for label names")

    parser.add_argument(
        '-g',
        '--ground-truth-labels',
        type=str,
        action='store',
        required=True,
        help="Path to the folder with png ground truth label data.")

    parser.add_argument(
        '-p',
        '--prediction-labels',
        type=str,
        action='store',
        required=True,
        help="Path to the folder with predicted label data. "
             "(PNG files, file name must be exactly the same as ground truth data)")

    parser.add_argument(
        '-i',
        '--instances',
        type=str,
        action='store',
        required=False,
        default=None,
        help="""Path to the folder with instance description.
(TXT files, file name must be exactly the same as ground truth except for the extension)
Every line must have the following format:
<path to instance binary mask> (<label id> | <label name>) <confidence score>
The paths to masks are relative to the TXT file.
Note that you need instance specific ground truth labels in order to evaluate on instances
""")

    parser.add_argument(
        '-j',
        '--jobs',
        type=int,
        action='store',
        required=False,
        default=1,
        const=None,
        nargs='?',
        help="""Number of processes to run in parallel.
Not specifying a number will use all available cores""")

    parser.add_argument(
        '--plot',
        action='store_true',
        help="Create plots of confusion matrix and precision recall diagrams")

    parser.add_argument(
        '--plot-extension',
        type=str,
        action='store',
        required=False,
        default='png',
        help="Extension of plots (png, svg, ...)")

    parser.add_argument(
        '--plot-dir',
        type=str,
        action='store',
        required=False,
        default=os.path.curdir,
        help="Directory to store plots, defaults to current directory")

    parser.add_argument(
        '--print-absolute-confusion-matrix',
        action='store_true',
        help="If this parameter is given, the absolute pixel values in the confusion matrix will be printed. Otherwise each row is normalized")

    parser.add_argument(
        '--print-full-confusion-matrix',
        action='store_true',
        help="If this parameter is given, the full confusion matrix is shown in the output. Otherwise only labels with the 'evaluate' flag set are printed")

    parser.add_argument(
        '--print-all-ious',
        action='store_true',
        help="If this parameter is given, the iou of all labels is shown in the output. Otherwise only labels with the 'evaluate' flag set are printed")


    args = parser.parse_args()

    config = json.load(args.dataset_config)
    labels = config['labels']

    if not os.path.isdir(args.prediction_labels):
        raise RuntimeError("Prediction directory does not exist!")

    if not os.path.isdir(args.ground_truth_labels):
        raise RuntimeError("Ground turth directory does not exist!")

    evaluate_dirs(
        labels,
        args,
    )

import cProfile, pstats, StringIO

if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'time'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
