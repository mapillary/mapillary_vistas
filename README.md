# Mapillary Vistas Dataset (MVD) Evaluation Scripts

This repository contains scripts to run evaluations on MVD predictions (e.g. on training and on validation data), and additionally form the basis of our evaluation server. Consequently, successfully running these scripts on machine-generated results helps to validate the predictor's output format required by the evaluation server.


## Usage

The main entry point is `mapillary_vistas/tools/evaluation.py`.

The following command line options are mandatory:

- `-c`, `--config` provide location of the config file distributed with your downloaded copy of MVD. 
- `-g`, `--ground-truth-labels` provide path to the instance-specific ground truth label files distributed with MVD.
- `-p`, `--prediction-labels` provide path to your predictions. The expected files are 8bit grayscale encoding in PNG format and pixel values are treated as indices for the label list in the config file. 

Both flags `-l` and `-p` assume that all generated files are located within the same, specified folder location.
The most important optional command line parameters are:

- `-i`, `--instances` provide path to the instance-specific prediction folder. For every ground truth file, the script looks for a file with the same name in the `--instances` folder, but with `.txt` extension. Every prediction for an image is stored in a separate line with format `<instance mask> <label id> <confidence>`. The path to `<instance mask>` is relative to the text file, `<label_id>` is an integer and `<confidence>` is a floating point number.

- `-j`, `--jobs` defines the number of processes to run script in parallel. It has the same syntax as `make`. Without this flag, only one process is used. If no number is provided, the number of available CPU cores is used.

- `--plot` If activated, the confusion matrix and (in case of instance-specific evaluation), the precision-recall plots for every class and different thresholds are plotted and saved to files.

- `--plot-extension` specifies the file extension of the plots. Use `svg` for scalable versions and `png` for rasterized plots.

- `--plot-dir` path to the plotted files. If not specified, the current directory is used.

For full command line usage, please run `python mapillary_vistas/tools/evaluation.py --help`

## Metrics

### Semantic image segmentation

The main metric used for semantic image segmentation is the mean intersection over union (IoU) score [1], also known as averaged Jaccard index. IoU is defined as TP / (TP + FP + FN), where TP, FP and FN are true positives, false positives and false negatives, respectively.

For categories where we have instance-specific annotations available, we also provide instance-specific intersection over union (iIoU) [2] scores, applying re-scaling of TP and FN during IoU calculation. The scaling factor is the ratio of the current object size to the average object size of the respective class. Consequently, this metric reduces the bias towards object classes with dominant segmentation sizes. This metric is not to be confused with the metric used below for instance-specific segmenation assessment, as it does not require information about individual object instances.

### Instance-specific semantic image segmentation

For instance-specific semantic image segmentation we calculate average precision (AP) scores, following [3]. We analyze each overlapping prediction per available ground truth instance. The prediction with the highest confidence for a given, minimal overlap is assigned to the ground truth instance. Unassigned ground truth instances count as false negatives (FN), unassigned predictions as false positives (FP), respectively.
Based on these assignments we generate a precision-recall curve for each label.
The average precision corresponds to the area under curve of the plot.
Similar to [2,4], we calculate the main metric, AP, by using thresholds from 50% to 95% in 5% steps and finally average the individual APs.

## References

[1] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn and A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. In _International Journal of Computer Vision_ (IJCV), 2010.

[2] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The Cityscapes Dataset for Semantic Urban Scene Understanding. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016.

[3] B. Hariharan, P. Arbeláez, R. B. Girshick, and J. Malik. Simultaneous detection and segmentation. In _European Conference on Computer Vision (ECCV)_, 2014.

[4] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft COCO: Common Objects in Context. In _European Conference on Computer Vision (ECCV)_, 2014.
