from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_image']
    pr = results[obj]['pred_image']
    gt = np.array(gt)
    pr = np.array(pr)
    try:
        if metric == 'image-auroc':
            performance = roc_auc_score(gt, pr)
        else:  # image-ap
            performance = average_precision_score(gt, pr)
        return performance
    except ValueError:
        return -1


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['gt_pixel']
    pr = results[obj]['pred_pixel']
    gt = np.array(gt)
    pr = np.array(pr)
    try:
        if metric == 'pixel-auroc':
            performance = roc_auc_score(gt.ravel(), pr.ravel())
        else:  # pixel-ap
            performance = average_precision_score(gt.ravel(), pr.ravel())
        return performance
    except ValueError:
        return -1
