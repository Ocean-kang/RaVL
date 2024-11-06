import numpy as np
import torch
from collections import defaultdict
from prettytable import PrettyTable


def compute_cluster_performance_gaps(
    incluster_label,
    incluster_pred,
    outcluster_label,
    outcluster_pred,
    class_labels,
):
    """
    Compute cluster performance gap metric (as defined in Section 3.1 of our paper).

    Parameters: 
        incluster_label (np.Array): True labels for images with at least one region in the cluster of interest
        incluster_pred (np.Array): Predicted labels for images with at least one region in the cluster of interest
        outcluster_label (np.Array): True labels for images with no regions in the cluster of interest
        outcluster_pred (np.Array): Predicted labels for images with no regions in the cluster of interest
        class_labels (list): All true image labels associated with the cluster of interest
    Returns:
        class_performance_gaps (np.Array): Cluster performance gaps for each class
        table (prettytable.PrettyTable): Unweighted cluster performance gaps organized in tabular format 
    """
    table = PrettyTable(["image-label", "acc-present", "acc-absent"])

    class_performance_gaps = {}
    for l in class_labels:
        incluster_idx = np.where(incluster_label == l)[0]
        if len(incluster_idx) == 0: continue
        incluster_acc = (incluster_pred[incluster_idx] == l).mean()

        outcluster_idx = np.where(outcluster_label == l)[0]
        if len(outcluster_idx) == 0: continue
        outcluster_acc = (outcluster_pred[outcluster_idx] == l).mean()

        total_samples = len(incluster_idx) + len(outcluster_idx)
        prob = min(len(incluster_idx), len(outcluster_idx)) / total_samples 
        weight = prob * 2 

        class_performance_gaps[l] = weight * (incluster_acc - outcluster_acc)

        table.add_row(
            [
                l,
                f"{round(incluster_acc,3)} ({len(incluster_idx)} samples)",
                f"{round(outcluster_acc,3)} ({len(outcluster_idx)} samples)",
            ]
        )

    return class_performance_gaps, table


def computer_cluster_influence_scores(
    region_idx,
    class_performance_gaps,
    img_idx_to_reg_idx, 
    img_true_label,
    img_pred_label,
    img_score_dist,
    reg_idx_to_img_idx,
    reg_score_dist,
):
    """
    Compute cluster influence score (as defined in Section 3.1 of our paper).

    Parameters: 
        region_idx (np.Array): Indices for regions in the cluster of interest 
        class_performance_gaps (dict): Cluster performance gaps for each class
        img_idx_to_reg_idx (dict): Mapping from image indices to region indices
        img_true_label (np.Array): True label for each image
        img_pred_label (np.Array): Zero-shot predicted label for each image
        img_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each image
        reg_idx_to_img_idx (dict): Mapping from region indices to image indices 
        reg_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each region
    Returns:
        influence_score (float): Cluster influence score for the cluster of interest
    """

    infl_err = defaultdict(list)

    img_idx = list(set([reg_idx_to_img_idx[r] for r in region_idx]))
    for i in img_idx:
        # We do not consider cases where the image only has one region
        if len(img_idx_to_reg_idx[i]) == 1:
            continue

        # Identify classification score distributions for the image and all regions in the image
        img_dist = img_score_dist[i]
        all_reg_idx = img_idx_to_reg_idx[i]
        reg_dist = torch.stack([reg_score_dist[r] for r in all_reg_idx])

        pred_cls_idx = img_dist.argmax()
        best_reg_idx = all_reg_idx[reg_dist[:, pred_cls_idx].argmax().item()].item()
        pres = best_reg_idx in region_idx

        correct = img_true_label[i] == img_pred_label[i]
        if (
            not correct and 
            img_true_label[i] in class_performance_gaps and 
            class_performance_gaps[img_true_label[i]] < 0
        ):
            infl_err[img_true_label[i]].append(pres)

    # Compute final influence score
    infl_err = {k: np.mean(v) for k, v in infl_err.items()}
    if len(infl_err) > 0: 
        return np.max([infl_err[i] for i in infl_err])
    else: 
        return -np.inf
