import torch
from rich import print
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import silhouette_score
from tqdm import tqdm
from ravl.discover_utils import (
    compute_cluster_performance_gaps, 
    computer_cluster_influence_scores
)

def cluster(reg_emb, n_clusters):
    """
    Cluster visually similar regions. 

    Parameters: 
        reg_emb (np.Array): Visual embeddings for each region
        n_clusters (list): Candidate cluster numbers
    Returns:
        clusters (np.Array): Cluster assignments for each region
        medoids (np.Array): Region indices corresponding to each cluster medoid
    """
    print("=> Starting RaVL Discovery - Step 1: Cluster visually similar regions")

    # Compute pairwise distances between regions
    dist = 1.0 - (reg_emb @ reg_emb.T)
    np.clip(dist, 0, 1, out=dist)
    np.fill_diagonal(dist, 0)

    # Identify best cluster number by sweeping cluster sizes
    print(f"=> Running KMedoids with {min(n_clusters)} to {max(n_clusters)} clusters")
    best = (0, 0)
    for c in tqdm(n_clusters):
        model = KMedoids(n_clusters=c, random_state=0, metric="precomputed", init="k-medoids++")
        kmedoids = model.fit(dist)
        clusters = kmedoids.labels_

        silhouette = silhouette_score(dist, clusters, metric="precomputed")
        if silhouette > best[0]:
            best = (silhouette, c)

    # Generate clusters
    model = KMedoids(n_clusters=best[1], random_state=0, metric="precomputed", init="k-medoids++")
    kmedoids = model.fit(dist)
    clusters = kmedoids.labels_

    # Compute clustering accuracy
    sil = silhouette_score(dist, clusters, metric="precomputed")
    print(f"Selected {best[1]} clusters (silhouette score = {str(round(sil,3))})")

    return clusters, kmedoids.medoid_indices_

def identify(
    clusters,
    img_true_label,
    img_pred_label,
    img_score_dist,
    reg_score_dist,
    num_regions,
):
    """
    Identify image features contributing to classification errors. 
    
    Parameters: 
        clusters (np.Array): Cluster assignments for each region
        img_true_label (np.Array): True label for each image
        img_pred_label (np.Array): Zero-shot predicted label for each image
        img_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each image
        reg_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each region
        num_regions (np.Array): Number of regions in each image
    Returns:
        filtered_cluster_idx (list): Cluster indices after filtering out low influence scores and 
                                    ranking by cluster performance gaps
        filtered_tables (list): List of formatted tables with (unweighted) cluster performance gaps for each cluster
    """

    print("=> Starting RaVL Discovery - Step 2: Identify image features contributing " + 
          "to classification errors")

    num_imgs = len(img_true_label)
    num_regs = sum(num_regions)
    img_idx_to_reg_idx = dict(
        zip(np.arange(num_imgs), torch.split(torch.arange(num_regs), num_regions))
    )
    reg_idx_to_img_idx = {a.item(): k for k, v in img_idx_to_reg_idx.items() for a in v}

    results = {
        "cluster_performance_gaps": [], 
        "influence_scores": [], 
        "satisfy_spurious_criteria": [], 
        "tables": []
    }
    # Iterate through each cluster of features
    for c in sorted(set(clusters)):
        # Identify region indices in the cluster
        region_idx = np.where(clusters == c)[0]

        # Obtain predicted and ground-truth labels for images associated with cluster
        image_idx_incluster = sorted(
            list(set([reg_idx_to_img_idx[i] for i in region_idx]))
        )
        img_pred_incluster = img_pred_label[image_idx_incluster]
        img_label_incluster = img_true_label[image_idx_incluster]
        all_img_labels = list(set(img_label_incluster))

        # Obtain predicted and ground-truth labels for images outside cluster that 
        # share ground-truth labels with cluster
        image_idx_outcluster = [
            i
            for i in range(len(img_true_label))
            if (img_true_label[i] in all_img_labels and i not in image_idx_incluster)
        ]
        img_pred_outcluster = img_pred_label[image_idx_outcluster]
        img_label_outcluster = img_true_label[image_idx_outcluster]
        assert (
            len(set(image_idx_outcluster).intersection(set(image_idx_incluster))) == 0
        )

        # Compute cluster performance gaps
        cluster_performance_gaps, table = compute_cluster_performance_gaps(
            img_label_incluster,
            img_pred_incluster,
            img_label_outcluster,
            img_pred_outcluster,
            all_img_labels,
        )

        # Compute cluster influence scores
        influence_score = computer_cluster_influence_scores(
            region_idx,
            cluster_performance_gaps,
            img_idx_to_reg_idx, 
            img_true_label,
            img_pred_label,
            img_score_dist,
            reg_idx_to_img_idx,
            reg_score_dist,
        )

        perf_gap_score = sum([np.abs(v) for k,v in cluster_performance_gaps.items()])
        results["cluster_performance_gaps"].append(perf_gap_score)
        results["influence_scores"].append(influence_score)
        results["satisfy_spurious_criteria"].append(
            len([v for k,v in cluster_performance_gaps.items() if v > 0]) > 0 and 
            len([v for k,v in cluster_performance_gaps.items() if v < 0]) > 0 
        )
        results["tables"].append(table)

    # Identify clusters that contain spurious features
    argsort = np.argsort(np.array(results["cluster_performance_gaps"]))[::-1]
    filtered_cluster_idx = [
        a for a in argsort 
        if results['satisfy_spurious_criteria'][a] is True and results["influence_scores"][a] >= 0.25
    ]
    filtered_tables = [results['tables'][i] for i in filtered_cluster_idx]

    return filtered_cluster_idx, filtered_tables
    
def rank(
    reg_emb, 
    clusters,        
    medoids, 
    filtered_cluster_idx, 
):
    """
    Rank image features by degree of learned spurious correlation. 
    
    Parameters: 
        reg_emb (np.Array): Visual embeddings for each region
        clusters (np.Array): Cluster assignments for each region
        medoids (np.Array): Region indices corresponding to each cluster medoid
        filtered_cluster_idx (list): Cluster indices after filtering out low influence scores and 
                                    ranking by cluster performance gaps
    Returns:
        ranked_list (np.Array): Ranked region indices associated with each cluster
    """

    print("=> Starting RaVL Discovery - Step 3: Rank image features by " + 
          "degree of learned spurious correlation")

    # Obtain ranking of spurious regions
    ranked_list = []
    for c in filtered_cluster_idx:
        medoid = reg_emb[medoids[c]].reshape(1, -1)
        cluster_region_idx = np.where(clusters == c)[0]
        cluster_region_emb = reg_emb[cluster_region_idx]
        sim = np.argsort((cluster_region_emb @ medoid.T).flatten())[::-1]
        ranked_list.append(cluster_region_idx[sim].tolist())
    
    return ranked_list


def discover(
    reg_embs,
    img_true_label,
    img_pred_label,
    img_score_dist,
    reg_score_dist,
    num_regions,
):
    """
    Run RaVL discovery. 
    
    Parameters: 
        reg_embs (np.Array): Visual embeddings for each region
        img_true_label (np.Array): True label for each image
        img_pred_label (np.Array): Zero-shot predicted label for each image
        img_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each image
        reg_score_dist (np.Array): Zero-shot score distribution (after softmax) across classes for each region
        num_regions (np.Array): Number of regions in each image
    Returns:
        ranked_region_idx[0] (np.Array): Region indices associated with the top ranked feature cluster
        filtered_tables[0] (prettytable.PrettyTable): Unweighted cluster performance gaps for the top
                                                      ranked feature cluster organized in tabular format
    """
    
    num_eval_classes = img_score_dist.shape[1]
    
    # Discovery - Step 1: Cluster visually similar regions
    clusters, medoids = cluster(
        reg_embs,
        np.arange(num_eval_classes * 2, num_eval_classes * 5),
    )

    # Discovery - Step 2: Identify spurious correlations
    filtered_cluster_idx, filtered_tables = identify(
        clusters,
        img_true_label,
        img_pred_label,
        img_score_dist,
        reg_score_dist,
        num_regions,
    )

    # Discovery - Step 3: Rank image features by degree of learned spurious correlation
    ranked_region_idx = rank(
        reg_embs, 
        clusters,        
        medoids, 
        filtered_cluster_idx, 
    )
    
    # Return top ranked cluster
    return ranked_region_idx[0], filtered_tables[0]
