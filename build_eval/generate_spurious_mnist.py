import numpy as np
import pandas as pd
import pyrootutils
import itertools
import shutil
from rich import print
import json
from scipy.stats.contingency import association

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

def get_params():
    """
    Parameters needed to create vision-language fine-tuning datasets for MNIST. The following two
    parameters can be customized: desired_num_images (desired number of images in the fine-tuning
    dataset) and desired_v (level of spurious correlation as measured by Cramer's V). Each parameter is 
    expressed as a list of values, and all combinations of these values are considered when building datasets. 
    
    We also define the spurious visual feature of interest (rectangle in this case) and the class labels.

    Returns: 
        params (dict): Dictionary containing specified parameters
    """

    params = {}
    params["desired_num_images"] = [10000]
    params["desired_v"] = [0.7, 0.8, 0.9, 0.95, 0.99]

    params["spurious_visual_feature"] = "rectangle"
    params["class_labels"] =  ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    return params


def compute_contingency_matrix(
    desired_num_images: int, num_classes: int, desired_v: float, spurious_class_idx: int
):
    """
    Compute frequency matrix with the number of samples needed per class and visual feature
    in order to achieve the desired spurious correlation strength (as measured by Cramer's V)

    Parameters: 
        desired_num_images (int): Desired number of images in the fine-tuning dataset (approximation)
        num_classes (int): Number of class labels
        desired_v (float): Desired strength of spurious correlation, ranging from 0 to 1
        spurious_class_idx (int): Index of class label that is spuriously associated with the rectangle
    Returns: 
        obs (numpy.array): Frequency matrix with the number of samples needed per class and visual feature
        cr_v_score (float): Actual Cramer's V score of the frequency matrix. Should be approximately 
                            equal to desired_v.
    """
    cls_counts = (
        np.ones(num_classes) * (desired_num_images / num_classes)
        + np.random.uniform(low=-20, high=20, size=num_classes)
    ).astype(int)
    num_images = cls_counts.sum()
    desired_chi_sq = (desired_v**2) * num_images

    obs = np.zeros((num_classes, 2))
    obs[:, 0] = cls_counts
    obs[spurious_class_idx] = [0, cls_counts[spurious_class_idx]]

    chi_sq = np.inf
    while chi_sq > desired_chi_sq:
        for idx in range(obs.shape[0]):
            sample = np.random.randint(1, 4)
            if idx == spurious_class_idx:
                obs[idx] += [sample, -sample]
            else:
                obs[idx] += [-sample, sample]

        marginals = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
        exp = (marginals[0] @ marginals[1]) / num_images
        chi_sq = np.sum(((obs - exp) ** 2) / exp)

    obs = obs.astype(int)
    cr_v_score = association(obs, method="cramer")

    return obs, cr_v_score

def sample_ann(base_ann: pd.DataFrame, matrix: np.array, class_labels: list, sp_attr: str):
    """
    Sample from base dataset to match computed frequencies. The generated dataframe will exhibit 
    the desired spurious correlation. 

    Parameters: 
        base_ann (pd.DataFrame): Dataframe associated with the base MNIST dataset
        matrix (np.array): Frequency counts per class and visual feature necessary for desired spurious correlation
        class_labels (list): Class labels associated with dataset
        sp_attr (str): Predefined spurious visual feature
    Returns: 
        sampled_df (pd.DataFrame): Dataframe exhibiting desired spurious correlation
    """

    selected_ids = []
    for idx in range(matrix.shape[0]):
        cls_samples = base_ann[base_ann["true_label"] == class_labels[idx]]

        non_sp = cls_samples[cls_samples["reg_to_attr"].apply(lambda x: [sp_attr] not in x)]
        selected_ids.extend(np.random.choice(non_sp["image_id"].values, matrix[idx, 0], replace=False))

        sp = cls_samples[cls_samples["reg_to_attr"].apply(lambda x: [sp_attr] in x)]
        selected_ids.extend(np.random.choice(sp["image_id"].values, matrix[idx, 1], replace=False))

    selected_ids = set(selected_ids)
    assert len(selected_ids) == matrix.sum()
    sampled_df = base_ann[base_ann["image_id"].apply(lambda x: x in selected_ids)]
    return sampled_df

def main():
    params = get_params()
    np.random.seed(0)

    base_data_dir = root / "data" / "mnist_base"
    base_ann_train = pd.read_feather(base_data_dir / "annotations_train.feather")
    base_ann_test = pd.read_feather(base_data_dir / "annotations_test.feather")

    out_dir_train = root / "data" / "mnist_train"
    out_dir_test = root / "data" / "mnist_test"
    if out_dir_train.exists() and out_dir_train.is_dir():
        shutil.rmtree(out_dir_train)
    if out_dir_test.exists() and out_dir_test.is_dir():
        shutil.rmtree(out_dir_test)
    out_dir_train.mkdir()
    out_dir_test.mkdir()

    # Fine-tuning datasets
    print("Generating fine-tuning datasets")
    counter = 0
    class_labels = params['class_labels']
    spurious_visual_feature = params['spurious_visual_feature']
    for desired_num_images, desired_v in itertools.product(params["desired_num_images"], params["desired_v"]): 
        for spurious_class_idx in range(len(class_labels)): 
            # Generate predefined spurious correlation
            spurious_class = class_labels[spurious_class_idx]
            print(f"=> Dataset {counter}: {desired_num_images} images with a spurious correlation " +
                f"of strength {desired_v} between {spurious_visual_feature} and {spurious_class}")

            # Compute frequency matrix with desired spurious correlation
            matrix, cr_v_score = compute_contingency_matrix(
                desired_num_images, len(class_labels), desired_v, spurious_class_idx
            )

            # Sample from base dataset to match computed frequencies
            ann = sample_ann(base_ann_train, matrix, class_labels, spurious_visual_feature)
            ann.reset_index(drop=True, inplace=True)
            ann = ann.drop(columns=["split"])
            split = np.random.choice(ann["image_id"].values, int(0.05 * ann.shape[0]), replace=False)
            ann["split"] = ann.apply(lambda x: "val" if x.image_id in split else "train", axis=1)

            # Store metadata
            metadata = {}
            metadata["cramer_v_score"] = cr_v_score
            metadata["spurious_textual_attribute"] = spurious_class
            metadata["spurious_visual_feature"] = spurious_visual_feature
            metadata["class_labels"] = class_labels
            metadata["num_images"] = int(matrix.sum())

            # Save files
            ann.to_feather(out_dir_train / f"annotations_{counter}.feather")
            with open(out_dir_train / f"metadata_{counter}.json", "w") as f:
                json.dump(metadata, f)
            counter += 1

    # Evaluation dataset
    print("Generating evaluation dataset")
    ann = base_ann_test
    ann = ann.drop(columns=["split"])
    sample = np.random.choice(ann["image_id"].values, int(0.5 * ann.shape[0]), replace=False)
    ann["split"] = ann.apply(lambda x: "val" if x.image_id in sample else "test", axis=1)
    ann.to_feather(out_dir_test / f"annotations.feather")

    metadata = {}
    matrix = np.zeros_like(matrix)
    for c in class_labels:
        matrix[class_labels.index(c), 0] = (
            ann["reg_to_attr"].apply(lambda x: [c] in x and [spurious_visual_feature] not in x).sum()
        )
        matrix[class_labels.index(c), 1] = (
            ann["reg_to_attr"].apply(lambda x: [c] in x and [spurious_visual_feature] in x).sum()
        )
    metadata["cramer_v_score"] = association(matrix, method="cramer")
    metadata["spurious_visual_feature"] = spurious_visual_feature
    metadata["class_labels"] = class_labels

    with open(out_dir_test / f"metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()