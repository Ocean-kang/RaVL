import numpy as np
import pandas as pd
import torchvision
from tqdm.contrib.concurrent import process_map
import sparse
from rich import print

DIGIT_TO_TEXT = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    None: None,
}
TEXT_TO_DIGIT = {v: k for k, v in DIGIT_TO_TEXT.items()}

def load_mnist_data(data_dir: str):
    """
    Load the MNIST dataset and create train, val, and test splits.

    Parameters:
        data_dir (str): Directory for storing MNIST dataset
    Returns:
        data (dict): Contains MNIST image data associated with train, val, and test splits
        targets (dict): Contains MNIST digit labels associated with each image
    """
    data = {}
    targets = {}

    mnist_train = torchvision.datasets.MNIST(
        root=f"{data_dir}/mnist", train=True, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root=f"{data_dir}/mnist", train=False, download=True
    )

    data["train"] = mnist_train.data.unsqueeze(1).repeat(1, 3, 1, 1)
    targets["train"] = mnist_train.targets

    data["test"] = mnist_test.data.unsqueeze(1).repeat(1, 3, 1, 1)
    targets["test"] = mnist_test.targets

    return data, targets


def get_caption(digit_label: int = None):
    """
    Generate a caption given a digit label

    Parameters:
        digit_label (int): Label associated with the MNIST digit
    Returns:
        caption (list): A text caption describing the digit
    """

    def get_digit_caption(d):
        if d is None:
            return ""

        templates = [
            f"the image shows a {d}",
            f"the digit appears to be {d}",
            f"there is an image showing a {d}",
            f"the number is a {d}",
        ]
        return np.random.choice(templates)

    return get_digit_caption(digit_label)

def _save_image(id: int, image: np.array, out_dir: str):
    """
    Save pixel data associated with a single image to disk

    Parameters:
        id (int): Image index in dataset
        image (np.array): Pixel data associated with a MNIST image
        out_dir (str): Directory for storing images
    Returns:
        img_fp (str): Filepath for saved image
    """
    img_fp = out_dir / f"{id}.npz"
    sparse.save_npz(img_fp, sparse.COO(image))
    return str(img_fp)


def save_images(images: list, out_dir: str, num_processes: int = 10):
    """
    Save pixel data associated with a list of images to disk

    Parameters:
        images (list): List of pixel data associated with MNIST images
        out_dir (str): Directory for storing images
        num_processes (int): Number of parallel workers
    Returns:
        paths (list): Filepath for all saved images
    """
    print(f"Saving images to {out_dir}")
    paths = process_map(
        _save_image,
        np.arange(len(images)),
        images,
        [out_dir] * len(images),
        max_workers=num_processes,
        chunksize=1,
    )
    return paths




def save_ann(
    img_paths: list,
    out_dir: str,
    data: dict,
    split: str,
):
    """
    Save annotations associated with each MNIST image

    Parameters:
        img_paths (list): Contains filepaths for all saved images
        out_dir (str): Directory for storing annotations
        data (dict): Contains metadata associated with each image
        split (str): "train", "val", or "test"
    """
    keys = data.keys()

    df = {}
    num_data = len(data["img"])

    # Image properties
    df["image_id"] = np.arange(num_data)
    df["image_size"] = [[56, 56]] * (num_data)
    df["image_filepath"] = ['/'.join(x.split('/')[-3:]) for x in img_paths]

    # Splits and text
    df["split"] = [split] * num_data
    df["text"] = data["text"]

    # Region properties
    reg_to_attr = []
    true_label = []
    for a in (data["attributes"]):
        reg_to_attr.append([[x] for x in a])
        true_label.append([x for x in a if x in TEXT_TO_DIGIT][0])
    df["region_coord"] = data["boxes"]
    df["num_regions"] = [len(x) for x in (data["boxes"])]
    df["reg_to_attr"] = reg_to_attr
    df["true_label"] = true_label
    df = pd.DataFrame(df)

    # Save to disk
    print(f"Saving annotations to {out_dir}/annotations_{split}.feather")
    df.to_feather(out_dir / f"annotations_{split}.feather")
    return df
