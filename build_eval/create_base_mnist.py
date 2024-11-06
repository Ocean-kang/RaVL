import numpy as np
import torch
from PIL import Image
from pathlib import Path
import shutil
import pyrootutils
import os
from build_eval.utils import (
    load_mnist_data, 
    get_caption, 
    save_images, 
    save_ann, 
    DIGIT_TO_TEXT
)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

def construct_mnist_dataset(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_images: int,
):
    """
    Construct a base MNIST dataset in our desired format. Each image in the base dataset consists of 
    four quadrants, with one quadrant containing an MNIST digit. A separate quadrant may contain a red 
    rectangle with 50% probability. The remaining quadrants are empty.

    Parameters:
        data (torch.Tensor): Contains MNIST image data
        targets (torch.Tensor): Contains MNIST digit labels associated with each image
        num_images (int): Desired size of base MNIST dataset
    Returns: 
        dataset (dict): Contains MNIST images, text, region box coordinates, and region-level labels
    """
    dataset = {"img": [], "text": [], "boxes": [], "attributes": []}
    quadrant_coord = np.array([(0, 0), (28, 0), (0, 28), (28, 28)])
    total_img = 0

    while total_img < num_images:
        # Randomly sample digit from MNIST dataset
        idx = np.random.choice(data.shape[0])
        digit = {
            "region": torch.clone(data[idx]),
            "label": DIGIT_TO_TEXT[targets[idx].item()],
            "type": "digit",
        }

        # Generate red rectangle
        sample = np.random.random_sample()
        if sample <= 0.5:
            shape_reg = torch.zeros_like(digit["region"])
            shape_reg[0, :, :] = 255
            shape = {
                "region": shape_reg,
                "label": "rectangle",
                "type": "shape",
            }
        else:
            shape = None

        # Generate caption for region
        txt = get_caption(digit_label=digit["label"])
        dataset["text"].append(txt)

        # Create image with generated regions
        regions = [digit, shape] if shape else [digit]
        np.random.shuffle(regions)
        selected_idx = quadrant_coord[
            sorted(
                np.random.choice(
                    range(len(quadrant_coord)), size=len(regions), replace=False
                )
            )
        ].tolist()

        new_im = Image.new("RGB", (56, 56))
        attributes = []
        for q in quadrant_coord.tolist():
            if q in selected_idx: 
                region = regions[selected_idx.index(q)]
                new_im.paste(
                    Image.fromarray(
                        np.transpose(region["region"].numpy().astype("uint8"), (1, 2, 0)),
                        "RGB",
                    ),
                    tuple(q),
                )
                if region["type"] == "digit" or region["type"] == "shape":
                    attributes.append(region["label"])
            else: 
                attributes.append("empty")

        new_im = np.array(new_im).transpose(2, 0, 1)
        dataset["img"].append(new_im)
        dataset["attributes"].append(attributes)
        dataset["boxes"].append([[q[0], q[1], q[0] + 28, q[1] + 28] for q in quadrant_coord])

        total_img += 1

    return dataset


def main():
    print("=> Generating base MNIST data")
    data_dir = os.path.join(root, "data")

    np.random.seed(0)
    data, targets = load_mnist_data(data_dir)

    # Construct base training dataset
    train_dataset = construct_mnist_dataset(
        data=data["train"],
        targets=targets["train"],
        num_images=40000,
    )

    # Save images and annotations to disk
    out_dir = Path(data_dir) / f"mnist_base"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    (out_dir / "images_train").mkdir()
    img_paths = save_images(
        images=train_dataset["img"],
        out_dir=out_dir / "images_train",
    )

    df = save_ann(img_paths=img_paths, out_dir=out_dir, data=train_dataset, split="train")

    # Construct base test dataset
    test_dataset = construct_mnist_dataset(
        data=data["test"],
        targets=targets["test"],
        num_images=2000,
    )

    # Save images and annotations to disk
    (out_dir / "images_test").mkdir()
    img_paths = save_images(
        images=test_dataset["img"],
        out_dir=out_dir / "images_test",
    )
    df = save_ann(img_paths=img_paths, out_dir=out_dir, data=test_dataset, split="test")


if __name__ == "__main__":
    main()