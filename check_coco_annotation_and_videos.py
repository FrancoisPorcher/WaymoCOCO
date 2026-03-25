import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default="/checkpoint/unicorns/shared/datasets/waymococo_f0",
        type=str,
        help="Base path containing images, annotations, and video tensor folders.",
    )

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--validation",
        action="store_true",
        help="Run checks only for the validation split.",
    )
    split_group.add_argument(
        "--train",
        action="store_true",
        help="Run checks only for the training split.",
    )
    split_group.add_argument(
        "--all",
        action="store_true",
        help="Run checks for both training and validation splits.",
    )
    return parser.parse_args()


def selected_splits(args):
    if args.validation:
        return ["val"]
    if args.train:
        return ["train"]
    return ["train", "val"]


def build_paths(base_path):
    return {
        "annotations": os.path.join(base_path, "annotations"),
        "train_images": os.path.join(base_path, "train2020"),
        "val_images": os.path.join(base_path, "val2020"),
        "train_tensors": os.path.join(base_path, "training_video_tensors"),
        "val_tensors": os.path.join(base_path, "validation_video_tensors"),
    }


def require_path(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_annotations(path):
    with open(path, "r") as f:
        return json.load(f)


def load_metadata(path):
    return pd.read_csv(path)


def count_jpgs(path):
    return len([name for name in os.listdir(path) if name.endswith(".jpg")])


def count_pts(path):
    return len([name for name in os.listdir(path) if name.endswith(".pt")])


def check_annotations(split_name, instances, image_dir):
    num_frames_on_disk = count_jpgs(image_dir)
    num_images_in_annotations = len(instances["images"])

    print()
    print(f"[{split_name}] Number of frames on disk: {num_frames_on_disk}")
    print(f"[{split_name}] Number of images in annotations: {num_images_in_annotations}")

    if num_frames_on_disk != num_images_in_annotations:
        raise ValueError(
            f"[{split_name}] Mismatch in number of frames between disk and annotations"
        )

    category_counts = {}
    for annotation in instances["annotations"]:
        category_id = annotation["category_id"]
        category_counts[category_id] = category_counts.get(category_id, 0) + 1

    print(f"[{split_name}] Category counts: {category_counts}")

    image_ids = [image["id"] for image in instances["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"[{split_name}] Duplicate image ids found in annotations")

    valid_image_ids = set(image_ids)
    for annotation in instances["annotations"]:
        if annotation["image_id"] not in valid_image_ids:
            raise ValueError(
                f"[{split_name}] Annotation refers to invalid image id {annotation['image_id']}"
            )

    total_annotations = len(instances["annotations"])
    print(f"[{split_name}] Total number of annotations: {total_annotations}")
    print(f"[{split_name}] All annotations refer to valid image ids.")

    annotated_image_ids = {annotation["image_id"] for annotation in instances["annotations"]}
    unannotated_image_ids = valid_image_ids - annotated_image_ids
    if unannotated_image_ids:
        print(
            f"[{split_name}] Warning: There are {len(unannotated_image_ids)} images without annotations."
        )


def check_videos(split_name, tensor_dir, df_metadata):
    num_videos_on_disk = count_pts(tensor_dir)
    num_videos_in_metadata = df_metadata.shape[0]

    print(f"[{split_name}] Number of videos on disk: {num_videos_on_disk}")
    print(f"[{split_name}] Number of videos in metadata dataframe: {num_videos_in_metadata}")

    if num_videos_on_disk != num_videos_in_metadata:
        raise ValueError(
            f"[{split_name}] Mismatch in number of videos between disk and metadata dataframe"
        )


def check_videos_and_annotations(split_name, df_metadata):
    print(f"[{split_name}] Metadata columns: {list(df_metadata.columns)}")
    print(f"[{split_name}] First row of metadata:")
    print(df_metadata.iloc[0])
    print(f"[{split_name}] Checks passed.")


def run_split_checks(split_name, image_dir, tensor_dir, annotation_path):
    require_path(image_dir, f"{split_name} image_dir")
    require_path(tensor_dir, f"{split_name} tensor_dir")
    require_path(annotation_path, f"{split_name} annotation file")

    metadata_path = os.path.join(tensor_dir, "df_metadata.csv")
    require_path(metadata_path, f"{split_name} metadata file")

    instances = load_annotations(annotation_path)
    df_metadata = load_metadata(metadata_path)

    check_videos(split_name=split_name, tensor_dir=tensor_dir, df_metadata=df_metadata)
    check_annotations(split_name=split_name, instances=instances, image_dir=image_dir)
    check_videos_and_annotations(split_name=split_name, df_metadata=df_metadata)


def main():
    args = parse_args()
    splits = selected_splits(args)
    paths = build_paths(args.base_path)

    print(f"waymococo_f0 path: {args.base_path}")
    print(f"requested splits: {splits}")
    print(f"annotations path: {paths['annotations']}")
    require_path(paths["annotations"], "annotations dir")

    split_config = {
        "train": {
            "image_dir": paths["train_images"],
            "tensor_dir": paths["train_tensors"],
            "annotation_path": os.path.join(paths["annotations"], "instances_train2020.json"),
        },
        "val": {
            "image_dir": paths["val_images"],
            "tensor_dir": paths["val_tensors"],
            "annotation_path": os.path.join(paths["annotations"], "instances_val2020.json"),
        },
    }

    for split_name in splits:
        config = split_config[split_name]
        print()
        print(f"Running checks for split: {split_name}")
        run_split_checks(
            split_name=split_name,
            image_dir=config["image_dir"],
            tensor_dir=config["tensor_dir"],
            annotation_path=config["annotation_path"],
        )


if __name__ == "__main__":
    main()
