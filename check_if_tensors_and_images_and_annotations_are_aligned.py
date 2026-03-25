import json
import os
import pandas as pd
import torch
import torchvision

# I also need to make imports to visualize the bounding boxes
from torchvision.utils import draw_bounding_boxes


def load_metadata(metadata_dir):
    path = os.path.join(metadata_dir, "df_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_annotations(annotation_dir, split):
    filename = f"instances_{split}2020.json"
    path = os.path.join(annotation_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


def main():
    base_path = "/private/home/francoisporcher/data/waymococo_f0"
    paths = {
        "train_images": os.path.join(base_path, "train2020"),
        "val_images": os.path.join(base_path, "val2020"),
        "annotations": os.path.join(base_path, "annotations"),
        "train_tensors": os.path.join(base_path, "training_video_tensors"),
        "val_tensors": os.path.join(base_path, "validation_video_tensors"),
    }

    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # df_train = load_metadata(paths["train_tensors"])
    df_val = load_metadata(paths["val_tensors"])
    # annotations_train = load_annotations(paths["annotations"], "train")
    annotations_val = load_annotations(paths["annotations"], "val")
    
    # build fast lookup for annotations by video filename
    anns_by_img = {}
    for ann in annotations_val['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    cat_id_to_name = {cat['id']: cat['name'] for cat in annotations_val['categories']}

    # Now pick a random row from the validation dataframe
    sample_row = df_val.sample(n=1).iloc[0]
    video_file_full_path = os.path.join(paths["val_tensors"], sample_row['video_filename'])
    # load the tensor
    video_tensor = torch.load(video_file_full_path)  # [T, 3, H, W]
    
    # Also load all the images in annotations with the corresponding video_id
    image_ids = sample_row['image_ids'] # this is a string like "[12345, 12346, ...]"
    image_ids = list(map(int, image_ids.strip("[]").split(","))) # list of int [12345, 12346, ...]
    

    image_lookup = {img['id']: img for img in annotations_val['images']}
    image_file_paths = [os.path.join(paths["val_images"], image_lookup[i]['file_name']) for i in image_ids]
    assert len(image_file_paths) == 16, f"Expected 16 image file paths, but got {len(image_file_paths)}"

    jpg_imgs = [torchvision.io.read_image(img_path) for img_path in image_file_paths]  # each is [3, H, W]
    jpg_video = torch.stack(jpg_imgs).permute(0, 2, 3, 1).contiguous()
    tensor_video = video_tensor.permute(0, 2, 3, 1).contiguous()
    
    # Now create the imgs with annotations
    class_colors = {
        1: "red",       # TYPE_VEHICLE
        2: "green",     # TYPE_PEDESTRIAN
        3: "blue",      # TYPE_CYCLIST
    }

    annotated_imgs = []
    tensor_annotated_imgs = []

    # video_tensor: [T, 3, H_res, W_res]
    T, _, H_res, W_res = video_tensor.shape

    for img, img_id, tensor_frame in zip(jpg_imgs, image_ids, video_tensor):
        annotations = anns_by_img.get(img_id, [])
        if len(annotations) == 0:
            annotated_imgs.append(img)
            tensor_annotated_imgs.append(tensor_frame)
            continue

        # build boxes and labels COCO bbox [x, y, w, h] to [x1, y1, x2, y2] in ORIGINAL resolution
        bboxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)  # [N, 4]
        bboxes[:, 2] += bboxes[:, 0]  # x2 = x + w
        bboxes[:, 3] += bboxes[:, 1]  # y2 = y + h

        labels = [cat_id_to_name[ann['category_id']] for ann in annotations]
        colors = [class_colors[ann['category_id']] for ann in annotations]

        # Draw on original JPG image
        img_with_boxes = draw_bounding_boxes(
            img,
            boxes=bboxes,
            labels=labels,
            colors=colors,
            width=5,
            font_size=30,
        )
        annotated_imgs.append(img_with_boxes)

        # ---- Now: draw on RESIZED tensor frame (512x512) ----
        # img is [3, H_orig, W_orig], tensor_frame is [3, H_res, W_res]
        _, H_orig, W_orig = img.shape
        sx = W_res / W_orig
        sy = H_res / H_orig

        bboxes_resized = bboxes.clone()
        bboxes_resized[:, [0, 2]] *= sx  # scale x1, x2
        bboxes_resized[:, [1, 3]] *= sy  # scale y1, y2

        # ensure uint8 for draw_bounding_boxes
        tf = tensor_frame
        if tf.dtype != torch.uint8:
            tf = (tf.clamp(0, 1) * 255).to(torch.uint8)

        tensor_frame_with_boxes = draw_bounding_boxes(
            tf,
            boxes=bboxes_resized,
            labels=labels,
            colors=colors,
            width=4,
            font_size=25,
        )
        tensor_annotated_imgs.append(tensor_frame_with_boxes)

    annotated_video = torch.stack(annotated_imgs).permute(0, 2, 3, 1).contiguous()
    tensor_annotated_video = torch.stack(tensor_annotated_imgs).permute(0, 2, 3, 1).contiguous()

    output_dir = os.path.join(os.path.dirname(__file__), "video_checks")
    # If it exists remove it
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    video_stem = os.path.splitext(sample_row['video_filename'])[0]

    tensor_path = os.path.join(output_dir, f"{video_stem}_tensor.mp4")
    jpg_path = os.path.join(output_dir, f"{video_stem}_images.mp4")
    annotated_path = os.path.join(output_dir, f"{video_stem}_annotated.mp4")
    annotated_tensor_path = os.path.join(output_dir, f"{video_stem}_annotated_tensor.mp4")

    fps = 10
    torchvision.io.write_video(tensor_path, tensor_video, fps=fps)
    torchvision.io.write_video(jpg_path, jpg_video, fps=fps)
    torchvision.io.write_video(annotated_path, annotated_video, fps=fps)
    torchvision.io.write_video(annotated_tensor_path, tensor_annotated_video, fps=fps)

    print(f"Wrote {tensor_path} and {jpg_path} and {annotated_path} and {annotated_tensor_path}")


if __name__ == "__main__":
    main()
