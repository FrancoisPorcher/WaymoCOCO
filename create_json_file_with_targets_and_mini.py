import json
import os
import pandas as pd

# Annotation dir
annotation_dir = "/private/home/francoisporcher/data/waymococo_f0/annotations"
annotation_val_file_name = "instances_val2020.json"
new_annotation_val_file_name = "instances_val2020_mini_target_only.json"
path_val_annotation_file = f"{annotation_dir}/{annotation_val_file_name}"
path_new_annotation_file = f"{annotation_dir}/{new_annotation_val_file_name}"

# dataframe file
path_video_tensors = "/private/home/francoisporcher/data/waymococo_f0/validation_video_tensors"
path_df_metadata = os.path.join(path_video_tensors, "df_metadata.csv")


def main():
    # Load data
    with open(path_val_annotation_file, "r") as f:
        val_data = json.load(f)
    
    df_metadata = pd.read_csv(path_df_metadata)
    
    # Get first occurrence for each (tfrecord_index, camera_name) combination
    df_subset = df_metadata.groupby(['tfrecord_index', 'camera_name'], as_index=False).first()
    
    print(f"Selected {len(df_subset)} subvideos across {df_subset['tfrecord_index'].nunique()} tfrecords "
          f"and {df_subset['camera_name'].nunique()} camera views")
    
    # Extract all image IDs from the subset into a list first
    image_ids_list = []
    for image_ids_str in df_subset['image_ids']:
        image_ids_list.extend(map(int, image_ids_str.strip("[]").split(",")))
    
    # Convert to set and verify uniqueness
    image_ids_to_keep = set(image_ids_list)
    
    # Check if all image IDs are unique
    assert len(image_ids_list) == len(image_ids_to_keep), \
        f"⚠️ Duplicate image IDs found! List: {len(image_ids_list)}, Set: {len(image_ids_to_keep)}"
    
    print(f"✓ All {len(image_ids_to_keep)} image IDs are unique")

    
    # Filter images (both by frame_type and image_id)
    filtered_images = [
        img for img in val_data['images'] 
        if img['frame_type'] == 'target' and img['id'] in image_ids_to_keep
    ]
    
    # There should be 202 videos, 5 camera views, and 8 frames each
    if not len(filtered_images) == 202 * 5 * 8:
        print(f"⚠️ Warning: Expected {202 * 5 * 8} images, but got {len(filtered_images)}")
    # Filter annotations based on the kept image IDs
    filtered_annotations = [
        ann for ann in val_data['annotations'] 
        if ann['image_id'] in image_ids_to_keep
    ]
    
    # Create new annotation data
    new_val_data = {
        **val_data,  # Keep other fields like 'categories', 'info', etc.
        'images': filtered_images,
        'annotations': filtered_annotations
    }
    
    print(f"Original: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"Filtered: {len(filtered_images)} images, {len(filtered_annotations)} annotations")
    
    # Remove existing file if present
    if os.path.exists(path_new_annotation_file):
        os.remove(path_new_annotation_file)
    
    # Write the new JSON file
    with open(path_new_annotation_file, "w") as f:
        json.dump(new_val_data, f, indent=2)
    
    print(f"Finished! New annotation file saved to: {path_new_annotation_file}")

if __name__ == "__main__":
    main()