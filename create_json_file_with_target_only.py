import json
import os

annotation_dir = "/private/home/francoisporcher/data/waymococo_f0/annotations"
annotation_val_file_name = "instances_val2020.json"
new_annotation_val_file_name = "instances_val2020_target_only.json"
path_val_annotation_file = f"{annotation_dir}/{annotation_val_file_name}"
path_new_annotation_file = f"{annotation_dir}/{new_annotation_val_file_name}"

def main():
    with open(path_val_annotation_file, "r") as f:
        val_data = json.load(f)
    
    val_data_images = val_data['images'] 
    # filter only val_data_images if 'frame_type' is 'target'
    val_data_images_target_only = [img for img in val_data_images if img['frame_type'] == 'target']
    
    # compare the two set of images ids
    val_data_images_ids = set(img['id'] for img in val_data_images)
    val_data_images_target_only_ids = set(img['id'] for img in val_data_images_target_only)
    print(f"Number of images in original val data: {len(val_data_images)}")
    print(f"Number of images in target only val data: {len(val_data_images_target_only)}")
    
    # Filter annotations to keep only those with image_id in target-only set
    val_data_annotations = val_data['annotations']
    val_data_annotations_target_only = [
        ann for ann in val_data_annotations 
        if ann['image_id'] in val_data_images_target_only_ids
    ]
    
    print(f"Number of annotations in original val data: {len(val_data_annotations)}")
    print(f"Number of annotations in target only val data: {len(val_data_annotations_target_only)}")
    
    # Create new data dictionary with filtered images and annotations
    new_val_data = val_data.copy()
    new_val_data['images'] = val_data_images_target_only
    new_val_data['annotations'] = val_data_annotations_target_only
    
    # check if path_new_annotation_file already exists
    if os.path.exists(path_new_annotation_file):
        os.remove(path_new_annotation_file)
    
    # Write the new JSON file
    with open(path_new_annotation_file, "w") as f:
        json.dump(new_val_data, f, indent=2)
    
    print(f"Finished! New annotation file saved to: {path_new_annotation_file}")

if __name__ == "__main__":
    main()