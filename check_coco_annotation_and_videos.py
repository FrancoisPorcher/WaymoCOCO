import pandas as pd
import os
import json


def main():
    path_waymococo_f0 = "/private/home/francoisporcher/data/waymococo_f0"
    path_train_2020 = os.path.join(path_waymococo_f0, "train2020")
    path_val_2020 = os.path.join(path_waymococo_f0, "val2020")
    
    path_annotations = os.path.join(path_waymococo_f0, "annotations")
    
    path_training_video_tensors = os.path.join(path_waymococo_f0, "training_video_tensors")
    path_validation_video_tensors = os.path.join(path_waymococo_f0, "validation_video_tensors")
    
    
    # print all the paths
    print(f"waymococo_f0 path: {path_waymococo_f0}")
    print(f"train_2020 path: {path_train_2020}")
    print(f"val_2020 path: {path_val_2020}")
    print(f"annotations path: {path_annotations}")
    print(f"training_video_tensors path: {path_training_video_tensors}")
    print(f"validation_video_tensors path: {path_validation_video_tensors}")
    
    # check if these two dirs exist
    if not os.path.exists(path_train_2020):
        raise ValueError(f"Path does not exist: {path_train_2020}")
    if not os.path.exists(path_val_2020):
        raise ValueError(f"Path does not exist: {path_val_2020}")
    if not os.path.exists(path_annotations):
        raise ValueError(f"Path does not exist: {path_annotations}")
    if not os.path.exists(path_training_video_tensors):
        raise ValueError(f"Path does not exist: {path_training_video_tensors}")
    if not os.path.exists(path_validation_video_tensors):
        raise ValueError(f"Path does not exist: {path_validation_video_tensors}")
    
    # Get the annotations.json files
    path_instances_val2020 = os.path.join(path_annotations, "instances_val2020.json")
    path_instances_train2020 = os.path.join(path_annotations, "instances_train2020.json")
    
    # Check if these files exist
    if not os.path.exists(path_instances_val2020):
        raise ValueError(f"Path does not exist: {path_instances_val2020}")
    if not os.path.exists(path_instances_train2020):
        raise ValueError(f"Path does not exist: {path_instances_train2020}")

    # Get the dataframes
    path_df_metadata_train = os.path.join(path_training_video_tensors, "df_metadata.csv")
    path_df_metadata_val = os.path.join(path_validation_video_tensors, "df_metadata.csv")
    
    if not os.path.exists(path_df_metadata_train):
        raise ValueError(f"Path does not exist: {path_df_metadata_train}")
    if not os.path.exists(path_df_metadata_val):
        raise ValueError(f"Path does not exist: {path_df_metadata_val}")
    
    # Read both dataframes
    df_metadata_train = pd.read_csv(path_df_metadata_train)
    df_metadata_val = pd.read_csv(path_df_metadata_val)
    
    
    # Load the josn files
    with open(path_instances_train2020, 'r') as f:
        instances_train2020 = json.load(f)
    with open(path_instances_val2020, 'r') as f:
        instances_val2020 = json.load(f)
    
    
    check_videos(path_training_video_tensors=path_training_video_tensors,
                 path_validation_video_tensors=path_validation_video_tensors,
                 df_metadata_train=df_metadata_train,
                 df_metadata_val=df_metadata_val)

    check_annotations(instances_train2020=instances_train2020,
                      instances_val2020=instances_val2020,
                      path_train_2020=path_train_2020,
                      path_val_2020=path_val_2020)


    check_videos_and_annotations(df_metadata_train=df_metadata_train,
                                 df_metadata_val=df_metadata_val)
    
    
def check_annotations(instances_train2020, instances_val2020, path_train_2020, path_val_2020): 
    ### COCO IMAGES
    num_training_frames = len([f for f in os.listdir(path_train_2020) if f.endswith('.jpg')])
    num_validation_frames = len([f for f in os.listdir(path_val_2020) if f.endswith('.jpg')])

    ### JSON ANNOTATIONS
    train_images = len(instances_train2020['images'])
    val_images = len(instances_val2020['images'])
    
    print()
    print(f"Number of training frames on disk: {num_training_frames}")
    print(f"Number of training images in annotations: {train_images}")
    print()
    print(f"Number of validation frames on disk: {num_validation_frames}")
    print(f"Number of validation images in annotations: {val_images}")

    
    if num_training_frames != train_images:
        raise ValueError("Mismatch in number of training frames between disk and annotations")
    if num_validation_frames != val_images:
        raise ValueError("Mismatch in number of validation frames between disk and annotations") 
    
    # check instaance_train2020['annotations'] and instance_val2020['annotations']
    # there is an attributed called category_id, go through all annotations and count how many of each category_id there are
    dict_train_category_counts = {}
    for annotation in instances_train2020['annotations']:
        category_id = annotation['category_id']
        if category_id not in dict_train_category_counts:
            dict_train_category_counts[category_id] = 0
        dict_train_category_counts[category_id] += 1
        
    # same for val
    dict_val_category_counts = {}
    for annotation in instances_val2020['annotations']:
        category_id = annotation['category_id']
        if category_id not in dict_val_category_counts:
            dict_val_category_counts[category_id] = 0
        dict_val_category_counts[category_id] += 1
        
    # Get the correspondance between category_id and category_name
    
    # print
    print()
    print(f"Training category counts: {dict_train_category_counts}")
    print(f"Validation category counts: {dict_val_category_counts}")
    print()

    # Go through all the iamges id in the images and check that they are unique
    train_image_ids = [image['id'] for image in instances_train2020['images']]
    val_image_ids = [image['id'] for image in instances_val2020['images']]
    if len(train_image_ids) != len(set(train_image_ids)):
        raise ValueError("Duplicate image ids found in training annotations")
    if len(val_image_ids) != len(set(val_image_ids)):
        raise ValueError("Duplicate image ids found in validation annotations")
    
    # check that all annotations refer to valid image ids
    valid_train_image_ids = set(train_image_ids)
    for annotation in instances_train2020['annotations']:
        if annotation['image_id'] not in valid_train_image_ids:
            raise ValueError(f"Annotation refers to invalid image id {annotation['image_id']} in training annotations")
    
    valid_val_image_ids = set(val_image_ids)
    for annotation in instances_val2020['annotations']:
        if annotation['image_id'] not in valid_val_image_ids:
            raise ValueError(f"Annotation refers to invalid image id {annotation['image_id']} in validation annotations")
        
    # Count how many annotations in total 
    total_train_annotations = len(instances_train2020['annotations'])
    total_val_annotations = len(instances_val2020['annotations'])
    print(f"Total number of training annotations: {total_train_annotations}")
    print(f"Total number of validation annotations: {total_val_annotations}")
    print("All annotations refer to valid image ids.")
    
    # Check if there are images without anny annotations
    annotated_train_image_ids = set([annotation['image_id'] for annotation in instances_train2020['annotations']])
    unannotated_train_image_ids = valid_train_image_ids - annotated_train_image_ids
    if len(unannotated_train_image_ids) > 0:
        print(f"Warning: There are {len(unannotated_train_image_ids)} training images without annotations.")   
    annotated_val_image_ids = set([annotation['image_id'] for annotation in instances_val2020['annotations']])
    unannotated_val_image_ids = valid_val_image_ids - annotated_val_image_ids
    if len(unannotated_val_image_ids) > 0:
        print(f"Warning: There are {len(unannotated_val_image_ids)} validation images without annotations.")
    
def check_videos_and_annotations(df_metadata_train, df_metadata_val):
    # print the column names of dataframes
    print("Training metadata columns:", df_metadata_train.columns)
    print("Validation metadata columns:", df_metadata_val.columns)
    
    # print the first row
    print("First row of training metadata:")
    print(df_metadata_train.iloc[0])


    
    print("test passed")

def check_videos(path_training_video_tensors, path_validation_video_tensors, df_metadata_train, df_metadata_val):
    # count the number of videos in training and validation sets
    list_of_training_videos = [f for f in os.listdir(path_training_video_tensors) if f.endswith('.pt')]
    list_of_validation_videos = [f for f in os.listdir(path_validation_video_tensors) if f.endswith('.pt')]

    nb_training_videos = len(list_of_training_videos)
    nb_validation_videos = len(list_of_validation_videos)
    
    nb_registered_training_videos = df_metadata_train.shape[0]
    nb_registered_validation_videos = df_metadata_val.shape[0]
    
    # print everything
    print(f"Number of training videos on disk: {nb_training_videos}")
    print(f"Number of training videos in metadata dataframe: {nb_registered_training_videos}")
    print()
    print(f"Number of validation videos on disk: {nb_validation_videos}")
    print(f"Number of validation videos in metadata dataframe: {nb_registered_validation_videos}")
    
    if nb_training_videos != nb_registered_training_videos:
        raise ValueError("Mismatch in number of training videos between disk and metadata dataframe")
    if nb_validation_videos != nb_registered_validation_videos:
        raise ValueError("Mismatch in number of validation videos between disk and metadata dataframe")
    
    
    

if __name__ == "__main__":
    main()