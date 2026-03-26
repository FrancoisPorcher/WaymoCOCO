# WaymoCOCO

## Download instructions

We are using the [Waymo Open Dataset v1.4.3](https://waymo.com/open/download/).

### 1. Install gcloud SDK

`gsutil` is bundled with the gcloud SDK. Install it locally (no sudo needed):

```bash
URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz"
curl -L -O "$URL"
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet
```


### 2. Authenticate

```bash
gcloud auth login
```

### 3. Create destination directories

Replace `YOUR_DATASET_PATH` with your storage path (e.g. a shared cluster directory):

```bash
mkdir -p YOUR_DATASET_PATH/waymo_v1_4_3/training
mkdir -p YOUR_DATASET_PATH/waymo_v1_4_3/validation
```

### 4. Download

Run both in parallel (separate terminals) to maximize throughput:

```bash
# Terminal 1 — training (~800 GB)
DEST="YOUR_DATASET_PATH/waymo_v1_4_3/training/"
gsutil -m cp "gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord" "$DEST"

# Terminal 2 — validation (~150 GB)
DEST="YOUR_DATASET_PATH/waymo_v1_4_3/validation/"
gsutil -m cp "gs://waymo_open_dataset_v_1_4_3/individual_files/validation/*.tfrecord" "$DEST"
```
# Installation (updated with uv)

```bash
uv sync
```

## Path Setup

Machine-specific dataset paths are configured through a local `.env` file instead of being hardcoded in the scripts.

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
WAYMOCOCO_BASE_PATH="/path/to/waymococo_f0"
WAYMO_TRAIN_TFRECORD_DIR="/path/to/waymo_v1_4_3/training"
WAYMO_VAL_TFRECORD_DIR="/path/to/waymo_v1_4_3/validation"
WAYMO_TEST_TFRECORD_DIR="/path/to/waymo_v1_4_3/testing"
```

The Python validation scripts read `WAYMOCOCO_BASE_PATH` from `.env` by default. The SLURM export scripts read `WAYMOCOCO_BASE_PATH` plus the split-specific TFRecord directory variables.

# Conversion

We first convert the raw Waymo dataset in a frieldy format for training video models.

### WaymoCOCO f0 (frame 0)

``` bash
# convert val
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/validation/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname val2020 \
    --image_filename_prefix val \
    --label_filename instances_val2020.json \
    --add_waymo_info
# convert train
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/training/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname train2020 \
    --image_filename_prefix train \
    --label_filename instances_train2020.json \
    --add_waymo_info
# convert test
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/testing/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname test2020 \
    --image_filename_prefix test \
    --label_filename image_info_test2020.json \
    --add_waymo_info
```

### WaymoCOCO full

Full conversion is also available. Please note that a machine with 208-416 GB of CPU memory is needed for full training in the case of MMDetection v2.0.

``` bash
# convert val
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/validation/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname val2020 \
    --image_filename_prefix val \
    --label_filename instances_val2020.json \
    --add_waymo_info
# convert train
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/training/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname train2020 \
    --image_filename_prefix train \
    --label_filename instances_train2020.json \
    --add_waymo_info
# convert test
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/testing/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname test2020 \
    --image_filename_prefix test \
    --label_filename image_info_test2020.json \
    --add_waymo_info
```

### Other options

Please see [convert_waymo_to_coco.py](convert_waymo_to_coco.py).

### Local run commands

The debug launchers in `.vscode/launch.json` should use your local paths from `.env`. Equivalent commands are:

``` bash
# training (first 150 sequences)
python convert_waymo_to_coco.py \
    --tfrecord_dir "${WAYMO_TRAIN_TFRECORD_DIR}" \
    --work_dir "${WAYMOCOCO_BASE_PATH}" \
    --image_dirname train2020 \
    --image_filename_prefix train \
    --label_filename instances_train2020.json \
    --add_waymo_info

# evaluation
python convert_waymo_to_coco.py \
    --tfrecord_dir "${WAYMO_VAL_TFRECORD_DIR}" \
    --work_dir "${WAYMOCOCO_BASE_PATH}" \
    --image_dirname val2020 \
    --image_filename_prefix val \
    --label_filename instances_val2020.json \
    --add_waymo_info

# test
python convert_waymo_to_coco.py \
    --tfrecord_dir "${WAYMO_TEST_TFRECORD_DIR}" \
    --work_dir "${WAYMOCOCO_BASE_PATH}" \
    --image_dirname test2020 \
    --image_filename_prefix test \
    --label_filename image_info_test2020.json \
    --add_waymo_info
```

### Validation checks

After `.env` is configured, you can run:

```bash
uv run python check_coco_annotation_and_videos.py --all
uv run python check_if_tensors_and_images_and_annotations_are_aligned.py --split val
```


## Creating symlinks (optional)

If you prepared WaymoCOCO f0 and WaymoCOCO full, the directory structure is as follows.

```
${HOME}/data
├── waymococo_f0
│   ├── annotations
│   ├── test2020
│   ├── train2020
│   └── val2020
└── waymococo_full
    ├── annotations
    ├── test2020
    ├── train2020
    └── val2020
```

Even if you use full training set for training, f0val is sufficient for validation.
It is useful to create symlinks for that.

``` bash
mkdir -p ${HOME}/data/waymococo/annotations
ln -s ${HOME}/data/waymococo_full/annotations/image_info_test2020.json ${HOME}/data/waymococo/annotations/image_info_test2020.json
ln -s ${HOME}/data/waymococo_full/annotations/instances_train2020.json ${HOME}/data/waymococo/annotations/instances_train2020.json
ln -s ${HOME}/data/waymococo_f0/annotations/instances_val2020.json ${HOME}/data/waymococo/annotations/instances_val2020.json
ln -s ${HOME}/data/waymococo_full/test2020 ${HOME}/data/waymococo/test2020
ln -s ${HOME}/data/waymococo_full/train2020 ${HOME}/data/waymococo/train2020
ln -s ${HOME}/data/waymococo_f0/val2020 ${HOME}/data/waymococo/val2020
```

<!--
```
${HOME}/data
└── waymococo
    ├── annotations
    │   ├── image_info_test2020.json -> ${HOME}/data/waymococo_full/annotations/image_info_test2020.json
    │   ├── instances_train2020.json -> ${HOME}/data/waymococo_full/annotations/instances_train2020.json
    │   └── instances_val2020.json -> ${HOME}/data/waymococo_f0/annotations/instances_val2020.json
    ├── test2020 -> ${HOME}/data/waymococo_full/test2020
    ├── train2020 -> ${HOME}/data/waymococo_full/train2020
    └── val2020 -> ${HOME}/data/waymococo_f0/val2020
```
-->

If you use mmdetection, it is recommended to create symlinks in your mmdetection directory.  

``` bash
ln -s ${HOME}/data/waymococo_f0 ${MMDET_DIR}/data/waymococo_f0
ln -s ${HOME}/data/waymococo_full ${MMDET_DIR}/data/waymococo_full
ln -s ${HOME}/data/waymococo ${MMDET_DIR}/data/waymococo
# or simply
ln -s ${HOME}/data ${MMDET_DIR}/data
```


## Acknowledgements

The files in waymo_open_dataset directory are borrowed from [the official code](https://github.com/waymo-research/waymo-open-dataset/) to mitigate dependency.
The official code and [Waymo-Dataset-Tool](https://github.com/RalphMao/Waymo-Dataset-Tool) (converter for KITTI format) were referred to write this converter.
