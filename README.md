# RISE-MICCAI LISA 2025: MRI Quality Control

Challenge: Low-field pediatric brain magnetic resonance Image Segmentation and quality Assurance Challenge

Multi-output multiclass classification deep learning pipeline for artifact detection in 3D brain MRI scans.

![Pipeline Architecture](images/pipeline.png)

## Quick Start

### Installation
```bash
git clone https://github.com/CristianLazoQuispe/lisa-challenge2025-task1
cd lisa-challenge2025-task1
pip install -r requirements.txt
```

### Training
```bash
python train.py \
  --train_csv ./data/train.csv \
  --save_dir ./models/exp1 \
  --n_splits 5 \
  --epochs 30 \
  --batch_size 32 \
  --base_model maxvit_nano_rw_256 \
  --image_size 256 \
  --device cuda:0
```

### Inference
```bash
# 1. Create metadata CSV
python csv_creation.py --val_path_dir /input --path_results ./results/

# 2. Convert 3D to 2D slices
python pipeline3dto2d.py \
  --results_dir ./results/preprocessed_data \
  --destination_dir ./results/2d_images_all

# 3. Run model inference
python train.py \
  --test_csv ./results/preprocessed_data/df_test_imgs.csv \
  --model_dir ./models/exp1 \
  --do_inference

# 4. Generate final submission file
python submission.py \
  --input ./results/preprocessed_data/submission_mean_preds.csv \
  --output /output
```

## Docker Usage

### Build & Run
```bash
# Build
docker build -t lisa_task1:latest .

# Development
docker compose run --rm dev bash

# Production
docker compose up run
```

### Synapse Submission
```bash
# 1. Tag
docker tag lisa_task1:latest docker.synapse.org/SYN_ID/task1:v1.0

# 2. Login
docker login docker.synapse.org

# 3. Push
docker push docker.synapse.org/SYN_ID/task1:v1.0

# 4. Test locally
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output:rw \
  docker.synapse.org/SYN_ID/task1:v1.0
```

## Key Features
- **7-class artifact detection**: Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion
- **Vision Transformer backbone**: MaxViT with label token attention heads
- **Brain-aware preprocessing**: Morphological operations + connected components
- **5-fold cross-validation**: Patient-level stratified splitting
- **Multi-aggregation strategies**: Mean, vote, max pooling across slices
- **Focal loss + GIoU**: Combined classification and spatial localization

## Requirements
- GPU: NVIDIA with 4GB+ VRAM
- CUDA: 11.7+
- Python: 3.8+
- Docker: 20.10+

## License

MIT



