# PyFaceTemplate

A Python tool for ArcFace-based facial feature extraction and template generation with GPU support.

## Features

- Face detection using SCRFD models
- Facial feature extraction using ArcFace models
- Parallel processing of images
- Support for saving cropped face images
- Simple command-line interface

## Directory Structure

```
PyFaceTemplate/
├── models/            # Pre-trained models (optional for custom models)
├── sample/            # Sample images
│   ├── Hanks.jpg
│   ├── Pikachu.png
│   └── Stallone.jpg
├── src/               # Source code
│   └── face_extractor.py
├── utils/             # Utility scripts
│   └── visualize.py           # Visualization tool
├── extract_faces.py   # Main script
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Step-by-Step Guide

### 1. Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PyFaceTemplate.git
   cd PyFaceTemplate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Model Setup

The tool uses two types of models:
- **Detection model**: Identifies faces in images
- **Recognition model**: Extracts facial features (templates)

By default, the tool uses the built-in `buffalo_l` model from InsightFace, which will be automatically downloaded to your home directory (~/.insightface/models/) when first used.

**Optional**: If you want to use custom models:
1. Create a `models` directory if it doesn't exist
2. Download ONNX model files from trusted sources
3. Place them in the `models` directory with appropriate names (e.g., `scrfd_10g_gnkps.onnx` for detection)

### 3. Extract Face Templates

To process images and extract facial templates:

```bash
python extract_faces.py -d /path/to/images -o /path/to/output
```

**Examples:**

```bash
# Process sample images with default settings
python extract_faces.py

# Process your own images and save cropped faces
python extract_faces.py -d ./my_images -o ./my_results --save-crops

# Use multiple threads for faster processing
python extract_faces.py -d ./my_images --threads 8
```

### 4. Visualize Face Detection (Optional)

To see the detected faces with bounding boxes and landmarks:

```bash
python utils/visualize.py ./sample/Hanks.jpg --output ./output/visualization.jpg
```

### 5. Locate Your Results

After processing, check these directories:

- **Templates**: `./output/templates/*.npy` - The facial feature vectors
- **Cropped Faces**: `./output/crops/*.jpg` - 112x112 pixel aligned face images (if --save-crops was used)
- **Summary Files**: 
  - `./output/summary/templates.txt` - List of successfully processed images
  - `./output/summary/missing_templates.txt` - List of images where no face was detected

### 6. Command-line Options Reference

| Option | Description |
|--------|-------------|
| `-d, --dir` | Path to directory with images (default: ./sample) |
| `-o, --output` | Directory where templates are saved (default: ./output) |
| `-t, --threads` | Number of worker processes (default: 4) |
| `-b, --batch` | Number of images per batch (default: 1) |
| `-e, --extension` | Allowed image extensions (default: .jpeg, .jpg, .bmp, .png, .webp, .tiff) |
| `--exclude` | Exclude images with no face detected |
| `--save-crops` | Save cropped faces when detected |
| `--detection-model` | Model to use for face detection (default: buffalo_l) |
| `--recognition-model` | Model to use for facial feature extraction (default: buffalo_l) |
| `--models-dir` | Directory containing model files (only needed for custom models) |
| `--use-gpu` | Use GPU acceleration if available |
| `--gpu-id` | GPU ID to use (default: 0) |
| `--num-gpus` | Number of GPUs to use for multi-GPU systems (default: 1) |



## GPU Support

PyFaceTemplate supports GPU acceleration for faster processing, including multi-GPU parallel processing. To use GPU:

1. Install the GPU version of ONNX Runtime:
   ```bash
   pip install onnxruntime-gpu
   ```

2. Run the script with the `--use-gpu` flag:
   ```bash
   python extract_faces.py --use-gpu
   ```

3. For multi-GPU systems, you can specify which GPU to use:
   ```bash
   python extract_faces.py --use-gpu --gpu-id 1  # Use GPU #1 instead of #0
   ```

4. For maximum performance on systems with multiple GPUs, specify the number of GPUs to use:
   ```bash
   python extract_faces.py --use-gpu --num-gpus 2  # Use 2 GPUs in parallel
   ```
   
   When multiple GPUs are specified, the tool will:
   - Automatically divide the workload across the specified number of GPUs
   - Create separate processes for each GPU to enable true parallel processing
   - Distribute images evenly across all available GPUs
   - Combine results from all GPUs for the final output

## Examples

### Using Built-in Models (Recommended)

```bash
# Run with built-in buffalo_l model (default)
python extract_faces.py
```

### Using Custom Models

```bash
# Run with manually downloaded models
python extract_faces.py --detection-model scrfd_10g_gnkps --recognition-model glintr100 --models-dir ./models
```

### Using GPU Acceleration for Maximum Speed

```bash
# Process a large dataset with GPU acceleration
python extract_faces.py -d ./large_dataset -o ./results --use-gpu --threads 8

# Use a specific GPU on a multi-GPU system
python extract_faces.py --use-gpu --gpu-id 1

# Process a batch of images using multiple GPUs
python extract_faces.py -d ./large_dataset --use-gpu --num-gpus 2 --batch 16
```

