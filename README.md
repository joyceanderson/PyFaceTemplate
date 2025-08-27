# PyFaceTemplate

**PyFaceTemplate** is a Python tool for extracting face embeddings (templates) using [InsightFace](https://github.com/deepinsight/insightface) models.  
It supports face detection, face alignment, and embedding generation, with optional GPU acceleration for faster processing.

---

## Features

- Face detection using **SCRFD** models  
- Face alignment with keypoints or bounding-box fallback  
- Facial feature extraction using **ArcFace/ONNX** models  
- Built-in support for the **buffalo_l** model (auto-downloaded if not provided)  
- Option to use custom ONNX models from a local `models/` folder  
- Save aligned cropped face images (default **112×112**, configurable) alongside embeddings  
- Multi-threaded and multi-GPU support for large datasets  
- Command-line interface with configurable options  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PyFaceTemplate.git
   cd PyFaceTemplate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Model Setup

PyFaceTemplate requires two types of models:
- **Detection model**: Finds faces in the image (e.g., scrfd_10g_gnkps.onnx)
- **Recognition model**: Extracts embeddings (e.g., glintr100.onnx or the built-in buffalo_l)

**Using built-in models**
If you don’t provide models, InsightFace will automatically download buffalo_l into:
   ```bash
   ~/.insightface/models/
   ```

**Using custom models**: 
If you want to use your own models:
1. Create a models directory in the repo root
2. Place .onnx files into the models/ directory, for example:
   ```bash
   PyFaceTemplate/models/
   ├── scrfd_10g_gnkps.onnx
   └── glintr100.onnx
    ```
3. Run with the --models-dir ./models option

### Extract Face Templates

**Basic usage::**
   ```bash
   # Process sample images with default settings (uses buffalo_l)
   python extract_faces.py

   # Process your own images and save cropped faces
   python extract_faces.py -d ./my_images -o ./my_results --save-crops

   # Use multiple threads for faster processing
   python extract_faces.py -d ./my_images --threads 8

   # Use custom models
   python extract_faces.py --detection-model scrfd_10g_gnkps --recognition-model glintr100 --models-dir ./models
   ```

### Visualize Face Detection (Optional)

To see detected faces with bounding boxes and landmarks:

   ```bash
   python utils/visualize.py ./sample/Hanks.jpg --output ./output/visualization.jpg
   ```

### Output Files

After processing, results are stored under the specified --output directory:

- **Templates**: `./output/templates/*.npy` – Facial embeddings (numpy vectors)
- **Cropped Faces**: `./output/crops/*.jpg` – Aligned face crops (default 112×112 RGB, if --save-crops is used)
- **Summary Files**: 
  - `./output/summary/templates.txt` – List of successfully processed images
  - `./output/summary/missing_templates.txt` - List of images where no face was detected

### Command-line Options Reference

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

