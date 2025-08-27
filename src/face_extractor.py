import argparse
import base64
import logging
import multiprocessing
import os
import time
from distutils import util
from functools import partial
from itertools import chain, islice
from pathlib import Path

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import uuid
from tqdm import tqdm
import warnings

# Suppress InsightFace warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

def list_files(path: str, allowed_ext: list) -> list:
    """List all files with allowed extensions in a directory."""
    return [
        str(os.path.join(dp, f)) 
        for dp, dn, filenames in os.walk(path) 
        for f in filenames 
        if os.path.splitext(f)[1].lower() in allowed_ext
    ]


def to_chunks(iterable, size=10):
    """Split an iterable into chunks of specified size."""
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def save_crop(img, name):
    """Save a cropped face image to the specified path."""
    cv2.imwrite(name, img)


def to_bool(input):
    """Convert string to boolean."""
    try:
        return bool(util.strtobool(input))
    except:
        return False


class FaceTemplateExtractor:
    """Class for extracting facial templates from images."""
    
    def __init__(self, detection_model='scrfd_10g_gnkps', recognition_model='glintr100', 
                 det_size=(640, 640), device='cpu', models_dir=None, use_gpu=False, gpu_id=0):
        """Initialize the face extractor with specified models."""
        try:
            # Configure providers based on device preference
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = gpu_id  # Use specified GPU ID
                logging.info(f"Using GPU acceleration (GPU ID: {gpu_id})")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = 0  # CPU context
                logging.info("Using CPU only mode")
                
            # Try to use local model if available
            if models_dir and os.path.exists(models_dir):
                logging.info(f"Using models from local directory: {models_dir}")
                # For detection model, use FaceAnalysis with local model directory
                self.app = FaceAnalysis(providers=providers)
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            else:
                # Fall back to default behavior (download from internet)
                logging.info(f"No local models found. Using default detection model: {detection_model}")
                self.app = FaceAnalysis(name=detection_model, providers=providers)
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                
            # For recognition model, try to load from local path first
            rec_model_path = None
            if models_dir:
                rec_model_path = os.path.join(models_dir, f"{recognition_model}.onnx")
                if os.path.exists(rec_model_path):
                    logging.info(f"Loading recognition model from: {rec_model_path}")
                else:
                    rec_model_path = None
                    
            # Special handling for buffalo_l model which is a built-in model
            if recognition_model == 'buffalo_l':
                logging.info(f"Using built-in buffalo_l model for recognition")
                # The buffalo_l model is already loaded by FaceAnalysis
                self.model = None
            elif rec_model_path:
                self.model = insightface.model_zoo.get_model(rec_model_path, 
                                                          providers=providers)
            else:
                logging.info(f"Using default recognition model: {recognition_model}")
                # Use default model path
                self.model = insightface.model_zoo.get_model(f'models/{recognition_model}.onnx', 
                                                          providers=providers)
            self.device = device
        except Exception as e:
            logging.error(f"Error initializing face extractor: {str(e)}")
            raise
        
    def extract(self, image_path, return_face_data=False):
        """Extract facial features from an image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to read image: {image_path}")
                return None, None
                
            faces = self.app.get(img)
            
            if len(faces) == 0:
                logging.warning(f"No face detected in {image_path}")
                return None, None
            
            # Get the face with the largest bounding box
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Extract facial features
            if self.model is None:
                # For buffalo_l, the embedding is already in the face object
                template = face.embedding
            else:
                # For other models, extract features using the model
                # Get aligned face for feature extraction
                bbox = face.bbox
                x1, y1, x2, y2 = map(int, bbox)
                # Add margin to the bounding box
                margin = 0.2
                h, w = y2 - y1, x2 - x1
                x1 = max(0, int(x1 - margin * w))
                y1 = max(0, int(y1 - margin * h))
                x2 = min(img.shape[1], int(x2 + margin * w))
                y2 = min(img.shape[0], int(y2 + margin * h))
                face_img = img[y1:y2, x1:x2]
                # Resize to model input size
                if hasattr(self.model, 'input_shape'):
                    input_size = self.model.input_shape[2:4]
                else:
                    # Default size for most face recognition models
                    input_size = (112, 112)
                face_img = cv2.resize(face_img, input_size)
                # Convert to RGB if needed
                if face_img.shape[2] == 1:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                elif face_img.shape[2] == 4:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
                else:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                # Get embedding
                template = self.model.get_feat(face_img).flatten()
            
            # Return face crop if requested
            if return_face_data:
                x1, y1, x2, y2 = map(int, face.bbox)
                face_img = img[y1:y2, x1:x2]
                return template, face_img
            
            return template, None
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None, None


def process_batch(batch_data):
    """Process a batch of images with a specific GPU."""
    image_files, args, gpu_id = batch_data
    
    # Initialize face extractor for this GPU
    try:
        extractor = FaceTemplateExtractor(
            detection_model=args.detection_model,
            recognition_model=args.recognition_model,
            device='cuda' if args.use_gpu else 'cpu',
            models_dir=args.models_dir,
            use_gpu=args.use_gpu,
            gpu_id=gpu_id
        )
    except Exception as e:
        logging.error(f"Failed to initialize face extractor on GPU {gpu_id}: {str(e)}")
        return [], []
    
    # Process each image in the batch
    successful = []
    missing = []
    
    for image_file in image_files:
        rel_path = os.path.relpath(image_file, args.dir)
        filename = os.path.splitext(os.path.basename(image_file))[0]
        
        # Extract template
        if args.save_crops:
            template, face_crop = extractor.extract(image_file, return_face_data=True)
        else:
            template, _ = extractor.extract(image_file)
            
        if template is not None:
            # Save template
            template_path = os.path.join(args.output, 'templates', f"{filename}.npy")
            np.save(template_path, template)
            
            # Save face crop if requested
            if args.save_crops and face_crop is not None:
                crop_path = os.path.join(args.output, 'crops', f"{filename}.jpg")
                cv2.imwrite(crop_path, face_crop)
            
            successful.append(rel_path)
        else:
            # Track images where no face was detected
            missing.append(rel_path)
            
            if not args.exclude:
                # If no face detected and not excluding, save empty template
                template_path = os.path.join(args.output, 'templates', f"{filename}.npy")
                np.save(template_path, np.zeros((512,), dtype=np.float32))
    
    return successful, missing


def process_images(args):
    """Process images and extract facial templates."""
    # Create output directories
    Path(args.output).mkdir(exist_ok=True)
    Path(os.path.join(args.output, 'templates')).mkdir(exist_ok=True)
    Path(os.path.join(args.output, 'summary')).mkdir(exist_ok=True)
    if args.save_crops:
        Path(os.path.join(args.output, 'crops')).mkdir(exist_ok=True)
        
    # Create summary files
    summary_file = os.path.join(args.output, 'summary', 'templates.txt')
    missing_file = os.path.join(args.output, 'summary', 'missing_templates.txt')
    summary_directory = Path(args.output).joinpath("summary")
    crops_directory = Path(args.output).joinpath("crops") if args.save_crops else None
    
    # Get list of image files
    image_files = list_files(args.dir, args.extension)
    logging.info(f"Total files detected: {len(image_files)}")
    
    # Start timing the processing
    start_time = time.time()
    
    # Determine if we're using multi-GPU processing
    if args.use_gpu and args.num_gpus > 1:
        logging.info(f"Using {args.num_gpus} GPUs for parallel processing")
        
        # Split images into batches for each GPU
        num_gpus = min(args.num_gpus, len(image_files))  # Don't use more GPUs than images
        batch_size = len(image_files) // num_gpus
        batches = []
        
        for i in range(num_gpus):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_gpus - 1 else len(image_files)
            gpu_batch = (image_files[start_idx:end_idx], args, i % args.num_gpus)
            batches.append(gpu_batch)
        
        # Process batches in parallel using multiple processes
        with multiprocessing.Pool(processes=num_gpus) as pool:
            results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing GPU batches"))
        
        # Combine results from all GPUs
        successful = []
        missing = []
        for s, m in results:
            successful.extend(s)
            missing.extend(m)
    else:
        # Single GPU or CPU processing
        gpu_id = args.gpu_id if args.use_gpu else 0
        logging.info(f"Using {'GPU ' + str(gpu_id) if args.use_gpu else 'CPU'} for processing")
        
        # Initialize face extractor
        try:
            extractor = FaceTemplateExtractor(
                detection_model=args.detection_model,
                recognition_model=args.recognition_model,
                device='cuda' if args.use_gpu else 'cpu',
                models_dir=args.models_dir,
                use_gpu=args.use_gpu,
                gpu_id=gpu_id
            )
        except Exception as e:
            logging.error(f"Failed to initialize face extractor: {str(e)}")
            logging.error("Please make sure you have the required models downloaded.")
            return
        
        # Track successful extractions and missing faces
        successful = []
        missing = []
        
        # Process images
        with tqdm(total=len(image_files)) as pbar:
            for image_file in image_files:
                rel_path = os.path.relpath(image_file, args.dir)
                filename = os.path.splitext(os.path.basename(image_file))[0]
                
                # Extract template
                if args.save_crops:
                    template, face_crop = extractor.extract(image_file, return_face_data=True)
                else:
                    template, _ = extractor.extract(image_file)
                    
                if template is not None:
                    # Save template
                    template_path = os.path.join(args.output, 'templates', f"{filename}.npy")
                    np.save(template_path, template)
                    
                    # Save face crop if requested
                    if args.save_crops and face_crop is not None:
                        crop_path = os.path.join(args.output, 'crops', f"{filename}.jpg")
                        cv2.imwrite(crop_path, face_crop)
                    
                    successful.append(rel_path)
                else:
                    # Track images where no face was detected
                    missing.append(rel_path)
                    
                    if not args.exclude:
                        # If no face detected and not excluding, save empty template
                        template_path = os.path.join(args.output, 'templates', f"{filename}.npy")
                        np.save(template_path, np.zeros((512,), dtype=np.float32))
                    
                pbar.update(1)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    processing_speed = len(image_files) / processing_time if processing_time > 0 else 0
    logging.info(f"Took: {processing_time:.3f} s. ({processing_speed:.3f} im/sec)")
    
    # Write summary files
    logging.info("Writing summary files...")
    
    # Write successful templates
    with open(summary_file, "w") as f:
        for template in successful:
            f.write(template + "\n")
    
    # Write missing templates if any
    if missing:
        with open(missing_file, "w") as f:
            for template in missing:
                f.write(template + "\n")
    
    logging.info("Done!")


if __name__ == "__main__":
    description = """
    PyFaceTemplate: A standalone tool for facial feature extraction.
    
    This tool extracts facial features from images and saves them as templates.
    
    By default, it will create the output directory if it doesn't exist.
    """
    
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--threads', default=4, type=int, help='number of worker processes')
    parser.add_argument('-b', '--batch', default=1, type=int, help='number of images per batch')
    parser.add_argument('-d', '--dir', type=str, help='path to directory with images', default='./sample')
    parser.add_argument('-o', '--output', type=str, help='directory where templates are saved', default="./output")
    parser.add_argument('--exclude', action='store_true', help="exclude images with no face detected")
    parser.add_argument('--save-crops', action='store_true', help="saves cropped faces when detected")
    parser.add_argument('-e', '--extension', nargs='+', help="allowed image extensions", 
                        default=['.jpeg', '.jpg', '.bmp', '.png', '.webp', '.tiff'])
    parser.add_argument('--detection-model', type=str, default='scrfd_10g_gnkps', 
                        help='model to use for face detection')
    parser.add_argument('--recognition-model', type=str, default='glintr100', 
                        help='model to use for facial feature extraction')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='directory containing local model files')
    parser.add_argument('--use-gpu', action='store_true', help="use GPU acceleration if available")
    parser.add_argument('--gpu-id', type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument('--num-gpus', type=int, default=1, 
                        help="Number of GPUs to use (for multi-GPU systems, default: 1)")
    
    
    args = parser.parse_args()
    process_images(args)
