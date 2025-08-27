#!/usr/bin/env python3
"""
Utility for visualizing face detection results.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import face extractor components
from face_extractor import FaceTemplateExtractor


def draw_face_boxes(image_path, output_path=None, detection_model='buffalo_l', models_dir='./models'):
    """
    Draw bounding boxes around detected faces in an image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, display instead)
        detection_model: Face detection model to use
        models_dir: Directory containing model files
    """
    # Initialize face analysis app directly
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # Detect faces
    faces = app.get(img)
    
    # Draw bounding boxes
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Draw landmarks if available
        if hasattr(face, 'landmark_2d_106'):
            landmarks = face.landmark_2d_106.astype(np.int32)
            for point in landmarks:
                cv2.circle(img, tuple(point), 1, (0, 0, 255), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")
        return True
    else:
        cv2.imshow("Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize face detection results")
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save output image (if not provided, display instead)')
    parser.add_argument('--detection-model', type=str, default='buffalo_l', 
                        help='Face detection model to use')
    parser.add_argument('--models-dir', type=str, default='./models', 
                        help='Directory containing model files')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        return
    
    # Draw face boxes
    draw_face_boxes(args.image, args.output, args.detection_model, args.models_dir)


if __name__ == "__main__":
    main()
