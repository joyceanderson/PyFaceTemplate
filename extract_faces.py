#!/usr/bin/env python3
"""
Main entry point for PyFaceTemplate.
This script provides a simple interface to the face extraction functionality.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import face extractor
from src.face_extractor import process_images, argparse

if __name__ == "__main__":
    # Parse command line arguments
    description = """
    PyFaceTemplate: A standalone tool for facial feature extraction.
    
    This tool extracts facial features from images and saves them as templates.
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
    parser.add_argument('--detection-model', type=str, default='buffalo_l', 
                        help='model to use for face detection (e.g., scrfd_10g_gnkps or buffalo_l)')
    parser.add_argument('--recognition-model', type=str, default='buffalo_l', 
                        help='model to use for facial feature extraction (e.g., glintr100 or buffalo_l)')
    parser.add_argument('--models-dir', type=str, default='./models', 
                        help='directory containing model files (only needed for custom models)')
    parser.add_argument('--use-gpu', action='store_true', help="use GPU acceleration if available")
    parser.add_argument('--gpu-id', type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument('--num-gpus', type=int, default=1, 
                        help="Number of GPUs to use (for multi-GPU systems, default: 1)")
    
    
    args = parser.parse_args()
    
    # Process images
    process_images(args)
