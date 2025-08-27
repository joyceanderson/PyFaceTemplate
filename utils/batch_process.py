#!/usr/bin/env python3
"""
Utility for batch processing of facial templates.
This script can be used to compare a query face against a database of templates.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_template(template_path):
    """Load a facial template from a file."""
    try:
        return np.load(template_path)
    except Exception as e:
        print(f"Error loading template {template_path}: {str(e)}")
        return None


def find_matches(query_template_path, database_dir, threshold=0.5, top_n=5):
    """
    Find matches for a query template in a database of templates.
    
    Args:
        query_template_path: Path to the query template file
        database_dir: Directory containing database template files
        threshold: Similarity threshold for considering a match
        top_n: Number of top matches to return
        
    Returns:
        List of (template_path, similarity) tuples for top matches
    """
    # Load query template
    query_template = load_template(query_template_path)
    if query_template is None:
        return []
    
    # Find all template files in database directory
    database_files = list(Path(database_dir).glob("*.npy"))
    
    # Calculate similarities
    similarities = []
    for db_file in tqdm(database_files, desc="Comparing templates"):
        # Skip comparing with self if in database
        if os.path.abspath(db_file) == os.path.abspath(query_template_path):
            continue
            
        db_template = load_template(db_file)
        if db_template is not None:
            similarity = cosine_similarity(query_template, db_template)
            similarities.append((db_file, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by threshold and get top N
    matches = [(path, sim) for path, sim in similarities if sim >= threshold]
    return matches[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Batch process facial templates")
    parser.add_argument('query', type=str, help='Path to query template (.npy file)')
    parser.add_argument('database', type=str, help='Path to directory containing database templates')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for match (default: 0.5)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top matches to return (default: 5)')
    
    args = parser.parse_args()
    
    # Check if files/directories exist
    if not os.path.exists(args.query):
        print(f"Error: Query template file not found: {args.query}")
        return
    
    if not os.path.exists(args.database) or not os.path.isdir(args.database):
        print(f"Error: Database directory not found: {args.database}")
        return
    
    # Find matches
    matches = find_matches(args.query, args.database, args.threshold, args.top_n)
    
    # Print results
    print(f"\nFound {len(matches)} matches for {os.path.basename(args.query)}:")
    for i, (path, similarity) in enumerate(matches, 1):
        print(f"{i}. {path.name}: {similarity:.4f}")


if __name__ == "__main__":
    main()
