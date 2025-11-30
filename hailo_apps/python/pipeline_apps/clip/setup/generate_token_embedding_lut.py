#!/usr/bin/env python3
"""
Generate Token Embedding LUT for CLIP text encoder.

This script extracts the token embedding layer (Look-Up Table) from CLIP models
and saves it as a numpy array for use with the Hailo text encoder.

The tokenizers package (already installed) handles text→token_ids.
This script extracts the token embedding LUT: token_ids→embeddings.

Requirements:
    pip install transformers torch

Usage:
    # For CLIP ViT-B/32 (default)
    python generate_token_embedding_lut.py
    
    # For specific model
    python generate_token_embedding_lut.py --model openai/clip-vit-base-patch32
    python generate_token_embedding_lut.py --model openai/clip-vit-large-patch14
"""

import argparse
import numpy as np
from pathlib import Path


def generate_embeddings(model_name="openai/clip-vit-base-patch32", output_path=None):
    """
    Extract and save token embeddings from CLIP model.
    
    Args:
        model_name: HuggingFace model name (e.g., 'openai/clip-vit-base-patch32')
        output_path: Path to save embeddings. Defaults to clip_token_embeddings.npy
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import CLIPTextModel
        import torch
    except ImportError:
        print("❌ Error: transformers and torch are required!")
        print("\nInstall with:")
        print("  pip install transformers torch")
        print("\nNote: The 'tokenizers' package (already installed) handles text→tokens.")
        print("      This script extracts the embedding LUT: tokens→embeddings.")
        return False
    
    if output_path is None:
        output_path = Path(__file__).parent / "token_embedding_lut.npy"
    
    print("="*80)
    print(f"Generating Token Embeddings for CLIP")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print()
    
    # Load the CLIP text model
    print("[1/3] Loading CLIP text model from HuggingFace...")
    print(f"      Model: {model_name}")
    print("      (This may take a few minutes on first run - downloading ~500MB)")
    
    try:
        model = CLIPTextModel.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("      ✓ Model loaded successfully")
    
    # Extract token embeddings
    print("\n[2/3] Extracting token embedding layer (LUT)...")
    print("      Location: model.text_model.embeddings.token_embedding.weight")
    
    # This is the embedding matrix: shape (vocab_size, embedding_dim)
    # For CLIP ViT-B/32: (49408, 512)
    embeddings = model.text_model.embeddings.token_embedding.weight.detach().cpu().numpy()
    
    print(f"      ✓ Embeddings extracted")
    print(f"      - Shape: {embeddings.shape}")
    print(f"      - Vocabulary size: {embeddings.shape[0]}")
    print(f"      - Embedding dimension: {embeddings.shape[1]}")
    print(f"      - Data type: {embeddings.dtype}")
    print(f"      - Memory: {embeddings.nbytes / (1024 * 1024):.1f} MB")
    
    # Verify expected dimensions for common CLIP models
    expected_dims = {
        "openai/clip-vit-base-patch32": (49408, 512),
        "openai/clip-vit-base-patch16": (49408, 512),
        "openai/clip-vit-large-patch14": (49408, 768),
    }
    
    if model_name in expected_dims:
        expected = expected_dims[model_name]
        if embeddings.shape != expected:
            print(f"      ⚠ Warning: Expected shape {expected}, got {embeddings.shape}")
    
    # Save to file
    print(f"\n[3/3] Saving embeddings to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"      ✓ Saved successfully ({file_size_mb:.1f} MB)")
    
    print("\n" + "="*80)
    print("✓ Token embeddings generated successfully!")
    print("="*80)
    print(f"\nFile: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"\nUsage:")
    print(f"  The tokenizer (already working) converts: text → token_ids")
    print(f"  This LUT converts: token_ids → embeddings")
    print(f"  Then Hailo text encoder processes: embeddings → text features")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP token embeddings for Hailo text encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for CLIP ViT-B/32 (default - matches clip_vit_b_32_text_encoder.hef)
  python generate_token_embedding_lut.py
  
  # Generate for CLIP ViT-L/14
  python generate_token_embedding_lut.py --model openai/clip-vit-large-patch14
  
  # Save to custom location
  python generate_token_embedding_lut.py --output /path/to/token_embedding_lut.npy

Supported Models:
  - openai/clip-vit-base-patch32  (ViT-B/32, 512-dim) [default]
  - openai/clip-vit-base-patch16  (ViT-B/16, 512-dim)
  - openai/clip-vit-large-patch14 (ViT-L/14, 768-dim)
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name (default: openai/clip-vit-base-patch32)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for token embedding LUT .npy file (default: ./token_embedding_lut.npy)"
    )
    
    args = parser.parse_args()
    
    # Convert output to Path if provided
    output_path = Path(args.output) if args.output else None
    
    success = generate_embeddings(args.model, output_path)
    
    if not success:
        return 1
    
    print("\n✓ Done! You can now use:")
    print("  - clip_text_utils.py functions")
    print("  - python clip_app.py --input usb")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
