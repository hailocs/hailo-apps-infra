#!/usr/bin/env python3
"""
Generate Text Projection Matrix for CLIP text encoder.

This script extracts the text projection layer from CLIP models and saves it
as a numpy array for use in postprocessing after Hailo text encoder inference.

The text projection matrix converts the encoder's hidden state to the final
text embedding dimension and is applied after the text encoder runs.

Requirements:
    pip install transformers torch

Usage:
    # For CLIP ViT-B/32 (default)
    python generate_text_projection.py
    
    # For specific model
    python generate_text_projection.py --model openai/clip-vit-base-patch32
    python generate_text_projection.py --model openai/clip-vit-large-patch14
"""

import argparse
import numpy as np
from pathlib import Path


def generate_text_projection(model_name="openai/clip-vit-base-patch32", output_path=None):
    """
    Extract and save text projection matrix from CLIP model.
    
    Args:
        model_name: HuggingFace model name (e.g., 'openai/clip-vit-base-patch32')
        output_path: Path to save projection. Defaults to text_projection.npy
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import CLIPModel
        import torch
    except ImportError:
        print("❌ Error: transformers and torch are required!")
        print("\nInstall with:")
        print("  pip install transformers torch")
        print("\nNote: This script extracts the text projection matrix used in postprocessing.")
        print("      It's applied AFTER the Hailo text encoder inference.")
        return False
    
    if output_path is None:
        output_path = Path(__file__).parent / "text_projection.npy"
    
    print("="*80)
    print(f"Generating Text Projection Matrix for CLIP")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print()
    
    # Load the CLIP text model
    print("[1/3] Loading CLIP text model from HuggingFace...")
    print(f"      Model: {model_name}")
    print("      (This may take a few minutes on first run - downloading ~500MB)")
    
    try:
        model = CLIPModel.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("      ✓ Model loaded successfully")
    
    # Extract text projection
    print("\n[2/3] Extracting text projection layer...")
    print("      Location: model.text_projection.weight")
    
    # The text projection is a linear layer that projects from hidden_dim to projection_dim
    # For CLIP ViT-B/32: (512, 512) - projects 512-dim hidden state to 512-dim embedding
    # For CLIP ViT-L/14: (768, 768)
    # Note: In CLIPModel (full model), text_projection is a Linear layer at the top level
    text_projection = model.text_projection.weight.detach().cpu().numpy()
    
    print(f"      ✓ Text projection extracted")
    print(f"      - Shape: {text_projection.shape}")
    print(f"      - Hidden dimension (input): {text_projection.shape[1]}")
    print(f"      - Projection dimension (output): {text_projection.shape[0]}")
    print(f"      - Data type: {text_projection.dtype}")
    print(f"      - Memory: {text_projection.nbytes / (1024 * 1024):.2f} MB")
    
    # Verify expected dimensions for common CLIP models
    expected_dims = {
        "openai/clip-vit-base-patch32": (512, 512),
        "openai/clip-vit-base-patch16": (512, 512),
        "openai/clip-vit-large-patch14": (768, 768),
    }
    
    if model_name in expected_dims:
        expected = expected_dims[model_name]
        if text_projection.shape != expected:
            print(f"      ⚠ Warning: Expected shape {expected}, got {text_projection.shape}")
    
    # Save to file
    print(f"\n[3/3] Saving text projection to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, text_projection)
    file_size_kb = output_path.stat().st_size / 1024
    print(f"      ✓ Saved successfully ({file_size_kb:.1f} KB)")
    
    print("\n" + "="*80)
    print("✓ Text projection matrix generated successfully!")
    print("="*80)
    print(f"\nFile: {output_path}")
    print(f"Shape: {text_projection.shape}")
    print(f"Size: {file_size_kb:.1f} KB")
    print(f"\nUsage:")
    print(f"  This matrix is applied in postprocessing AFTER Hailo text encoder inference:")
    print(f"  1. Extract EOT token position from encoder output")
    print(f"  2. Apply projection: final_embedding = encoder_output @ text_projection.T")
    print(f"  3. L2 normalize the result")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP text projection matrix for postprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for CLIP ViT-B/32 (default - matches clip_vit_b_32_text_encoder.hef)
  python generate_text_projection.py
  
  # Generate for CLIP ViT-L/14
  python generate_text_projection.py --model openai/clip-vit-large-patch14
  
  # Save to custom location
  python generate_text_projection.py --output /path/to/text_projection.npy

Supported Models:
  - openai/clip-vit-base-patch32  (ViT-B/32, 512x512) [default]
  - openai/clip-vit-base-patch16  (ViT-B/16, 512x512)
  - openai/clip-vit-large-patch14 (ViT-L/14, 768x768)

Note:
  The text projection is applied AFTER the Hailo text encoder inference,
  not inside the HEF model. It's part of the Python postprocessing.
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
        help="Output path for text projection .npy file (default: ./text_projection.npy)"
    )
    
    args = parser.parse_args()
    
    # Convert output to Path if provided
    output_path = Path(args.output) if args.output else None
    
    success = generate_text_projection(args.model, output_path)
    
    if not success:
        return 1
    
    print("\n✓ Done! You can now use:")
    print("  - text_encoding_postprocessing() in clip_text_utils.py")
    print("  - run_text_encoder_inference() for complete pipeline")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
