# CLIP Text Encoder Setup

This directory contains utilities for CLIP text encoding with Hailo hardware acceleration.

## Quick Start

### 1. Generate Required Files (One-Time Setup)

You need three files for the text encoder to work. All generator scripts and generated files are in the `setup/` subfolder:

```bash
cd setup

# Step 1: Generate tokenizer (text → token IDs)
python3 generate_tokenizer.py

# Step 2: Generate token embedding LUT (token IDs → embeddings)
python3 generate_token_embedding_lut.py

# Step 3: Generate text projection matrix (for postprocessing)
python3 generate_text_projection.py
```

**Requirements for generation:**
```bash
pip install tokenizers transformers torch
```

**Note:** The generated files will be created in the `setup/` folder and automatically used by `clip_text_utils.py`.

### 2. Generate Sample Embeddings JSON (Optional)

After generating the required files above, you can create a sample embeddings JSON file with pre-computed text embeddings:

```bash
cd setup

# Generate embeddings.json with sample text entries
python3 build_sample_embeddings_json.py
```

This will:
- Use the Hailo text encoder to generate embeddings for sample texts
- Create an `embeddings.json` file in the parent directory
- Include entries for: person, sea, dog, cat, snake, and empty strings

**Note:** This step requires:
- All three files from Step 1 must be generated first
- A valid Hailo text encoder HEF file
- The `hailo_platform` package installed

The generated `embeddings.json` can be used as a starting point for your own custom embeddings.

### 3. Use the Text Encoder

Once the files are generated, you can use them:

```python
from clip_text_utils import prepare_text_for_hailo_encoder

# Prepare text for Hailo text encoder
result = prepare_text_for_hailo_encoder("A photo of a cat")

# Get the embeddings ready for HEF model
token_embeddings = result['token_embeddings']  # Shape: (1, 77, 512)
last_token_position = result['last_token_position']  # For postprocessing
```

Or run the complete pipeline with inference:

```python
from clip_text_utils import run_text_encoder_inference

# Run inference on Hailo hardware
text_features = run_text_encoder_inference(
    text="A photo of a cat",
    hef_path="clip_vit_b_32_text_encoder.hef"
)
```

## File Overview

### Generated Files (in `setup/` folder)

| File | Size | Purpose | Generator Script |
|------|------|---------|------------------|
| `setup/clip_tokenizer.json` | ~3.5 MB | Converts text → token IDs | `setup/generate_tokenizer.py` |
| `setup/token_embedding_lut.npy` | ~97 MB | Converts token IDs → embeddings | `setup/generate_token_embedding_lut.py` |
| `setup/text_projection.npy` | ~1 MB | Projects encoder output to final embeddings | `setup/generate_text_projection.py` |

### Generated Configuration Files (in main folder)

| File | Size | Purpose | Generator Script |
|------|------|---------|------------------|
| `embeddings.json` | ~200 KB | Pre-computed text embeddings for runtime use | `setup/build_sample_embeddings_json.py` |

**Note:** The `embeddings.json` file will be **overwritten** if you run `build_sample_embeddings_json.py` again. Back up your custom embeddings before regenerating.

### Source Files

| File | Location | Purpose |
|------|----------|---------|
| `clip_text_utils.py` | Main folder | Main utilities for text encoding (load, prepare, infer) |
| `clip_app.py` | Main folder | Full CLIP application with text and image encoding |
| `generate_tokenizer.py` | `setup/` | One-time script to generate tokenizer |
| `generate_token_embedding_lut.py` | `setup/` | One-time script to generate embedding LUT |
| `generate_text_projection.py` | `setup/` | One-time script to generate text projection matrix |
| `build_sample_embeddings_json.py` | `setup/` | Script to generate sample embeddings JSON |

## Architecture

```
Text Input
    ↓
[Tokenizer] (setup/clip_tokenizer.json)
    ↓ Token IDs (integers)
[Token Embedding LUT] (setup/token_embedding_lut.npy)
    ↓ Token Embeddings (vectors)
[Hailo Text Encoder] (clip_vit_b_32_text_encoder.hef)
    ↓ Encoder Output (hidden states)
[Text Projection] (setup/text_projection.npy)
    ↓ Projected embeddings
[L2 Normalization]
    ↓ Text Features (normalized embeddings)
    ↓
[Save to embeddings.json] (Optional, for runtime use)
```

## Model Information

- **Model**: CLIP ViT-B/32
- **HuggingFace ID**: `openai/clip-vit-base-patch32`
- **Vocabulary Size**: 49,408 tokens
- **Embedding Dimension**: 512
- **Max Sequence Length**: 77 tokens

## Notes

- All setup files (tokenizer, embedding LUT, text projection) are stored in the `setup/` subfolder
- These files are extracted from the same CLIP ViT-B/32 model that matches your HEF
- All files only need to be generated once
- After generation, you can uninstall `transformers` and `torch` if desired
- The `tokenizers` package must remain installed for runtime use
- `clip_text_utils.py` automatically looks for files in the `setup/` folder
- **Running `build_sample_embeddings_json.py` will overwrite existing `embeddings.json`** - back up custom embeddings first

## Testing

Test the setup:

```bash
python3 clip_text_utils.py
```

This will verify:
1. Tokenizer loading
2. Token embedding LUT loading  
3. Text preparation pipeline
4. Batch processing

## Troubleshooting

### "Tokenizer not found"
```bash
cd setup
python3 generate_tokenizer.py
```

### "Token embeddings not found"
```bash
cd setup
python3 generate_token_embedding_lut.py
```

### "Text projection file not found"
```bash
cd setup
python3 generate_text_projection.py
```

### "embeddings.json was overwritten"
- **Solution**: The `build_sample_embeddings_json.py` script always overwrites the output file
- **Prevention**: Back up your custom `embeddings.json` before running the script
- **Alternative**: Modify the `json_output_path` variable in the script to save to a different filename

### Missing dependencies
```bash
# For one-time generation
pip install tokenizers transformers torch

# For runtime (after files are generated)
pip install tokenizers  # Only this is needed
```
