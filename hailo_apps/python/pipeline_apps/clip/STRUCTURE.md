# CLIP Directory Structure

## Main Folder (clip/)
```
clip/
├── README_SETUP.md           # Complete setup documentation
├── clip_text_utils.py        # Main utility functions (uses setup/ files)
├── clip_app.py              # Full CLIP application
└── setup/                   # Setup scripts and generated files
    ├── generate_tokenizer.py
    ├── generate_token_embedding_lut.py
    ├── generate_text_projection.py
    ├── clip_tokenizer.json          # Generated (3.5 MB)
    ├── token_embedding_lut.npy      # Generated (97 MB)
    └── text_projection.npy          # Generated (1 MB)
```

## Key Changes

1. **All generator scripts** → Moved to `setup/` folder
2. **All generated files** → Stored in `setup/` folder
3. **clip_text_utils.py** → Updated to look for files in `setup/`
4. **README files** → Merged into single `README_SETUP.md` in main folder

## Usage

### Generate Files (One-Time)
```bash
cd setup
python3 generate_tokenizer.py
python3 generate_token_embedding_lut.py
python3 generate_text_projection.py
```

### Use in Code
```python
# clip_text_utils.py automatically uses setup/ folder
from clip_text_utils import run_text_encoder_inference

text_features = run_text_encoder_inference(
    text="A photo of a cat",
    hef_path="clip_vit_b_32_text_encoder.hef"
)
```

## Benefits

✓ Clean separation of setup vs runtime code
✓ Generated files organized in one place
✓ Easy to gitignore the entire setup/ folder if needed
✓ Clear documentation in main folder
