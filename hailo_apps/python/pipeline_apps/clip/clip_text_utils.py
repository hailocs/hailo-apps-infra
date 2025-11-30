"""
CLIP Text Utilities
Handles tokenization and token embedding for CLIP text encoder.
Provides tokenizer and pre-computed token embedding LUT (Look-Up Table).
"""

from pathlib import Path
from tokenizers import Tokenizer
import numpy as np
from hailo_platform import FormatType

# Default paths (in setup subfolder)
DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "setup" / "clip_tokenizer.json"
DEFAULT_TOKEN_EMBEDDING_PATH = Path(__file__).parent / "setup" / "token_embedding_lut.npy"
DEFAULT_TEXT_PROJECTION_PATH = Path(__file__).parent / "setup" / "text_projection.npy"


def load_clip_tokenizer(tokenizer_path=None):
    """Load CLIP tokenizer from local file."""
    if tokenizer_path is None:
        tokenizer_path = DEFAULT_TOKENIZER_PATH

    tokenizer_file = Path(tokenizer_path)

    # Check if tokenizer exists locally
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    return Tokenizer.from_file(str(tokenizer_file))


def tokenize_text(text, tokenizer=None, max_length=77, tokenizer_path=None):
    """Tokenize text using CLIP tokenizer and return numpy int32 input_ids."""
    if tokenizer is None:
        tokenizer = load_clip_tokenizer(tokenizer_path)

    # Handle single string or list of strings
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)

    # Tokenize all texts
    all_tokens = []
    for single_text in texts:
        encoding = tokenizer.encode(single_text)
        ids = encoding.ids[:max_length]
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        all_tokens.append(ids)

    # Convert to numpy array
    input_ids = np.array(all_tokens, dtype=np.int32)

    return {"input_ids": input_ids}


def load_token_embeddings(embeddings_path=None):
    """Load pre-computed token embedding matrix (LUT)."""
    if embeddings_path is None:
        embeddings_path = DEFAULT_TOKEN_EMBEDDING_PATH

    embeddings_file = Path(embeddings_path)

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Token embeddings file not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)
    return embeddings


def tokens_to_embeddings(token_ids, token_embeddings=None, embeddings_path=None):
    """Convert token IDs to embedding vectors using token embedding LUT."""
    if token_embeddings is None:
        token_embeddings = load_token_embeddings(embeddings_path)

    # Look up embeddings for each token ID
    embeddings = token_embeddings[token_ids]

    return embeddings.astype(np.float32)


def prepare_text_for_encoder(text, tokenizer=None, token_embeddings=None, max_length=77,
                             tokenizer_path=None, embeddings_path=None):
    """Complete pipeline: text → token IDs → token embeddings."""
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_clip_tokenizer(tokenizer_path)

    # Load token embeddings if not provided
    if token_embeddings is None:
        token_embeddings = load_token_embeddings(embeddings_path)

    # Step 1: Tokenize text to get token IDs
    tokens = tokenize_text(text, tokenizer, max_length, tokenizer_path)
    token_ids = tokens['input_ids']

    # Step 2: Convert token IDs to embeddings
    embeddings = tokens_to_embeddings(token_ids, token_embeddings, embeddings_path)

    return {
        'token_ids': token_ids,
        'token_embeddings': embeddings
    }


def preprocess_for_text_encoder(token_embeddings, token_ids, sequence_length=77,
                                end_of_text_token_id=49407, pad_token_id=0):
    """Preprocess token embeddings for CLIP text encoder (Hailo model input)."""
    batch_size = token_embeddings.shape[0]
    current_length = token_embeddings.shape[1]
    embedding_dim = token_embeddings.shape[2]

    # Calculate padding length
    padding_length = sequence_length - current_length

    if padding_length > 0:
        # Pad with EOT token embedding or zeros
        # Find EOT token positions
        eot_mask = (token_ids == end_of_text_token_id)
        # Use first EOT position per batch or last token if none
        last_token_positions = []
        for b in range(batch_size):
            eot_indices = np.where(eot_mask[b])[0]
            if len(eot_indices) > 0:
                last_pos = int(eot_indices[0])
            else:
                last_pos = current_length - 1
            last_token_positions.append(last_pos)
        last_token_positions = np.array(last_token_positions, dtype=np.int32)

        # Use the embedding at last_token_position as pad embedding
        pad_embeddings = []
        for b in range(batch_size):
            pad_embeddings.append(
                np.repeat(
                    token_embeddings[b, last_token_positions[b]:last_token_positions[b] + 1, :],
                    padding_length,
                    axis=0,
                )
            )
        pad_embeddings = np.stack(pad_embeddings, axis=0)
        token_embeddings = np.concatenate([token_embeddings, pad_embeddings], axis=1)

    elif padding_length < 0:
        # Truncate to sequence_length
        token_embeddings = token_embeddings[:, :sequence_length, :]
        token_ids = token_ids[:, :sequence_length]

        # Recompute last_token_positions on truncated ids
        eot_mask = (token_ids == end_of_text_token_id)
        last_token_positions = []
        for b in range(batch_size):
            eot_indices = np.where(eot_mask[b])[0]
            if len(eot_indices) > 0:
                last_pos = int(eot_indices[0])
            else:
                last_pos = sequence_length - 1
            last_token_positions.append(last_pos)
        last_token_positions = np.array(last_token_positions, dtype=np.int32)

    else:
        # No padding/truncation needed
        eot_mask = (token_ids == end_of_text_token_id)
        last_token_positions = []
        for b in range(batch_size):
            eot_indices = np.where(eot_mask[b])[0]
            if len(eot_indices) > 0:
                last_pos = int(eot_indices[0])
            else:
                last_pos = current_length - 1
            last_token_positions.append(last_pos)
        last_token_positions = np.array(last_token_positions, dtype=np.int32)

    # Ensure the shape is correct
    assert token_embeddings.shape == (batch_size, sequence_length, embedding_dim), \
        f"Expected shape ({batch_size}, {sequence_length}, {embedding_dim}), got {token_embeddings.shape}"

    return {
        'token_embeddings': token_embeddings,
        'last_token_positions': last_token_positions,
    }


def prepare_text_for_hailo_encoder(text, tokenizer=None, token_embeddings=None,
                                   sequence_length=77, tokenizer_path=None,
                                   embeddings_path=None):
    """Prepare text all the way to Hailo CLIP text encoder input (float32 embeddings)."""
    encoded = prepare_text_for_encoder(
        text,
        tokenizer=tokenizer,
        token_embeddings=token_embeddings,
        max_length=sequence_length,
        tokenizer_path=tokenizer_path,
        embeddings_path=embeddings_path,
    )

    preprocessed = preprocess_for_text_encoder(
        encoded['token_embeddings'],
        encoded['token_ids'],
        sequence_length=sequence_length,
    )

    return {
        'token_embeddings': preprocessed['token_embeddings'],
        'last_token_positions': preprocessed['last_token_positions'],
    }


def text_encoding_postprocessing(encoder_output, last_token_positions, text_projection=None,
                                 text_projection_path=None):
    """Project encoder output to final text features using optional projection matrix."""
    if text_projection is None and text_projection_path is not None:
        proj_file = Path(text_projection_path)
        if not proj_file.exists():
            raise FileNotFoundError(f"Text projection file not found: {proj_file}")
        text_projection = np.load(proj_file)

    # Gather the features at last_token_positions
    batch_size = encoder_output.shape[0]
    embed_dim = encoder_output.shape[-1]
    gathered = np.zeros((batch_size, embed_dim), dtype=encoder_output.dtype)
    for b in range(batch_size):
        gathered[b] = encoder_output[b, last_token_positions[b], :]

    if text_projection is not None:
        # Apply linear projection: (batch, dim) x (dim, out_dim)
        gathered = gathered @ text_projection

    return gathered


def run_text_encoder_inference(text, hef_path, vdevice, 
                               tokenizer=None, token_embeddings=None,
                               text_projection=None, tokenizer_path=None,
                               embeddings_path=None, text_projection_path=None,
                               timeout_ms=1000):
    """
    Complete pipeline: text → Hailo text encoder → normalized text embeddings.
    
    This function:
    1. Tokenizes and prepares text for the encoder
    2. Runs inference on Hailo text encoder
    3. Post-processes the output with text projection and normalization
    
    Args:
        text: String or list of strings to encode
        hef_path: Path to CLIP text encoder HEF file
        vdevice: VDevice instance (required)
        tokenizer: Tokenizer instance. If None, loads from tokenizer_path or default.
        token_embeddings: Token embedding matrix. If None, loads from embeddings_path or default.
        text_projection: Text projection matrix. If None, loads from file.
        tokenizer_path: Path to tokenizer JSON file (optional)
        embeddings_path: Path to token embeddings .npy file (optional)
        text_projection_path: Path to text projection .npy file (optional)
        timeout_ms: Inference timeout in milliseconds (default: 1000)
    
    Returns:
        Normalized text embeddings, shape (batch, embedding_dim)
    """
    # Step 1: Prepare text input (tokenization + embedding lookup + preprocessing)
    prepared = prepare_text_for_hailo_encoder(
        text=text,
        tokenizer=tokenizer,
        token_embeddings=token_embeddings,
        tokenizer_path=tokenizer_path,
        embeddings_path=embeddings_path
    )
    input_embeddings = prepared['token_embeddings']  # Shape: (batch, 77, 512)
    last_token_positions = prepared['last_token_positions']  # Shape: (batch,)
    
    # Step 2: Run Hailo inference
    infer_model = vdevice.create_infer_model(str(hef_path))
    with infer_model.configure() as configured_infer_model:
        configured_infer_model.activate()
        bindings = configured_infer_model.create_bindings()
        input_buffer = np.ascontiguousarray(input_embeddings, dtype=np.uint16)
        bindings.input().set_buffer(input_buffer)
        output_buffer = np.empty(infer_model.output().shape).astype(np.uint8)
        bindings.output().set_buffer(output_buffer)
        configured_infer_model.run([bindings], timeout_ms)
        output_buffer = bindings.output().get_buffer()
        configured_infer_model.deactivate()
            
    # Step 3: Post-process the encoder output
    normalized_embeddings = text_encoding_postprocessing(
        encoder_output=output_buffer,
        last_token_positions=last_token_positions,
        text_projection=text_projection,
        text_projection_path=text_projection_path
    )
    
    return normalized_embeddings


# Example usage
if __name__ == "__main__":
    # This module is intended to be used from the CLIP pipeline app, not run directly.
    pass
