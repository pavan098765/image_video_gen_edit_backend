# glm_ocr_inference.py
# ✅ GLM-OCR Inference Worker
# Model: zai-org/GLM-OCR
# Supports Text Recognition, Formula Recognition, and Table Recognition

import gc
import os
import sys
import time
import base64
import json
import tempfile
from io import BytesIO
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

import torch
from PIL import Image

# Set CUDA memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Aggressive GPU cleanup at import
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_start = time.time()
    merge_main()
    merge_time = time.time() - merge_start
    print(f"✅ Merge completed in {merge_time:.1f}s")
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# ==============================
# Environment Configuration
# ==============================
GLM_OCR_DIR = os.environ.get("GLM_OCR_LOCAL_PATH", "/app/models/glm_ocr")

# Prompt types supported
PROMPT_TYPES = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:"
}

# ==============================
# Pipeline Cache
# ==============================
PIPELINE_CACHE: Dict[str, Any] = {}

# ==============================
# Helper Functions
# ==============================
def cleanup_gpu_memory(verbose: bool = False):
    """Thorough GPU memory cleanup."""
    try:
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Before cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(f"[GPU] After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")


def load_image_from_base64(img_b64: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")


def load_image_from_url(image_url: str) -> Image.Image:
    """Load PIL Image from URL."""
    import requests
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_image_to_temp(image: Image.Image, job_id: str, suffix: str = ".png") -> str:
    """Save PIL Image to temporary file."""
    temp_dir = os.path.join(tempfile.gettempdir(), f"glm_ocr_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"input_image{suffix}")
    image.save(temp_path)
    return temp_path


# ==============================
# Pipeline Loading
# ==============================
def get_glm_ocr_pipeline(cache_key: str = "glm_ocr"):
    """
    Load or retrieve cached GLM-OCR pipeline.
    
    Returns:
        Dict with keys: processor, model, device
    """
    global PIPELINE_CACHE
    
    if cache_key in PIPELINE_CACHE:
        print(f"📦 Using cached pipeline: {cache_key}")
        sys.stdout.flush()
        return PIPELINE_CACHE[cache_key]
    
    print(f"🚀 Loading GLM-OCR pipeline")
    print(f"   Model directory: {GLM_OCR_DIR}")
    sys.stdout.flush()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"   Device: {device}")
    sys.stdout.flush()
    
    start_time = time.time()
    print(f"⏳ Starting GPU cleanup...")
    sys.stdout.flush()
    cleanup_gpu_memory(verbose=True)
    
    try:
        print(f"📦 Importing transformers modules...")
        sys.stdout.flush()
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print(f"✅ Transformers imported successfully")
        sys.stdout.flush()
        
        # CRITICAL: Only use local files - never download from HuggingFace
        if not os.path.exists(GLM_OCR_DIR):
            error_msg = (
                f"❌ Model directory not found: {GLM_OCR_DIR}\n"
                f"   Models must be pre-downloaded to this path.\n"
                f"   Run prep_model.py to download models first."
            )
            print(error_msg)
            sys.stdout.flush()
            raise FileNotFoundError(error_msg)
        
        try:
            files = os.listdir(GLM_OCR_DIR)
            if not files:
                error_msg = (
                    f"❌ Model directory is empty: {GLM_OCR_DIR}\n"
                    f"   Models must be pre-downloaded to this path.\n"
                    f"   Run prep_model.py to download models first."
                )
                print(error_msg)
                sys.stdout.flush()
                raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"❌ Error accessing model directory {GLM_OCR_DIR}: {e}"
            print(error_msg)
            sys.stdout.flush()
            raise
        
        print(f"📁 Using LOCAL model files only (no HuggingFace download)")
        print(f"   Model directory: {GLM_OCR_DIR}")
        print(f"   Files found: {len(files)} items")
        sys.stdout.flush()
        
        # Load processor and model from LOCAL files only
        print(f"⏳ Loading processor from local files...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            processor = AutoProcessor.from_pretrained(
                GLM_OCR_DIR,
                local_files_only=True
            )
            print(f"✅ Processor loaded in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load processor: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        print(f"⏳ Loading model from local files...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=GLM_OCR_DIR,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            print(f"✅ Model loaded in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load model: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        elapsed = time.time() - start_time
        print(f"✅ GLM-OCR pipeline fully loaded in {elapsed:.1f}s")
        sys.stdout.flush()
        
        PIPELINE_CACHE[cache_key] = {
            "processor": processor,
            "model": model,
            "device": device,
        }
        
        return PIPELINE_CACHE[cache_key]
        
    except ImportError as e:
        print(f"❌ transformers not installed! Error: {e}")
        sys.stdout.flush()
        raise ImportError(
            f"GLM-OCR requires transformers>=5.1.0. "
            f"Install with: pip install 'git+https://github.com/huggingface/transformers.git'. "
            f"Original error: {e}"
        )
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


def preload_pipeline():
    """
    Preload pipeline in background.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("PIPELINE PRELOAD: Starting pipeline preload...")
        print("=" * 80)
        sys.stdout.flush()
        
        start_time = time.time()
        result = get_glm_ocr_pipeline()
        
        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"✅ Pipeline preloaded successfully in {elapsed:.1f}s")
        print("=" * 80)
        sys.stdout.flush()
        return True
    except Exception as e:
        print("=" * 80)
        print(f"❌ Pipeline preload failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


def unload_pipeline():
    """Unload pipeline(s) and free GPU memory."""
    global PIPELINE_CACHE
    PIPELINE_CACHE.clear()
    print("🧹 Unloaded all pipelines")
    cleanup_gpu_memory(verbose=True)


# ==============================
# Inference Functions
# ==============================
def run_inference_glm_ocr(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    GLM-OCR inference for text, formula, or table recognition.
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt (can be "Text Recognition:", "Formula Recognition:", or "Table Recognition:")
        params:
            - input_image: Base64 encoded image or image URL (required)
            - prompt_type: One of "text", "formula", "table" (default: "text")
            - max_new_tokens: Maximum tokens to generate (default: 8192)
            - output_dir: Output directory (default: /app/worker/outputs)
            - output_format: "txt" or "json" (default: "txt")
    
    Returns:
        Absolute path to generated text file
    """
    import logging
    logger = logging.getLogger("worker")
    
    logger.info(f"[{job_id}] 🎯 Starting GLM-OCR inference")
    print(f"🎯 Starting GLM-OCR job: {job_id}")
    sys.stdout.flush()
    
    cleanup_gpu_memory()
    
    # Parameters
    input_image = params.get("input_image")
    if not input_image:
        raise ValueError("input_image is required in params (base64 or URL)")
    
    prompt_type = params.get("prompt_type", "text")
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of {list(PROMPT_TYPES.keys())}")
    
    # Use provided prompt or default based on prompt_type
    if not prompt or prompt.strip() == "":
        prompt = PROMPT_TYPES[prompt_type]
    else:
        # If prompt is provided, use it as-is
        pass
    
    max_new_tokens = int(params.get("max_new_tokens", 8192))
    output_dir = params.get("output_dir", "/app/worker/outputs")
    output_format = params.get("output_format", "txt")
    
    logger.info(f"[{job_id}] 📐 Configuration: prompt_type={prompt_type}, max_tokens={max_new_tokens}, format={output_format}")
    print(f"📐 Configuration:")
    print(f"   Prompt type: {prompt_type}")
    print(f"   Prompt: {prompt}")
    print(f"   Max tokens: {max_new_tokens}")
    print(f"   Output format: {output_format}")
    sys.stdout.flush()
    
    try:
        # Load pipeline
        logger.info(f"[{job_id}] Loading pipeline...")
        data = get_glm_ocr_pipeline()
        processor = data["processor"]
        model = data["model"]
        device = data["device"]
        logger.info(f"[{job_id}] Pipeline loaded successfully")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare input image
        logger.info(f"[{job_id}] Loading input image...")
        print(f"🖼️  Loading input image...")
        sys.stdout.flush()
        
        try:
            if input_image.startswith("http"):
                image = load_image_from_url(input_image)
            elif input_image.startswith("data:") or len(input_image) > 500:
                image = load_image_from_base64(input_image)
            else:
                # Assume it's a file path
                image = Image.open(input_image).convert("RGB")
            logger.info(f"[{job_id}] Image loaded successfully: {image.size}")
        except Exception as e:
            error_msg = f"Failed to load input image: {e}"
            logger.error(f"[{job_id}] ❌ {error_msg}")
            print(f"❌ {error_msg}")
            sys.stdout.flush()
            raise
        
        # Save image to temp file for processor (processor expects file path in "url" field)
        image_path = save_image_to_temp(image, job_id)
        
        # Prepare messages in the format expected by GLM-OCR (matching Instructions.txt format)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_path  # File path as shown in Instructions.txt
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]
        
        logger.info(f"[{job_id}] 🚀 Starting OCR generation...")
        print(f"🚀 Generating OCR output...")
        sys.stdout.flush()
        
        inference_start_time = time.time()
        
        try:
            # Process messages
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            
            # Remove token_type_ids if present
            inputs.pop("token_type_ids", None)
            
            # Generate
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # Decode output
            output_text = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )
            
            inference_elapsed = time.time() - inference_start_time
            logger.info(f"[{job_id}] ✅ Generation completed in {inference_elapsed:.1f}s")
            print(f"   ✅ Generation completed in {inference_elapsed:.1f}s")
            sys.stdout.flush()
            
            # Save output
            if output_format == "json":
                output_path = os.path.join(output_dir, f"glm_ocr_{job_id}.json")
                output_data = {
                    "job_id": job_id,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "output_text": output_text,
                    "max_tokens": max_new_tokens
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                output_path = os.path.join(output_dir, f"glm_ocr_{job_id}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_text)
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                logger.info(f"[{job_id}] ✅ Output saved: {output_path} ({file_size:.2f} KB)")
                print(f"✅ Output saved: {output_path} ({file_size:.2f} KB)")
                print(f"   Output preview: {output_text[:200]}...")
            else:
                error_msg = f"Output file not found at expected path: {output_path}"
                logger.error(f"[{job_id}] ⚠️  WARNING: {error_msg}")
                print(f"⚠️  WARNING: {error_msg}")
            sys.stdout.flush()
            
            return output_path
            
        except Exception as pipeline_error:
            error_type = type(pipeline_error).__name__
            error_msg = f"Pipeline generation failed: {error_type}: {pipeline_error}"
            logger.error(f"[{job_id}] ❌ {error_msg}")
            logger.exception(f"[{job_id}] Full traceback:")
            print(f"❌ {error_msg}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Inference failed: {error_type}: {e}"
        logger.error(f"[{job_id}] ❌ {error_msg}")
        logger.exception(f"[{job_id}] Full traceback:")
        print(f"❌ {error_msg}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        logger.info(f"[{job_id}] 🧹 Cleaning up GPU memory...")
        print(f"🧹 Cleaning up GPU memory...")
        sys.stdout.flush()
        cleanup_gpu_memory(verbose=True)


# ==============================
# Main usage (for testing)
# ==============================
if __name__ == "__main__":
    import uuid
    
    print("=" * 80)
    print("GLM-OCR - Direct Inference Test")
    print("=" * 80)
    print("This script tests the inference function directly.")
    print("For production use, use the FastAPI app (app.py)")
    print("=" * 80)
    sys.stdout.flush()
    
    job_id = str(uuid.uuid4())
    
    # Example: You would load an image here
    # For testing, create a dummy image
    input_img = Image.new("RGB", (512, 512), color="white")
    
    params = {
        "input_image": None,  # Would be base64 or URL
        "prompt_type": "text",
        "max_new_tokens": 8192,
        "output_dir": "./outputs",
        "output_format": "txt"
    }
    
    # Convert image to base64 for testing
    import io
    buffer = io.BytesIO()
    input_img.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    params["input_image"] = image_b64
    
    try:
        output_file = run_inference_glm_ocr(
            job_id,
            "Text Recognition:",
            params
        )
        print(f"\n✅ [GLM-OCR] Saved output at: {output_file}")
    except Exception as e:
        print(f"\n❌ [GLM-OCR] Test failed: {e}")
        import traceback
        traceback.print_exc()
