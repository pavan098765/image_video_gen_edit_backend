"""
Video post-processing utilities for RIFE frame interpolation and Real-ESRGAN upscaling.
Adapted from CogVideoX-5B-Space implementation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from typing import Union, List
import math
import itertools
import safetensors.torch
from spandrel import ModelLoader
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.image_processor import VaeImageProcessor

# Add RIFE import - try multiple possible locations
RIFE_AVAILABLE = False
RIFEModel = None
ssim_matlab = None
RIFE_IMPORT_ERROR = None

# Try importing RIFE from various possible locations
# RIFE is not a pip package - it needs to be available as a local module
# It should be in a 'rife' directory with RIFE_HDv3.py and pytorch_msssim.py

# First, try from local rife directory (most common case)
rife_local = os.path.join(os.path.dirname(__file__), "rife")
if os.path.exists(rife_local):
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from rife.RIFE_HDv3 import Model as RIFEModel
        from rife.pytorch_msssim import ssim_matlab
        RIFE_AVAILABLE = True
        print(f"✓ RIFE loaded from local directory: {rife_local}")
    except ImportError as e:
        RIFE_IMPORT_ERROR = f"Failed to import from local rife directory: {str(e)}"
        # Remove from path if it didn't work
        if os.path.dirname(__file__) in sys.path:
            sys.path.remove(os.path.dirname(__file__))

# If local didn't work, try direct import (if rife is installed as package or in PYTHONPATH)
if not RIFE_AVAILABLE:
    try:
        from rife.RIFE_HDv3 import Model as RIFEModel
        from rife.pytorch_msssim import ssim_matlab
        RIFE_AVAILABLE = True
        print("✓ RIFE loaded from system/PYTHONPATH")
    except ImportError as e:
        if RIFE_IMPORT_ERROR:
            RIFE_IMPORT_ERROR += f"\nAlso failed direct import: {str(e)}"
        else:
            RIFE_IMPORT_ERROR = f"Failed direct import: {str(e)}"

# If still not available, provide helpful error message
if not RIFE_AVAILABLE:
    print(f"⚠️  RIFE not available. Import error details:")
    print(f"   {RIFE_IMPORT_ERROR}")
    print(f"   RIFE needs to be available as a local 'rife' module.")
    print(f"   Expected structure: rife/RIFE_HDv3.py and rife/pytorch_msssim.py")
    print(f"   Checked location: {rife_local}")


def pad_image(img, scale):
    """Pad image to be divisible by scale."""
    _, _, h, w = img.shape
    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, 0, pw - w, ph - h)
    return F.pad(img, padding)


def make_inference(model, I0, I1, upscale_amount, n):
    """Recursive frame interpolation."""
    middle = model.inference(I0, I1, upscale_amount)
    if n == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, upscale_amount, n=n // 2)
    second_half = make_inference(model, middle, I1, upscale_amount, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


@torch.inference_mode()
def ssim_interpolation_rife(model, samples, exp=1, upscale_amount=1, output_device="cpu"):
    """RIFE frame interpolation on tensor format [f, c, h, w]."""
    output = []
    device = samples.device
    
    # [f, c, h, w]
    for b in range(samples.shape[0]):
        frame = samples[b : b + 1]
        _, _, h, w = frame.shape
        I0 = samples[b : b + 1]
        I1 = samples[b + 1 : b + 2] if b + 2 <= samples.shape[0] else samples[-1:]
        I1 = pad_image(I1, upscale_amount)
        
        # [c, h, w]
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        
        if ssim > 0.996:
            I1 = I0
            I1 = pad_image(I1, upscale_amount)
            I1 = make_inference(model, I0, I1, upscale_amount, 1)
            
            I1_small = F.interpolate(I1[0], (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = I1[0]
            I1 = I1[0]
        
        tmp_output = []
        if ssim < 0.2:
            for i in range((2**exp) - 1):
                tmp_output.append(I0)
        else:
            tmp_output = make_inference(model, I0, I1, upscale_amount, 2**exp - 1) if exp else []
        
        frame = pad_image(frame, upscale_amount)
        tmp_output = [frame] + tmp_output
        for i, frame in enumerate(tmp_output):
            output.append(frame.to(output_device))
    return output


def load_torch_file(ckpt, device=None, dtype=torch.float16):
    """Load torch checkpoint file."""
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd
    
    sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    """Replace prefix in state dict keys."""
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    """Calculate tiled scale steps."""
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def module_size(module):
    """Calculate module memory size."""
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    """Tiled upscaling to handle large images."""
    dims = len(tile)
    output = torch.empty(
        [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])),
        device=output_device,
    )
    
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )
        
        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []
            
            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))
            
            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= (1.0 / feather) * (t + 1)
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= (1.0 / feather) * (t + 1)
            
            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])
            
            o += ps * mask
            o_d += mask
            
            if pbar is not None:
                pbar.update(1)
        
        output[b : b + 1] = out / out_div
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    """Tiled upscaling wrapper."""
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


def load_sd_upscale(ckpt, inf_device):
    """Load Real-ESRGAN upscaling model."""
    sd = load_torch_file(ckpt, device=inf_device)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).half()
    return out


def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    """Upscale tensor using Real-ESRGAN with tiling."""
    upscale_model.to(inf_device)
    tile = 512
    overlap = 32
    
    s = tiled_scale(
        samples=tensor.to(torch.float16),
        function=lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale,
        pbar=None,
    )
    
    upscale_model.to(output_device)
    return s


def frames_to_tensor(frames):
    """Convert frames (numpy array or PIL images) to tensor format [f, c, h, w]."""
    if isinstance(frames, np.ndarray):
        # Assume shape is [f, h, w, c] or [h, w, c]
        if len(frames.shape) == 3:
            frames = frames[np.newaxis, ...]  # Add frame dimension
        if frames.shape[-1] == 3:  # [f, h, w, c]
            frames = np.transpose(frames, (0, 3, 1, 2))  # [f, c, h, w]
        # Normalize to [0, 1]
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames).float()
    elif isinstance(frames, list) and isinstance(frames[0], Image.Image):
        # Convert PIL images to tensor
        tensor_frames = []
        for img in frames:
            img_array = np.array(img)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            img_array = img_array.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1)))  # [c, h, w]
            tensor_frames.append(img_tensor)
        tensor = torch.stack(tensor_frames)  # [f, c, h, w]
    else:
        raise ValueError(f"Unsupported frame format: {type(frames)}")
    
    return tensor


def tensor_to_frames(tensor):
    """Convert tensor format [f, c, h, w] back to numpy frames [f, h, w, c]."""
    # Clamp to [0, 1] and convert to uint8
    tensor = torch.clamp(tensor, 0, 1)
    if tensor.dtype != torch.uint8:
        tensor = (tensor * 255).round().byte()
    
    # Convert to numpy and transpose
    frames_np = tensor.cpu().numpy()
    if len(frames_np.shape) == 4:  # [f, c, h, w]
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # [f, h, w, c]
    
    return frames_np


def check_rife_code_available():
    """Check if RIFE code is available and provide helpful message if not."""
    if RIFE_AVAILABLE:
        return True
    
    rife_local = os.path.join(os.path.dirname(__file__), "rife")
    print(f"\n⚠️  RIFE code not available. To enable RIFE frame interpolation:")
    print(f"   1. Clone RIFE repository:")
    print(f"      git clone https://github.com/hzwer/RIFE.git {rife_local}")
    print(f"   2. Or download and extract RIFE code to: {rife_local}")
    print(f"   3. Required files: rife/RIFE_HDv3.py and rife/pytorch_msssim.py")
    if RIFE_IMPORT_ERROR:
        print(f"\n   Import error details: {RIFE_IMPORT_ERROR}")
    return False


def download_rife_model(models_dir):
    """Download RIFE model weights locally.
    
    Note: This downloads the model weights (flownet.pkl), not the RIFE code.
    The RIFE code (Python module) must be available separately.
    """
    rife_model_dir = os.path.join(models_dir, "rife")
    if os.path.exists(rife_model_dir) and os.listdir(rife_model_dir):
        print(f"✓ RIFE model weights already exist at {rife_model_dir}")
        return rife_model_dir
    
    print(f"⬇️ Downloading RIFE model weights to {rife_model_dir}...")
    os.makedirs(rife_model_dir, exist_ok=True)
    
    try:
        snapshot_download(
            "AlexWortega/RIFE",
            local_dir=rife_model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ RIFE model weights downloaded successfully")
        return rife_model_dir
    except Exception as e:
        print(f"❌ Failed to download RIFE model weights: {e}")
        return None


def download_realesrgan_model(models_dir, upscale_factor=4):
    """Download Real-ESRGAN model locally.
    
    Args:
        models_dir: Directory to save models
        upscale_factor: Upscale factor (2, 4, or 8)
    
    Returns:
        Path to downloaded model file or None if failed
    """
    if upscale_factor not in [2, 4, 8]:
        print(f"⚠️  Invalid upscale factor {upscale_factor}, must be 2, 4, or 8. Using 4x.")
        upscale_factor = 4
    
    realesrgan_dir = os.path.join(models_dir, "realesrgan")
    os.makedirs(realesrgan_dir, exist_ok=True)
    
    model_filename = f"RealESRGAN_x{upscale_factor}.pth"
    model_file = os.path.join(realesrgan_dir, model_filename)
    
    if os.path.exists(model_file):
        print(f"✓ Real-ESRGAN model already exists at {model_file}")
        return model_file
    
    print(f"⬇️ Downloading Real-ESRGAN {upscale_factor}x model to {realesrgan_dir}...")
    
    try:
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename=model_filename,
            local_dir=realesrgan_dir,
        )
        print(f"✓ Real-ESRGAN {upscale_factor}x model downloaded successfully")
        return model_file
    except Exception as e:
        print(f"❌ Failed to download Real-ESRGAN model: {e}")
        return None


def load_rife_model(model_path):
    """Load RIFE model for frame interpolation.
    
    Args:
        model_path: Path to RIFE model directory (should contain flownet.pkl)
    
    Returns:
        Loaded RIFE model or None if failed
    """
    if not RIFE_AVAILABLE:
        print(f"⚠️  RIFE not available (import failed)")
        if RIFE_IMPORT_ERROR:
            print(f"   Error details: {RIFE_IMPORT_ERROR}")
        print(f"   RIFE code needs to be available as a local 'rife' module.")
        print(f"   You may need to clone/download RIFE code from: https://github.com/hzwer/RIFE")
        return None
    
    try:
        # RIFE model's load_model() expects a DIRECTORY path and internally appends "/flownet.pkl"
        # So if we get a file path, we need to use its parent directory
        if os.path.isfile(model_path):
            # It's a file - use the parent directory
            model_dir = os.path.dirname(model_path)
            print(f"   📁 RIFE model file provided, using directory: {model_dir}")
        elif os.path.isdir(model_path):
            # It's a directory - use it directly
            model_dir = model_path
            # Verify flownet.pkl exists in the directory
            pkl_file = os.path.join(model_dir, "flownet.pkl")
            if not os.path.exists(pkl_file):
                # Try to find any .pkl file
                pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                if not pkl_files:
                    raise FileNotFoundError(f"No .pkl file found in {model_dir}")
                print(f"   ℹ️  Using {pkl_files[0]} instead of flownet.pkl")
        else:
            raise FileNotFoundError(f"RIFE model path not found: {model_path}")
        
        # Verify the directory exists and has a .pkl file
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"RIFE model directory not found: {model_dir}")
        
        print(f"   📁 Loading RIFE model from directory: {model_dir}")
        model = RIFEModel()
        # RIFE's load_model expects directory path - it will append "/flownet.pkl" internally
        model.load_model(model_dir, -1)
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️  Failed to load RIFE model: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_rife_exp(original_fps, target_fps):
    """Calculate RIFE exp parameter to achieve target FPS.
    
    RIFE inserts 2^exp - 1 frames between each pair of frames.
    - exp=0: 0 frames (1x, no interpolation)
    - exp=1: 1 frame (2x FPS: 16fps -> 32fps)
    - exp=2: 3 frames (4x FPS: 16fps -> 64fps)
    - exp=3: 7 frames (8x FPS: 16fps -> 128fps)
    
    Args:
        original_fps: Original frame rate
        target_fps: Desired frame rate
    
    Returns:
        exp value (0-3) that gets closest to target FPS
    """
    if target_fps <= original_fps:
        return 0  # No interpolation needed
    
    ratio = target_fps / original_fps
    frames_per_gap = ratio - 1
    
    # Find the smallest exp that gives at least the desired frames
    # We want 2^exp - 1 >= frames_per_gap
    # So exp >= log2(frames_per_gap + 1)
    exp = max(0, math.ceil(math.log2(frames_per_gap + 1)))
    
    # Cap at 3 (max reasonable for RIFE)
    exp = min(exp, 3)
    
    # Calculate actual FPS we'll get
    actual_fps = original_fps * (2 ** exp)
    
    return exp, actual_fps


def apply_rife_interpolation(model, frames, device="cuda", original_fps=16, target_fps=None):
    """Apply RIFE frame interpolation to frames (numpy array).
    
    Args:
        model: RIFE model
        frames: Input frames (numpy array [f, h, w, c] or list of PIL Images)
        device: Device to run on
        original_fps: Original frame rate
        target_fps: Target frame rate (if None, doubles the FPS)
    
    Returns:
        Interpolated frames (numpy array [f, h, w, c]), actual_fps
    """
    if model is None or not RIFE_AVAILABLE:
        return frames, original_fps
    
    try:
        # Calculate exp parameter
        if target_fps is None:
            # Default: double FPS
            exp = 1
            actual_fps = original_fps * 2
        else:
            exp, actual_fps = calculate_rife_exp(original_fps, target_fps)
            if exp == 0:
                print(f"   ℹ️  Target FPS ({target_fps}) <= original FPS ({original_fps}), skipping interpolation")
                return frames, original_fps
        
        # Convert frames to tensor [f, c, h, w] in [0, 1] range
        if isinstance(frames, np.ndarray):
            # Already numpy array - ensure correct format
            if len(frames.shape) == 4 and frames.shape[-1] == 3:  # [f, h, w, c]
                # Convert to [f, c, h, w] and normalize
                if frames.dtype == np.uint8:
                    frames_norm = frames.astype(np.float32) / 255.0
                else:
                    frames_norm = frames.astype(np.float32)
                tensor = torch.from_numpy(np.transpose(frames_norm, (0, 3, 1, 2))).float()
            else:
                tensor = frames_to_tensor(frames)
        else:
            tensor = frames_to_tensor(frames)
        
        tensor = tensor.to(device)
        
        # Apply RIFE interpolation
        interpolated_frames = ssim_interpolation_rife(model, tensor, exp=exp, upscale_amount=1, output_device="cpu")
        
        # Stack frames back to tensor [f, c, h, w]
        pt_image = torch.stack([f.squeeze(0) for f in interpolated_frames])
        
        # Use VaeImageProcessor for proper conversion (same as CogVideoX)
        # This ensures colors are preserved correctly
        pt_image = pt_image.clamp(0, 1)  # Ensure [0, 1] range
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # [f, h, w, c]
        
        return image_np, actual_fps
    except Exception as e:
        print(f"⚠️  RIFE interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return frames, original_fps


def apply_realesrgan_upscaling(model, frames, device="cuda"):
    """Apply Real-ESRGAN upscaling to frames (numpy array).
    
    Args:
        model: Real-ESRGAN model (already loaded with specific upscale factor)
        frames: Input frames (numpy array [f, h, w, c] or list of PIL Images)
        device: Device to run on
    
    Returns:
        Upscaled frames (numpy array [f, h, w, c])
    """
    if model is None:
        return frames
    
    try:
        # Convert frames to tensor [f, c, h, w] in [0, 1] range
        if isinstance(frames, np.ndarray):
            # Already numpy array - ensure correct format
            if len(frames.shape) == 4 and frames.shape[-1] == 3:  # [f, h, w, c]
                # Convert to [f, c, h, w] and normalize
                if frames.dtype == np.uint8:
                    frames_norm = frames.astype(np.float32) / 255.0
                else:
                    frames_norm = frames.astype(np.float32)
                tensor = torch.from_numpy(np.transpose(frames_norm, (0, 3, 1, 2))).float()
            else:
                tensor = frames_to_tensor(frames)
        else:
            tensor = frames_to_tensor(frames)
        
        tensor = tensor.to(device)
        
        # Upscale using tiled upscaling
        upscaled_tensor = upscale(model, tensor, device, "cpu")
        
        # Convert back to numpy using VaeImageProcessor for proper color space handling
        # This ensures colors are preserved correctly
        upscaled_tensor = upscaled_tensor.clamp(0, 1)  # Ensure [0, 1] range
        
        # Use VaeImageProcessor for proper conversion (same as CogVideoX)
        image_np = VaeImageProcessor.pt_to_numpy(upscaled_tensor)  # [f, h, w, c]
        
        return image_np
    except Exception as e:
        print(f"⚠️  Real-ESRGAN upscaling failed: {e}")
        import traceback
        traceback.print_exc()
        return frames
