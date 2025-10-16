"""
submit_job_multi.py
Enhanced job submitter that supports multiple requests across different models.
"""

import os
import sys
import json
import base64
import hmac
import hashlib
import requests
import asyncio
import argparse
from io import BytesIO
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "https://genedit.mooo.com")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "SECRETKEY")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "SECRET")

# Model configurations
MODEL_CONFIGS = {
    "sd15": {
        "prompt": "A serene landscape with mountains and a lake",
        "negative_prompt": "blurry, distorted, lowres",
        "params": {
            "steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
        }
    },
    "sd35": {
        "prompt": "A futuristic cityscape at night",
        "negative_prompt": "blurry, distorted, lowres",
        "params": {
            "steps": 25,
            "guidance_scale": 8.0,
            "width": 768,
            "height": 768,
        }
    },
    "sdxl": {
        "prompt": "A detailed portrait of a fantasy character",
        "negative_prompt": "blurry, distorted, lowres",
        "params": {
            "steps": 30,
            "guidance_scale": 9.0,
            "width": 1024,
            "height": 1024,
        }
    },
    "diffusers_outpaint": {
        "prompt": "Extend this image naturally",
        "negative_prompt": "blurry, distorted, lowres",
        "params": {
            "steps": 25,
            "guidance_scale": 7.5,
            "width": 768,
            "height": 768,
        }
    },
    "qwen_image_edit": {
        "prompt": "Change body orientation to face the camera directly",
        "negative_prompt": "blurry, artifacts",
        "params": {
            "steps": 8,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
        }
    },
    "qwen_image_edit_2509": {
        "prompt": "Change body orientation to face the camera directly",
        "negative_prompt": "blurry, distorted",
        "params": {
            "steps": 8,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
        }
    },
    "flux_kontext_dev": {
        "prompt": "Change body orientation to face the camera directly",
        "negative_prompt": "blurry, distorted",
        "params": {
            "steps": 28,
            "guidance_scale": 0.0,
            "width": 512,
            "height": 512,
        }
    },
    "fastwan_21_1b": {
        "prompt": "Enhance the details of this image",
        "negative_prompt": "blurry, noise",
        "params": {
            "steps": 20,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
        }
    },
    "fastwan_22_5b": {
        "prompt": "Improve image quality and sharpness",
        "negative_prompt": "blurry, noise",
        "params": {
            "steps": 20,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
        }
    },
    "image_upscaler_aurasrv2": {
        "prompt": "Upscale and enhance image quality",
        "negative_prompt": "blurry, artifacts",
        "params": {
            "scale": 2,
        }
    },
    "image_upscaler_realesrgan": {
        "prompt": "Upscale image with enhanced details",
        "negative_prompt": "blurry, noise",
        "params": {
            "scale": 4,
        }
    },
    "image_upscaler_swin2sr": {
        "prompt": "Super-resolution enhancement",
        "negative_prompt": "blurry, lowres",
        "params": {
            "scale": 2,
        }
    }
}

# Sample image URLs for different models
SAMPLE_IMAGES = {
    "default": "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/input_image.jpg",
    "upscaler": "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/input_image.jpg",
    "edit": "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg",
}

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_and_encode_image(url: str) -> str:
    """Fetch image and return base64-encoded string."""
    try:
        print(f"🖼️  Downloading image from: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        image_data = BytesIO(resp.content)
        return base64.b64encode(image_data.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        sys.exit(1)

def sign_payload(payload: dict) -> dict:
    """Return headers with X-Signature + Authorization."""
    payload_bytes = json.dumps(payload).encode("utf-8")
    signature = "sha256=" + hmac.new(
        CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-Signature": signature,
        "Content-Type": "application/json",
    }
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    return headers

async def submit_request(model: str, request_id: int) -> dict:
    """Submit a single request to the controller."""
    config = MODEL_CONFIGS[model]
    
    # Select appropriate image URL based on model type
    image_url = SAMPLE_IMAGES["default"]
    if "upscaler" in model:
        image_url = SAMPLE_IMAGES["upscaler"]
    elif "edit" in model:
        image_url = SAMPLE_IMAGES["edit"]
    
    # Prepare request payload
    params = config["params"].copy()
    params["init_image"] = fetch_and_encode_image(image_url)
    params["seed"] = request_id  # Use request_id as seed for variety
    params["timeout"] = 300
    params["job_id"] = f"{model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{request_id}"  # Ensure unique job ID
    
    payload = {
        "model": model,
        "prompt": f"{config['prompt']} (Request {request_id})",
        "params": params,
    }
    
    headers = sign_payload(payload)
    url = CONTROLLER_URL.rstrip("/") + "/submit"
    
    job_id = params["job_id"]
    print(f"\n🚀 Submitting {model} job #{request_id} (Job ID: {job_id}) to: {url}")
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        result["job_id"] = job_id  # Add job_id to result for tracking
        print(f"✅ Request {request_id} for {model} submitted successfully with Job ID: {job_id}")
        return result
    except Exception as e:
        print(f"❌ Request {request_id} for {model} failed: {e}")
        return {"error": str(e)}

async def process_model_requests(model: str, num_requests: int) -> List[dict]:
    """Process multiple requests for a specific model."""
    tasks = []
    for i in range(num_requests):
        tasks.append(submit_request(model, i + 1))
    return await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Submit multiple jobs to different models")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), 
                      help="Models to test (space-separated list)")
    parser.add_argument("--requests", type=int, default=1, 
                      help="Number of requests per model")
    parser.add_argument("--parallel", action="store_true", 
                      help="Run requests in parallel")
    parser.add_argument("--output", type=str, default="results.json",
                      help="Output file for results")
    
    args = parser.parse_args()
    
    # If no models specified, use all models
    models = args.models if args.models else list(MODEL_CONFIGS.keys())
    
    print(f"\n🎯 Testing {len(models)} models with {args.requests} requests each")
    print(f"📝 Models: {', '.join(models)}")
    print(f"⚙️  Mode: {'Parallel' if args.parallel else 'Sequential'}")
    
    results = {}
    job_ids = set()  # Track unique job IDs
    start_time = datetime.now()
    
    async def run_tests():
        if args.parallel:
            # Run all requests for all models in parallel
            tasks = []
            for model in models:
                tasks.extend([submit_request(model, i + 1) for i in range(args.requests)])
            results_list = await asyncio.gather(*tasks)
            
            # Organize results by model and track job IDs
            for i, result in enumerate(results_list):
                model = models[i // args.requests]
                if model not in results:
                    results[model] = []
                results[model].append(result)
                if "job_id" in result:
                    job_ids.add(result["job_id"])
        else:
            # Run models sequentially
            for model in models:
                results[model] = await process_model_requests(model, args.requests)
    
    # Run the async event loop
    asyncio.run(run_tests())
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results
    output = {
        "timestamp": start_time.isoformat(),
        "duration": duration,
        "config": {
            "models": models,
            "requests_per_model": args.requests,
            "parallel": args.parallel
        },
        "total_jobs": len(job_ids),
        "unique_job_ids": list(job_ids),
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✨ Testing completed in {duration:.2f} seconds")
    print(f"📊 Results saved to {args.output}")

if __name__ == "__main__":
    main()