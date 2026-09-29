import torch
from diffusers import LongSanaVideoPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

pipe = LongSanaVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers", torch_dtype=torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")

prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=481,
    guidance_scale=1.0,
    timesteps=[1000, 960, 889, 727, 0],  # Multi-step denoising per chunk
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]
export_to_video(video, "longsana.mp4", fps=16)
