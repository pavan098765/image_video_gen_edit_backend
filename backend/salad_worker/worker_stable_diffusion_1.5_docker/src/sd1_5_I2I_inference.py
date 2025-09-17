from stable_diffusion_cpp import StableDiffusion
from PIL import Image

def main():
    # Load SD 1.5 GGUF + VAE
    model = StableDiffusion(
        model_path="../models/sd15-Q4_K_M.gguf",  # your converted model
        vae_path="../models/vae-ft-mse-840000-ema-pruned.safetensors",
    )

    # Generate with image-to-image
    output = model.generate_image(
        prompt="A medieval castle in fantasy landscape, highly detailed",
        negative_prompt="blurry, distorted, lowres",
        images=["input.png"],  # <-- key difference
        strength=0.65,         # how much to transform (0 = keep original, 1 = ignore original)
        width=512,
        height=512,
        sample_steps=30,
        cfg_scale=7.5,
        sample_method="euler",
    )

    output[0].save("output_img2img_sd15.png")
    print("✅ SD 1.5 img2img saved as output_img2img_sd15.png")

if __name__ == "__main__":
    main()
