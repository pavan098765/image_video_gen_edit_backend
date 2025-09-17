from stable_diffusion_cpp import StableDiffusion

def main():
    # SD 1.5 GGUF + VAE
    model = StableDiffusion(
        model_path="../models/sd15-Q4_K_M.gguf",  # converted from .ckpt/.safetensors
        vae_path="../models/vae-ft-mse-840000-ema-pruned.safetensors",
    )

    output = model.generate_image(
        prompt="A cyberpunk city at night, neon lights, highly detailed, artstation",
        negative_prompt="blurry, low quality, distorted",
        width=512,
        height=512,          # SD 1.5 native resolution
        sample_steps=25,
        cfg_scale=7.5,
        sample_method="euler",
    )

    output[0].save("output_sd15.png")
    print("✅ SD 1.5 image saved as output_sd15.png")

if __name__ == "__main__":
    main()
