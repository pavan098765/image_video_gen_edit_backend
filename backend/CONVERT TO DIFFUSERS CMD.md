TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "d9d8a62fa13d36b44cbb16188734e5dc17eec3fd65c2ebc4e0d533bd9bc6c33f",
)


python upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers" --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw" --token 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


https://civitai.com/models/1888857?modelVersionId=2137972 FEM BD ENHANCER 

https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/tree/main


https://github.com/huggingface/diffusers/blob/main/scripts/convert_wan_to_diffusers.py

# Download only two specific files from the repo
python convert_wan_to_diffusers.py --output_path ./wan_out --model_type Wan2.2-T2V-A14B \
  --hf_repo FX-FeiHou/wan2.2-Remix \
  --hf_transformer_file NSFW/Wan2.2_Remix_NSFW_t2v_14b_high_lighting_v2.0.safetensors \
  --hf_transformer_2_file NSFW/Wan2.2_Remix_NSFW_t2v_14b_low_lighting_v2.0.safetensors

# Sharded weights (comma-separated)
python convert_wan_to_diffusers.py --output_path ./wan_out --model_type Wan2.2-T2V-A14B \
  --hf_repo some/repo \
  --hf_transformer_file transformer/model-00001.safetensors,transformer/model-00002.safetensors \
  --hf_transformer_2_file transformer_2/model.safetensors


# Download from HF and convert (high + low transformers)
python convert_wan_to_diffusers.py --output_path ./wan_out --model_type Wan2.2-T2V-A14B --hf_repo Wan-AI/Wan2.2-T2V-A14B

# Same with fp16 and a specific revision
python convert_wan_to_diffusers.py --output_path ./wan_out --model_type Wan2.2-T2V-A14B --hf_repo Wan-AI/Wan2.2-T2V-A14B --dtype fp16 --revision main

# Local path (unchanged; uses model_id from config for that model_type)
python convert_wan_to_diffusers.py --output_path ./wan_out --model_type Wan2.2-T2V-A14B --dtype fp16


 python3 upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_sndq_fp8_svd_64" --model-dir "/work_space/models/wan2.2_i2v_a14b_remix_nsfw_sdnq_fp8_svd_r64" --token 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


python3 convert_wan_lora_to_diffusers.py \
      --hf-repo lkzd7/WAN2.2_LoraSet_NSFW \
      --hf-files W22_Multiscene_Photoshoot_Softcore_i2v_HN.safetensors W22_Multiscene_Photoshoot_Softcore_i2v_LN.safetensors



python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t1-file DR34ML4Y_I2V_14B_HIGH_V2_diffusers.safetensors \
        --lora-t2-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t2-file DR34ML4Y_I2V_14B_LOW_V2_diffusers.safetensors \
        --lora-strength 0.8 

python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t1-file "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_high_noise_diffusers.safetensors" \
        --lora-t2-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t2-file "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_low_noise_diffusers.safetensors" \
        --lora-strength 0.8 


python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t1-file W22_Multiscene_Photoshoot_Softcore_i2v_HN_diffusers.safetensors \
        --lora-t2-repo rockapaper/wan22_nsfw_lora_diffusers --lora-t2-file W22_Multiscene_Photoshoot_Softcore_i2v_LN_diffusers.safetensors \
        --lora-strength 0.8 



python3 upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64" --model-dir "/work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64" --token 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


python3 convert_wan_lora_to_diffusers.py \
      --hf-repo lkzd7/WAN2.2_LoraSet_NSFW \
      --hf-files W22_Multiscene_Photoshoot_Softcore_i2v_HN.safetensors W22_Multiscene_Photoshoot_Softcore_i2v_LN.safetensors \
      --upload-repo rockapaper/wan22_nsfw_lora_diffusers \
      --token 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


python3 quantize_sdnq.py \
        --lora /work_space/models/loras/W22_Multiscene_Photoshoot_Softcore_i2v_HN_diffusers.safetensors \
        --model-dir /work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers

python3 quantize_sdnq.py \
    --lora /work_space/models/loras/W22_Multiscene_Photoshoot_Softcore_i2v_HN_diffusers.safetensors \
    --model-dir /work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers

# Civitai LoRA (use model version ID from the Civitai download URL)
python3 quantize_sdnq.py --model-dir /path/to/model --dtype fp8 \
    --lora-t1-civitai 123456 --lora-t2-civitai 789012 \
    --civitai-token YOUR_CIVITAI_API_KEY

# Or with env var
CIVITAI_API_KEY=xxx python3 quantize_sdnq.py --model-dir /path/to/model --dtype fp8 \
    --lora-t1-civitai 123456 --lora-t2-civitai 789012



# Civitai only
python3 convert_wan_lora_to_diffusers.py \
    --civitai-files 2073605 2083303 \
    --civitai-token a7818ca32e626842791a60c0e97fb3cd

FM BDY ENHCR : 2137972 = high ; 2137989 = low
NSFW Posing : 2391828 = high ; 2391829 = low
GENERAL NSFW : 2073605 = high ; 2083303 = low

# With env var
CIVITAI_API_KEY=xxx python3 convert_wan_lora_to_diffusers.py --civitai-files 123456    

# Civitai + HF (both sources)
python3 convert_wan_lora_to_diffusers.py \
    --civitai-files 123456 \
    --hf-repo profpeng/nsfwv2 \
    --hf-files DR34ML4Y_I2V_14B_HIGH_V2.safetensors




python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1 /work_space/models/loras/NSFW-22-H-e8_diffusers.safetensors \
        --lora-t2 /work_space/models/loras/NSFW-22-L-e8_diffusers.safetensors \
	--lora-strength 0.8 

python3 upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_GENERAL_NSFW_sdnq_fp8_svd_r64" --model-dir "/work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_GENERAL_NSFW_sdnq_fp8_svd_r64" --token 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


python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1 /work_space/models/loras/W22_NSFW_Posing_Nude_i2v_HN_v2_diffusers.safetensors \
        --lora-t2 /work_space/models/loras/W22_NSFW_Posing_Nude_i2v_LN_v2_diffusers.safetensors \
	--lora-strength 0.8 

python3 upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64" --model-dir "/work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64" --token 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


python3 quantize_sdnq.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1 /work_space/models/loras/Wan2.2-T2V-Femaled-Vaginus-High-v1_diffusers.safetensors \
        --lora-t2 /work_space/models/loras/Wan2.2-T2V-Femaled-Vaginus-High-v1_diffusers.safetensors \
	--lora-strength 0.8 

python3 upload_quantized.py --repo-id "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64" --model-dir "/work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64" --token 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



python3 convert_wan_lora_to_diffusers.py \
      --hf-repo vita-video-gen/svi-model \
      --hf-files version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors \
      --upload-repo rockapaper/_wan22_nsfw_lora_diffusers \
      --token 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


    python3 quantize_sdnq_multi.py --repo rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers --dtype fp8 --svd-rank 64 --svd-steps 16 \
        --lora-t1-repo "rockapaper/_wan22_nsfw_lora_diffusers:W22_Multiscene_Photoshoot_Softcore_i2v_HN_diffusers.safetensors:0.8" \
        --lora-t2-repo "rockapaper/_wan22_nsfw_lora_diffusers:W22_Multiscene_Photoshoot_Softcore_i2v_LN_diffusers.safetensors:0.8" \
	--lora-t1-repo "rockapaper/_wan22_nsfw_lora_diffusers:SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro_diffusers.safetensors:1.0" \
        --lora-t2-repo "rockapaper/_wan22_nsfw_lora_diffusers:SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro_diffusers.safetensors:1.0"



python diagnose_lora_keys.py \
  --lora /work_space/models/loras/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro_diffusers.safetensors \
         /work_space/models/loras/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro_diffusers.safetensors \
  --model-dir /work_space/models/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers \
  --dump


