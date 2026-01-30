POD_STRATEGY="machine_based"
QUICKPOD_EMAIL="pavanalur3@gmail.com"
QUICKPOD_PASSWORD="C@9rw6fxATDm8sM"

SD15_TEMPLATE_UUID="41df2876-3dcf-4820-b2de-cd4cd3d4cb26"
SD35_TEMPLATE_UUID="1e734d51-6496-4e8d-9c4f-54346e8c0175"
SDXL_TEMPLATE_UUID="950771a9-5d7e-40f6-9705-d2f8e2076629"
DIFFUSERS_OUTPAINT_TEMPLATE_UUID="da3f806a-5052-4aaf-96b0-0dd4dbcf72bd"

QWEN_IMAGE_EDIT_TEMPLATE_UUID="38a17e41-29de-412c-9bf2-f649d76cf941"
QWEN_IMAGE_EDIT_2509_TEMPLATE_UUID="2c3597b1-e70a-44cd-8d68-ba7f8ecf035e"
FLUX_KONTEXT_DEV_TEMPLATE_UUID="a543beaf-e954-4758-ba80-8da80f103f3e"
Z_IMAGE_TURBO_TEMPLATE_UUID="c120cb9f-14ea-4f7d-af55-b71314064465"

# INT4 variant templates for fallback to 4090 GPUs (required for variant fallback)
# Create these templates in QuickPod and add UUIDs here
Z_IMAGE_TURBO_INT4_TEMPLATE_UUID="73c54872-3f89-4fe6-8afd-d8a348d0848d"
QWEN_IMAGE_EDIT_INT4_TEMPLATE_UUID="490ed475-9301-4a1d-952f-ece79023f412"
QWEN_IMAGE_EDIT_2509_INT4_TEMPLATE_UUID="e5f9c9f6-27ce-4984-8854-7e4ce38cb0bf"
FLUX_KONTEXT_DEV_INT4_TEMPLATE_UUID="f0c11fe0-b7b1-4bcc-a3dd-a56a00fb8303"
QWEN_IMAGE_INT4_TEMPLATE_UUID="0d67bd11-9d14-48e2-9387-34e8be22c4db"

FASTWAN_21_1B_TEMPLATE_UUID="666e5614-72cf-44ef-b6d5-212b3374cb47"
FASTWAN_22_5B_TEMPLATE_UUID="d33b0a6e-126e-4843-85ce-7110292db51f"
AURASRV2_TEMPLATE_UUID="b2c939c2-774d-4cb4-8965-39707bdb90f9"
REALESRGAN_TEMPLATE_UUID="329b0dd7-2006-4756-92ef-0f8fe65d415d"
SWIN2SR_TEMPLATE_UUID="de5ee65e-0294-4ce6-8a20-fadcfbfcd24f"
BASIC_UTILS_TEMPLATE_UUID="0045f811-b69f-4ae3-bc5a-96af28f7ff4a"
WARM_POD_TEMPLATE_UUID="a2ad9208-7047-4f24-9a53-9ddf4c8d2b04"
DEFAULT_TEMPLATE_UUID="a2ad9208-7047-4f24-9a53-9ddf4c8d2b04"

JOB_AUTO_RECOVER="false"
IMAGE_PREFIX="image-vide-generator-api"
WORKER_IDLE_TIMEOUT="20"
WORKER_REGISTRATION_TIMEOUT="250"
QUEUE_SIZE_THRESHOLD="5"
MAX_INDIVIDUAL_PODS="50"
CONTROLLER_PUBLIC_URL="https://genedit.mooo.com"
CONTROLLER_SHARED_SECRET="SECRETKEY"

# FIXED: Removed /bucket-name from endpoint - bucket is specified separately
R2_ENDPOINT="https://9ad1ad9de5a81cf253aa25267a464c3f.r2.cloudflarestorage.com"
R2_BUCKET="image-vide-generator-api"
R2_ACCESS_KEY_ID="0dbb2d0d9763966579e0a72fc57c1e83"
R2_SECRET_ACCESS_KEY="f129df93f725ad08619e5ca4100febb59d218921972bc12e8e237e40b19ebde1"
R2_TOKEN="HEIU2jmxr7UQazVyriqN-6cMP9w1mwO4-Ed3MPHh"

# Google Play
GOOGLE_PLAY_PACKAGE_NAME="com.shambhutech.kalaai"
# GOOGLE_PLAY_SERVICE_ACCOUNT_PATH="/app/secrets/google-play-service-account.json
GOOGLE_PLAY_SERVICE_ACCOUNT_PATH="/home/ubuntu/python_projects/img_gen_orchestrator/server_controller/src/serviceAccountKey.json"

# RECOMMENDED: Use R2 Custom Domain or Public R2.dev URL for better compatibility
# Option 1: Public R2.dev URL (easiest, works everywhere)
# R2_PUBLIC_URL="https://pub-<your-subdomain>.r2.dev
# 
# Option 2: Custom domain (best for production)
# R2_CUSTOM_DOMAIN="https://cdn.yourdomain.com
#
# If set, presigned URLs will use this domain instead of the default R2 endpoint
CONTROLLER_API_KEYS="SECRETKEY"
DEVICE_SIGNING_SECRET="SECRETKEY"
DEVICE_TIMESTAMP_WINDOW="300"
DB_PATH="/home/ubuntu/python_projects/img_gen_orchestrator/server_controller/controller.db"
GOOGLE_APPLICATION_CREDENTIALS="/home/ubuntu/python_projects/img_gen_orchestrator/server_controller/src/serviceAccountKey.json"

# ="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="
# Cloudflare D1 Database Configuration
# ="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="="
# When these are set, the server uses Cloudflare D1 instead of local SQLite.
# Leave empty/unset to use local SQLite database.

# Cloudflare Account ID (found in Cloudflare dashboard URL)
CF_ACCOUNT_ID="9ad1ad9de5a81cf253aa25267a464c3f"

# D1 Database ID (found in D1 dashboard after creating database)
CF_D1_DATABASE_ID="f42c2b67-6457-4fd3-be85-298c9574342d"

# Cloudflare API Token with D1 permissions
# Create at: https://dash.cloudflare.com/profile/api-tokens
# Required permissions: Account > D1 > Edit
CF_API_TOKEN="HBJyhVVL4mR979hpv5QV9MKyB1tV0wCYKQU7HisA"

# Set to 'true' to force D1 usage (error if D1 config incomplete)
# Set to 'false' or leave empty for auto-detect (use D1 if configured, else SQLite)
USE_D1_DATABASE="false"

ADGEM_POSTBACK_KEY="4jg79i9fa9lg74g9lcj336g1"
ADGEM_WEBHOOK_SECRET="07gdnhc57j961mkl5k4c6g9c"
