# image_video_gen_edit_backend

Backend inference and worker code for image & video generation and editing. This repository contains controller tooling, worker implementations, model assets, and helper utilities used to run, orchestrate, and test inference workers that perform tasks such as image upscaling, video processing, and diffusion-based image generation/editing.

This README provides a high-level overview, repository layout, quickstart instructions, development notes, and pointers to where models and worker entrypoints live.

## Key features

- Multiple worker implementations for image/video tasks (Stable Diffusion, Real-ESRGAN, custom upscalers, etc.).
- Controller utilities for submitting and managing jobs.
- Docker-ready artifacts and environment files for reproducible deployments.
- A local `wheels/` folder with many Python wheel dependencies to support offline installs.

## Repository layout

- `backend/` - Main backend code and assets
	- `controller.db` - Lightweight DB used by controller tooling (checked in for convenience).
	- `Dockerfile` - Dockerfile for building an environment image (see folder for variants).
	- `quickpod_individual_worker/` - Per-worker model assets and specialized worker folders (Real-ESRGAN models, upscalers, SD variants, etc.).
		- `realesrgan_models/`, `realesrgan_models_full/` - Model weights for Real-ESRGAN-based upscaling workers.
	- `quickpod_worker/` - Python worker implementation used by pods
		- `requirements.txt` - Python dependencies for the worker
		- `config.env` - Example environment file consumed by worker entrypoints
		- `worker_entrypoint_warm_pod.sh` - An example entrypoint script used when launching a warmed worker pod
		- `src/` - Worker source code and implementation modules
	- `server_controller/` - Controller-side code, tooling and tests
		- `tools/` - Utilities for submitting jobs and local testing (e.g. `submit_job.py` is used in tooling)
		- `src/`, `test/`, `utilities/` - Controller sources and helpers
	- `wheels/` - Pre-downloaded wheel files for offline or reproducible pip installs

Files of interest (examples)
- `backend/quickpod_worker/requirements.txt` - install dependencies for the worker
- `backend/quickpod_worker/config.env` - example environment variables used by the worker
- `backend/quickpod_individual_worker/*` - contains model weights and worker-specific scripts
- `backend/server_controller/tools/` - scripts for submitting and controlling jobs locally

## Quickstart

Prerequisites

- Docker (if you plan to build the image)
- Python 3.10+ (for local development)
- Optional: GPU drivers and CUDA if running GPU inference

Run worker locally (developer flow)

1. Create a virtual environment and install dependencies for the quickpod worker:

	 - On Windows (PowerShell):

		 ```powershell
		 python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r backend/quickpod_worker/requirements.txt
		 ```

	 - If you prefer to use the offline wheels in `backend/wheels/`, install them instead of PyPI packages.

2. Prepare environment variables by copying `backend/quickpod_worker/config.env` into a local `.env` or exporting them in your shell.

3. Run a worker entrypoint script for local testing (example):

	 - The repository contains `backend/quickpod_worker/worker_entrypoint_warm_pod.sh` as a reference entrypoint used in pods. For local Python runs, invoke the worker module from `backend/quickpod_worker/src/` (the exact module name depends on the specific worker you want to run).

Controller / submitting jobs

- The controller tooling lives under `backend/server_controller/tools/`. There are example utilities (such as `submit_job.py`) used to submit test jobs to the controller. You can run controller scripts from that folder to create jobs against the local controller DB.

Docker image

- A `Dockerfile` is available at `backend/Dockerfile`; there may be variants (copy named differently). Use it to build a worker/controller image. Example build command:

	```powershell
	docker build -t image-video-backend -f backend/Dockerfile .
	```

## Adding or modifying workers

- Add a new worker directory under `backend/quickpod_individual_worker/` (or extend `backend/quickpod_worker/src/` for generic workers).
- Place model weights under the appropriate `realesrgan_models*` or worker-specific model folder.
- Update entrypoint scripts (see `worker_entrypoint_warm_pod.sh`) to accept environment variables and to perform any required model downloads or warm-up steps.

## Developer notes

- The repo includes a `controller.db` at `backend/controller.db` which the controller tooling uses for simple job storage. For production you may wish to swap this for a dedicated DB.
- Many heavy dependencies are present as wheels in `backend/wheels/` to simplify offline installs or deterministic builds.
- When testing GPU workloads, ensure your Docker runtime and host drivers match the CUDA/cudnn versions required by the wheels used.

## Troubleshooting

- If Python imports fail, double-check that you installed the matching wheel files (see `backend/wheels/`) or the correct versions from PyPI.
- If GPU initialization fails, confirm the host NVIDIA driver and CUDA toolkit are compatible with the wheels and your Docker runtime (nvidia-container-toolkit / NVIDIA Container Runtime).

## Contributing

Contributions are welcome. Please open issues or pull requests. When adding new workers, include a short README in the worker directory describing: expected inputs, outputs, environment variables, and any model weight or license requirements.

## License

This repository includes a `LICENSE` file at the project root. Refer to it for license terms.

## Contact / Maintainers

For questions about this repository, check the project issues or contact the maintainer listed in the repository metadata.

---

Last updated: see commit history.
