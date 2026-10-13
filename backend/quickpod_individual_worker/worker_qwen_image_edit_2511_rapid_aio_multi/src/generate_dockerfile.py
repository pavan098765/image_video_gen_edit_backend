#!/usr/bin/env python3
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
QPW_ROOT = os.path.dirname(PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "Dockerfile")

sys.path.insert(0, os.path.join(QPW_ROOT, "worker_multi_common", "src"))
import generate_dockerfile_snippets as dsl  # noqa: E402

MAX_ST = 850 * 1024 * 1024


def main():
    with open(OUT, "w", encoding="utf-8") as f:
        dsl.write_dockerfile_header(f)
        WORKER_DIR = os.path.basename(os.path.abspath(PROJECT_ROOT))
        dsl.copy_model_layers(f, PROJECT_ROOT, QPW_ROOT, MAX_ST)
        dsl.write_omni_code_copies(f, WORKER_DIR)
        f.write("\n")
        dsl.write_omni_run_install(f)
        f.write("\n")
        dsl.write_omni_pythonpath(f)
        dsl.write_omni_hf_env(f)
        dsl.write_flavor_env(f, "qwen")
        f.write("ENV AIO_VERSION=v19\n")
        f.write("\n")
        dsl.write_dockerfile_footer(f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
