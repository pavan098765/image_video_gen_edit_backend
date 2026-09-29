---
name: GLM shared bake + fleet fallback
overview: Single canonical plan (merged). Implemented in repo — GLM shared bake, OMNI_SHARED_BAKED_MODELS download exclusion, scheduler group + JSON fix, FLEET_OMNI_TEMPLATE_PRIORITY for shared-bake cold start.
todos:
  - id: bake-glm
    content: ensure_glm_ocr in omni_prep_bake; glm prep_model uses run_omni_full_bake only
    status: completed
  - id: inference-bake-verify
    content: generate_dockerfile_snippets.write_omni_code_copies includes worker_glm_ocr + other siblings
    status: completed
  - id: shared-exclude
    content: OMNI_SHARED_BAKED_MODELS + all 15 *_multi app.py
    status: completed
  - id: scheduler-glm-group
    content: STATIC_GROUP + enrich union; model_requirements glm_ocr -> image_aio_4x_utils
    status: completed
  - id: template-any-omni
    content: _resolve_any_omni_fleet_template + get_template_for_variant + snapshot + config.env
    status: completed
isProject: true
---

# GLM shared bake + omni shared models (canonical — implemented)

The duplicate `c:\Users\pavan\.cursor\plans\glm_shared_bake_+_fleet_fallback_285f98c1.plan.md` was removed; this file is the only plan.

## What shipped


| Area       | Change                                                                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Workers    | `OMNI_SHARED_BAKED_MODELS` in `omni_config.py`; all `worker_*_multi/src/app.py` use `BAKED_MODELS | OMNI_SHARED_BAKED_MODELS`                                                                                 |
| Bake       | `ensure_glm_ocr` in `omni_prep_bake.py`; `worker_glm_ocr_multi/prep_model.py` = full omni bake only (like AuraSR)                                                                                             |
| Controller | `glm_ocr` in `STATIC_GROUP_DEFINITIONS`; `FLEET_OMNI_SHARED_BAKE_MODELS`; `_resolve_any_omni_fleet_template()` + `get_template_for_variant` logic; enrich unions `supported_models` with static group members |
| JSON       | `model_requirements.json` `glm_ocr` → `image_aio_4x_utils` + full member list                                                                                                                                 |
| Config     | `config.env` comments for `FLEET_OMNI_TEMPLATE_PRIORITY`                                                                                                                                                      |


**Ops:** Rebuild all omni multi Docker images after `prep_model` so GLM weights exist on disk. Set at least one primary `*_MULTI_TEMPLATE_UUID` and optionally `FLEET_OMNI_TEMPLATE_PRIORITY` for cold-start ordering.