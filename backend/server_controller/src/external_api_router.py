import base64
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger("external_api_router")


def _deep_get(data: Dict[str, Any], path: str) -> Any:
    if not path:
        return None
    cur: Any = data
    token = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if token:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(token)
                token = ""
            i += 1
            continue
        if c == "[":
            if token:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(token)
                token = ""
            j = path.find("]", i)
            if j == -1:
                return None
            idx_str = path[i + 1 : j]
            if not isinstance(cur, list):
                return None
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            i = j + 1
            continue
        token += c
        i += 1
    if token:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(token)
    return cur


def _is_http_or_data(v: str) -> bool:
    v = v.strip().lower()
    return v.startswith("http://") or v.startswith("https://") or v.startswith("data:")


def _to_data_uri(v: str, mime: str = "image/png") -> str:
    if _is_http_or_data(v):
        return v
    return f"data:{mime};base64,{v}"


@dataclass
class ExternalExecutionResult:
    success: bool
    result_url: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    pending: bool = False
    external_task_id: Optional[str] = None
    result_bytes: Optional[bytes] = None
    result_text: Optional[str] = None
    result_content_type: Optional[str] = None


class ExternalApiRouter:
    def __init__(self, config_path: Optional[str] = None):
        default_path = str(Path(__file__).resolve().parent / "external_api_config.json")
        self._config_path = config_path or os.getenv(
            "EXTERNAL_API_CONFIG_PATH", default_path
        )
        self._config_mtime: Optional[float] = None
        self._config: Dict[str, Any] = {"defaults": {}, "models": {}}
        self._load_config(force=True)

    def _load_config(self, force: bool = False) -> None:
        try:
            mtime = os.path.getmtime(self._config_path)
            if not force and self._config_mtime == mtime:
                return
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            self._config_mtime = mtime
            logger.info(
                f"[EXTERNAL_API] Loaded config from {self._config_path} "
                f"for {len(self._config.get('models', {}))} models"
            )
        except FileNotFoundError:
            logger.warning(
                f"[EXTERNAL_API] Config file not found at {self._config_path}; "
                "external routing disabled"
            )
            self._config = {"defaults": {}, "models": {}}
            self._config_mtime = None
        except Exception as e:
            logger.error(f"[EXTERNAL_API] Failed to load config: {e}")
            self._config = {"defaults": {}, "models": {}}
            self._config_mtime = None

    def reload(self) -> None:
        self._load_config(force=True)

    def get_model_config(self, model: str) -> Optional[Dict[str, Any]]:
        self._load_config()
        return self._config.get("models", {}).get(model)

    def should_use_external(self, model: str, params: Optional[Dict[str, Any]]) -> bool:
        cfg = self.get_model_config(model)
        if not cfg or not cfg.get("provider"):
            return False
        policy = (cfg.get("route_policy") or "quickpod_only").lower()
        if policy == "external_only":
            return True
        if policy == "quickpod_only":
            return False

        params = params or {}
        # Explicit param overrides defaults
        if "use_external_api" in params:
            try:
                return bool(params.get("use_external_api"))
            except Exception:
                return False
        return bool(cfg.get("default_use_external", False))

    def execute(
        self, model: str, prompt: str, params: Dict[str, Any]
    ) -> ExternalExecutionResult:
        cfg = self.get_model_config(model)
        if not cfg:
            return ExternalExecutionResult(
                success=False, error=f"No external config for model: {model}"
            )

        try:
            args = self._build_arguments(cfg, prompt or "", params or {})
        except Exception as e:
            return ExternalExecutionResult(
                success=False, error=f"Failed to build external arguments: {e}"
            )

        provider = (cfg.get("provider", {}).get("type") or "").lower()
        if provider == "fal":
            return self._run_fal(cfg, args)
        if provider == "replicate":
            return self._run_replicate(cfg, args)
        if provider == "kie":
            return self._run_kie(cfg, args, prompt or "", params or {})
        if provider == "gemini":
            return self._run_gemini(cfg, args, params or {})
        return ExternalExecutionResult(
            success=False, error=f"Unsupported external provider type: {provider}"
        )

    def _build_arguments(
        self, cfg: Dict[str, Any], prompt: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        mapped: Dict[str, Any] = {}
        ctx = {"prompt": prompt, "params": params}

        param_map: Dict[str, Any] = cfg.get("param_map", {})
        for target, rule in param_map.items():
            value = None
            if isinstance(rule, dict):
                if "value" in rule:
                    value = rule["value"]
                else:
                    source = rule.get("source")
                    value = self._read_source_value(ctx, source)
                    value = self._apply_transform(value, rule.get("transform"))
            else:
                value = rule
            if value is not None:
                mapped[target] = value

        # Apply upscale remapping as "quality/resolution style" controls.
        upscale_map = cfg.get("upscale_map") or {}
        if upscale_map:
            requested = self._is_upscale_requested(params)
            block = upscale_map.get("when_true" if requested else "when_false") or {}
            set_map = block.get("set", {})
            for key, set_rule in set_map.items():
                if isinstance(set_rule, dict):
                    src = set_rule.get("from")
                    v = self._read_source_value(ctx, src)
                    v = self._apply_transform(v, set_rule.get("transform"))
                else:
                    v = set_rule
                if v is not None:
                    mapped[key] = v

        required = cfg.get("required_params", []) or []
        missing = [k for k in required if k not in mapped or mapped.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Missing required external params: {missing}")
        return mapped

    def _is_upscale_requested(self, params: Dict[str, Any]) -> bool:
        if bool(params.get("enable_upscaling")):
            return True
        if bool(params.get("upscale_output")):
            return True
        sf = params.get("scale_factor")
        try:
            return sf is not None and float(sf) > 1.0
        except Exception:
            return False

    def _read_source_value(self, ctx: Dict[str, Any], source: Optional[str]) -> Any:
        if not source:
            return None
        if source == "prompt":
            return ctx.get("prompt")
        if source.startswith("params."):
            return _deep_get({"params": ctx.get("params", {})}, source)
        return _deep_get(ctx, source)

    def _apply_transform(self, value: Any, transform: Optional[str]) -> Any:
        if transform is None or transform == "":
            return value
        out = value
        for t in str(transform).split("|"):
            t = t.strip()
            if not t:
                continue
            if t == "passthrough":
                continue
            if t == "string_trim":
                out = str(out).strip() if out is not None else out
                continue
            if t == "int_or_null":
                try:
                    out = int(float(out)) if out is not None and out != "" else None
                except Exception:
                    out = None
                continue
            if t.startswith("default:"):
                d = t.split(":", 1)[1]
                if out is None or out == "":
                    out = d
                continue
            if t.startswith("int_clamp:"):
                parts = t.split(":")
                if len(parts) == 3:
                    lo = int(parts[1])
                    hi = int(parts[2])
                    try:
                        v = int(float(out))
                        out = max(lo, min(hi, v))
                    except Exception:
                        out = None
                continue
            if t.startswith("multiply:"):
                mul = float(t.split(":", 1)[1])
                try:
                    out = float(out) * mul
                except Exception:
                    out = None
                continue
            if t == "media_input":
                if out is None:
                    continue
                out = _to_data_uri(str(out))
                continue
            if t == "http_passthrough":
                continue
            if t == "as_list":
                if out is None:
                    continue
                if not isinstance(out, list):
                    out = [out]
                continue
            # Unknown transform -> keep current value.
        return out

    def _timeout_config(self) -> Tuple[int, int]:
        defaults = self._config.get("defaults", {})
        timeout_ms = int(defaults.get("timeout_ms", 120000))
        poll_ms = int(defaults.get("poll_interval_ms", 1500))
        return timeout_ms, poll_ms

    def _extract_result_url(self, cfg: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
        response_map = cfg.get("response_map", {})
        path = response_map.get("result_url_path")
        if not path:
            return None
        v = _deep_get(payload, path)
        if isinstance(v, str):
            return v
        return None

    def _run_fal(self, cfg: Dict[str, Any], args: Dict[str, Any]) -> ExternalExecutionResult:
        provider = cfg.get("provider", {})
        model_id = provider.get("model_id")
        api_key = os.getenv(provider.get("api_key_env", "FAL_KEY"), "")
        if not model_id:
            return ExternalExecutionResult(success=False, error="FAL model_id not configured")
        if not api_key:
            return ExternalExecutionResult(success=False, error="FAL api key env missing")

        timeout_ms, poll_ms = self._timeout_config()
        headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
        base = f"https://queue.fal.run/{model_id}"
        try:
            submit = requests.post(base, headers=headers, json=args, timeout=30)
            submit.raise_for_status()
            submit_data = submit.json()
            req_id = submit_data.get("request_id") or submit_data.get("id")
            if not req_id:
                return ExternalExecutionResult(
                    success=False, error=f"FAL submit missing request_id: {submit_data}"
                )

            deadline = time.time() + (timeout_ms / 1000.0)
            while time.time() < deadline:
                status_resp = requests.get(
                    f"{base}/requests/{req_id}/status", headers=headers, timeout=20
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = str(status_data.get("status", "")).upper()
                if status in ("COMPLETED", "SUCCEEDED"):
                    result_resp = requests.get(
                        f"{base}/requests/{req_id}", headers=headers, timeout=30
                    )
                    result_resp.raise_for_status()
                    result_data = result_resp.json()
                    payload = result_data.get("response") or result_data.get("result") or result_data
                    url = self._extract_result_url(cfg, payload)
                    if not url and isinstance(payload, dict):
                        # Common fallback shapes
                        url = _deep_get(payload, "image.url") or _deep_get(payload, "images[0].url")
                    if not url:
                        return ExternalExecutionResult(
                            success=False,
                            error=f"FAL completed but no result URL found: {payload}",
                            raw=result_data,
                        )
                    return ExternalExecutionResult(success=True, result_url=url, raw=result_data)
                if status in ("FAILED", "ERROR", "CANCELLED"):
                    return ExternalExecutionResult(
                        success=False, error=f"FAL request failed: {status_data}", raw=status_data
                    )
                time.sleep(poll_ms / 1000.0)
            return ExternalExecutionResult(success=False, error="FAL request timed out")
        except Exception as e:
            return ExternalExecutionResult(success=False, error=f"FAL error: {e}")

    def _run_replicate(self, cfg: Dict[str, Any], args: Dict[str, Any]) -> ExternalExecutionResult:
        provider = cfg.get("provider", {})
        token = os.getenv(provider.get("api_key_env", "REPLICATE_API_TOKEN"), "")
        if not token:
            return ExternalExecutionResult(
                success=False, error="Replicate API token env missing"
            )
        headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        }
        timeout_ms, poll_ms = self._timeout_config()
        model = provider.get("model")
        version = provider.get("version")
        create_url = "https://api.replicate.com/v1/predictions"
        create_body: Dict[str, Any] = {"input": args}
        if version:
            create_body["version"] = version
        elif model:
            create_body["model"] = model
        else:
            return ExternalExecutionResult(
                success=False, error="Replicate model/version not configured"
            )

        try:
            create_resp = requests.post(create_url, headers=headers, json=create_body, timeout=30)
            create_resp.raise_for_status()
            pred = create_resp.json()
            pred_id = pred.get("id")
            get_url = _deep_get(pred, "urls.get") or (
                f"https://api.replicate.com/v1/predictions/{pred_id}" if pred_id else None
            )
            if not get_url:
                return ExternalExecutionResult(
                    success=False, error=f"Replicate missing prediction URL: {pred}"
                )

            deadline = time.time() + (timeout_ms / 1000.0)
            while time.time() < deadline:
                poll = requests.get(get_url, headers=headers, timeout=20)
                poll.raise_for_status()
                data = poll.json()
                status = (data.get("status") or "").lower()
                if status == "succeeded":
                    output = data.get("output")
                    url = None
                    if isinstance(output, str):
                        url = output
                    elif isinstance(output, list) and output:
                        if isinstance(output[0], str):
                            url = output[0]
                    elif isinstance(output, dict):
                        url = _deep_get(output, "url")
                    if not url:
                        payload = {"output": output}
                        url = self._extract_result_url(cfg, payload)
                    if not url:
                        return ExternalExecutionResult(
                            success=False,
                            error=f"Replicate succeeded but no result URL found: {data}",
                            raw=data,
                        )
                    return ExternalExecutionResult(success=True, result_url=url, raw=data)
                if status in ("failed", "canceled"):
                    return ExternalExecutionResult(
                        success=False, error=f"Replicate request failed: {data}", raw=data
                    )
                time.sleep(poll_ms / 1000.0)
            return ExternalExecutionResult(success=False, error="Replicate request timed out")
        except Exception as e:
            return ExternalExecutionResult(success=False, error=f"Replicate error: {e}")

    def _run_kie(
        self,
        cfg: Dict[str, Any],
        args: Dict[str, Any],
        prompt: str,
        params: Dict[str, Any],
    ) -> ExternalExecutionResult:
        import kie_client

        ok, err = kie_client.ensure_credits()
        if not ok:
            return ExternalExecutionResult(success=False, error=err)

        public = (os.getenv("CONTROLLER_PUBLIC_URL") or os.getenv("CONTROLLER_URL") or "").rstrip("/")
        if not public:
            return ExternalExecutionResult(
                success=False, error="CONTROLLER_PUBLIC_URL is required for Kie callBackUrl"
            )
        callback_url = f"{public}/kie/callback"

        input_payload = dict(args)
        recraft = bool(cfg.get("recraft_limits"))
        upload_fields = list(cfg.get("kie_upload_fields") or [])
        for key, value in input_payload.items():
            if key in upload_fields:
                continue
            values = value if isinstance(value, list) else [value]
            if any(isinstance(item, str) and kie_client.is_binary_object_url(item) for item in values):
                upload_fields.append(key)
        for field in upload_fields:
            value = input_payload.get(field)
            if value is None:
                continue
            try:
                if isinstance(value, list):
                    urls = []
                    for item in value:
                        src = str(item).strip()
                        if not src.startswith("http"):
                            continue
                        if recraft:
                            urls.append(kie_client.prepare_recraft_image(src))
                        else:
                            urls.append(kie_client.upload_file_from_url(src))
                    if urls:
                        input_payload[field] = urls
                    else:
                        input_payload.pop(field, None)
                else:
                    src = str(value).strip()
                    if not src.startswith("http"):
                        input_payload.pop(field, None)
                        continue
                    if recraft:
                        input_payload[field] = kie_client.prepare_recraft_image(src)
                    else:
                        input_payload[field] = kie_client.upload_file_from_url(src)
            except Exception as e:
                return ExternalExecutionResult(success=False, error=f"Kie upload failed: {e}")

        if cfg.get("force_resolution_1k") and "resolution" in (cfg.get("param_map") or {}):
            input_payload["resolution"] = "1K"
        elif cfg.get("force_resolution_1k"):
            # Only set if the Kie model documents a resolution field
            if cfg.get("has_resolution", True):
                input_payload["resolution"] = "1K"

        kie_model = cfg.get("provider", {}).get("model_id")
        if not kie_model:
            return ExternalExecutionResult(success=False, error="Kie model_id missing")
        try:
            task_id = kie_client.create_task(kie_model, input_payload, callback_url)
        except Exception as e:
            return ExternalExecutionResult(success=False, error=str(e))
        return ExternalExecutionResult(
            success=True,
            pending=True,
            external_task_id=task_id,
            raw={"taskId": task_id},
        )

    def _run_gemini(
        self,
        cfg: Dict[str, Any],
        args: Dict[str, Any],
        params: Dict[str, Any],
    ) -> ExternalExecutionResult:
        import gemini_ocr

        image_url = args.get("image") or args.get("image_url") or params.get("init_image")
        if not image_url:
            return ExternalExecutionResult(success=False, error="Gemini OCR needs an image URL")
        try:
            text = gemini_ocr.extract_text_from_image_url(str(image_url))
        except Exception as e:
            return ExternalExecutionResult(success=False, error=str(e))
        return ExternalExecutionResult(
            success=True,
            result_text=text,
            result_content_type="text/plain",
            raw={"text": text},
        )


