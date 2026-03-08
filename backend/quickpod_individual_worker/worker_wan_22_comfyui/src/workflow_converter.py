#!/usr/bin/env python3
"""
workflow_converter.py
=====================

Converts a ComfyUI **visual-format** workflow JSON (nodes/links arrays)
into the **API format** (dict keyed by string node-IDs with class_type + inputs)
that ComfyUI's ``POST /prompt`` endpoint expects.

Visual format structure
-----------------------
- ``nodes``: list of node objects, each with ``id``, ``type``, ``inputs``,
  ``outputs``, ``widgets_values``, ``mode``, ``properties``, ...
- ``links``: list of ``[link_id, from_node, from_slot, to_node, to_slot, type]``

API format structure
--------------------
- Dict keyed by ``str(node_id)``
- Each value: ``{"class_type": "...", "inputs": {...}}``
  - Connected inputs:  ``[str(source_node_id), source_output_index]``
  - Widget inputs:     plain value (str, int, float, bool, ...)

Virtual nodes (resolved, NOT emitted)
--------------------------------------
- ``Reroute``: visual pass-through, resolved to direct connection
- ``SetNode`` / ``GetNode``: named data pipes (KJNodes), resolved to
  direct connections from the SetNode's source
- ``Note``: purely decorative, skipped
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Virtual node types that are resolved instead of emitted
# ---------------------------------------------------------------------------

VIRTUAL_NODE_TYPES: Set[str] = {"Reroute", "SetNode", "GetNode", "Note"}

# Some workflows save Reroute (or similar pass-through) nodes with an internal
# UUID instead of "Reroute". These are typically ComfyUI built-in pass-through
# nodes that got exported with an internal type ID. Treat them as Reroute:
# resolve through them and do not emit into the API prompt.
REROUTE_UUID_TYPES: Set[str] = {
    "340f324c-f8b3-46e1-a35b-4561ecd7027c",
    "a7d2f0d7-ff69-400d-aa64-ca94fcff6246",
    "c7b2a9fc-a78e-4638-ab98-6ee92ecc3067",
}

# ---------------------------------------------------------------------------
# Filename corrections
# ---------------------------------------------------------------------------
# The workflow JSON may reference filenames that differ from the actual files
# on disk (e.g. the workflow author renamed a model locally).  Map the
# workflow value -> actual filename.

_FILENAME_CORRECTIONS: Dict[str, str] = {
    "svi-film-opt-10212025.safetensors": "svi-film-opt-10212025_lora_rank_128_fp16.safetensors",
    "wan_2.1_vae.safetensors": "Wan2_1_VAE_bf16.safetensors",
    "umt5_xxl_fp16.safetensors": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_visual_to_api(visual_workflow: dict) -> dict:
    """
    Convert a full visual-format workflow dict to the API prompt dict.

    Parameters
    ----------
    visual_workflow : dict
        The parsed JSON of a ``.json`` workflow saved from ComfyUI's UI
        (contains ``nodes``, ``links``, ``groups``, etc.).

    Returns
    -------
    dict
        API-format dict suitable for ``POST /prompt {"prompt": <this>}``.
    """
    nodes: List[dict] = visual_workflow.get("nodes", [])
    links: List[list] = visual_workflow.get("links", [])

    # 1) Build link map: link_id -> (source_node_id, source_output_slot)
    link_map: Dict[int, Tuple[int, int]] = {}
    for link in links:
        # link = [link_id, from_node, from_slot, to_node, to_slot, type_str]
        link_id = link[0]
        from_node = link[1]
        from_slot = link[2]
        link_map[link_id] = (from_node, from_slot)

    # 2) Index nodes by ID
    nodes_by_id: Dict[int, dict] = {n["id"]: n for n in nodes}

    # 3) Build SetNode name -> source mapping
    #    SetNode stores data: widgets_values[0] = name, inputs[0].link = source
    set_node_sources: Dict[str, Tuple[int, int]] = {}
    for node in nodes:
        if node.get("type") != "SetNode":
            continue
        if node.get("mode") == 2:  # bypassed
            continue
        wv = node.get("widgets_values", [])
        name = wv[0] if wv else None
        if not name:
            continue
        node_inputs = node.get("inputs", [])
        if node_inputs:
            input_link = node_inputs[0].get("link")
            if input_link is not None and input_link in link_map:
                set_node_sources[name] = link_map[input_link]

    # 4) Resolution function: given a (node_id, slot), follow through
    #    Reroute / GetNode / SetNode chains to reach the real source.
    _resolve_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def resolve_source(node_id: int, slot: int, visited: Optional[set] = None) -> Tuple[int, int]:
        """Recursively resolve virtual nodes to find the real source."""
        key = (node_id, slot)
        if key in _resolve_cache:
            return _resolve_cache[key]

        if visited is None:
            visited = set()
        if key in visited:
            # Circular reference - return as-is
            return key
        visited.add(key)

        node = nodes_by_id.get(node_id)
        if node is None:
            return key

        node_type = node.get("type", "")

        if node_type == "Reroute" or node_type in REROUTE_UUID_TYPES:
            # Follow the reroute's input to find the real source
            node_inputs = node.get("inputs", [])
            if node_inputs:
                input_link = node_inputs[0].get("link")
                if input_link is not None and input_link in link_map:
                    src_id, src_slot = link_map[input_link]
                    result = resolve_source(src_id, src_slot, visited)
                    _resolve_cache[key] = result
                    return result
            return key

        elif node_type == "GetNode":
            # Look up the matching SetNode by name
            wv = node.get("widgets_values", [])
            name = wv[0] if wv else None
            if name and name in set_node_sources:
                src_id, src_slot = set_node_sources[name]
                result = resolve_source(src_id, src_slot, visited)
                _resolve_cache[key] = result
                return result
            return key

        elif node_type == "SetNode":
            # Follow the SetNode's input
            node_inputs = node.get("inputs", [])
            if node_inputs:
                input_link = node_inputs[0].get("link")
                if input_link is not None and input_link in link_map:
                    src_id, src_slot = link_map[input_link]
                    result = resolve_source(src_id, src_slot, visited)
                    _resolve_cache[key] = result
                    return result
            return key

        else:
            # Real node - this is the final source
            _resolve_cache[key] = key
            return key

    # 5) Convert each active, non-virtual node
    api_prompt: Dict[str, dict] = {}

    for node in nodes:
        # Skip disabled nodes (mode == 2 means "bypassed / muted")
        if node.get("mode") == 2:
            continue

        node_id = node["id"]
        node_type = node["type"]

        # Skip virtual / decorative nodes - they are resolved, not emitted
        if node_type in VIRTUAL_NODE_TYPES or node_type in REROUTE_UUID_TYPES:
            continue

        inputs_dict: Dict[str, Any] = {}

        # -- Resolve connection-based inputs --
        node_inputs: List[dict] = node.get("inputs", [])
        # Track which widget-capable inputs are connected (so we skip their
        # widget_value entry later).
        connected_widget_names: set = set()

        for inp in node_inputs:
            inp_name: str = inp["name"]
            link_id = inp.get("link")

            if link_id is not None and link_id in link_map:
                # Connected -> resolve through virtual nodes to real source
                raw_src_id, raw_src_slot = link_map[link_id]
                real_src_id, real_src_slot = resolve_source(raw_src_id, raw_src_slot)
                inputs_dict[inp_name] = [str(real_src_id), real_src_slot]

                # If this input is also a widget, mark it as connected
                widget_info = inp.get("widget")
                if widget_info:
                    widget_name = widget_info.get("name", inp_name) if isinstance(widget_info, dict) else inp_name
                    connected_widget_names.add(widget_name)
            # else: not connected, will be handled by widgets_values below

        # -- Map widgets_values to named inputs --
        widgets_values = node.get("widgets_values", [])

        if isinstance(widgets_values, dict):
            # Some newer nodes store widgets as a named dict directly
            for k, v in widgets_values.items():
                if k not in inputs_dict:  # Don't overwrite connected inputs
                    inputs_dict[k] = v
        elif isinstance(widgets_values, list) and widgets_values:
            # Positional widget values: iterate through inputs that have
            # a ``widget`` property, in declaration order, assigning values
            # from widgets_values sequentially.
            #
            # IMPORTANT: ComfyUI's visual format includes *hidden* widget
            # values in ``widgets_values`` that don't appear in the ``inputs``
            # array.  The most common one is ``control_after_generate`` which
            # appears immediately after every ``seed`` / ``noise_seed`` INT
            # widget.  We must detect and skip these to keep the positional
            # mapping aligned.
            CONTROL_AFTER_GENERATE_VALUES = frozenset({
                "fixed", "increment", "decrement", "randomize",
            })

            widget_inputs = [
                inp for inp in node_inputs
                if inp.get("widget") is not None
            ]

            wi = 0  # index into widgets_values
            for w_inp in widget_inputs:
                if wi >= len(widgets_values):
                    break
                widget_info = w_inp.get("widget", {})
                widget_name = (
                    widget_info.get("name", w_inp["name"])
                    if isinstance(widget_info, dict)
                    else w_inp["name"]
                )

                if widget_name in connected_widget_names:
                    # This widget input is connected -> skip its value in
                    # widgets_values but still consume the slot
                    wi += 1
                    # Also skip hidden control_after_generate if present
                    if _is_seed_widget(w_inp) and wi < len(widgets_values):
                        next_val = widgets_values[wi]
                        if isinstance(next_val, str) and next_val in CONTROL_AFTER_GENERATE_VALUES:
                            wi += 1
                    continue

                inputs_dict[widget_name] = widgets_values[wi]
                wi += 1

                # After a seed widget, skip the hidden control_after_generate
                if _is_seed_widget(w_inp) and wi < len(widgets_values):
                    next_val = widgets_values[wi]
                    if isinstance(next_val, str) and next_val in CONTROL_AFTER_GENERATE_VALUES:
                        wi += 1  # skip hidden widget value

        # Normalize Windows paths in string values (backslash -> forward slash,
        # fix truncated extensions like .safetensor -> .safetensors)
        inputs_dict = _normalize_paths(inputs_dict)

        api_prompt[str(node_id)] = {
            "class_type": node_type,
            "inputs": inputs_dict,
        }

    return api_prompt


def _normalize_paths(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix Windows-style paths in string values.

    Workflows saved on Windows embed backslash paths (e.g.
    ``Wan\\model.safetensor``) which fail on Linux ComfyUI servers.

    Fixes:
    - Backslash ``\\`` -> forward slash ``/``
    - Truncated ``.safetensor`` -> ``.safetensors``
    - Known filename mismatches (see ``_FILENAME_CORRECTIONS``)
    """
    for key, val in inputs.items():
        if not isinstance(val, str):
            continue
        # Backslash -> forward slash
        if "\\" in val:
            val = val.replace("\\", "/")
            inputs[key] = val
        # Fix truncated .safetensor extension (missing trailing 's')
        if val.endswith(".safetensor"):
            val = val + "s"
            inputs[key] = val
        # Apply known filename corrections (check basename)
        basename = val.rsplit("/", 1)[-1] if "/" in val else val
        if basename in _FILENAME_CORRECTIONS:
            corrected = _FILENAME_CORRECTIONS[basename]
            val = val.replace(basename, corrected)
            inputs[key] = val
    return inputs


def _is_seed_widget(inp: dict) -> bool:
    """
    Check if an input is a seed-type widget (which has a hidden
    ``control_after_generate`` widget following it in widgets_values).
    """
    name = inp.get("name", "").lower()
    # Common seed input names in ComfyUI nodes
    return name in ("seed", "noise_seed")


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


def load_and_convert(workflow_path: str | Path) -> dict:
    """
    Load a visual workflow JSON file and return the API-format dict.
    """
    path = Path(workflow_path)
    with open(path, "r", encoding="utf-8") as f:
        visual = json.load(f)
    return convert_visual_to_api(visual)


def save_api_workflow(api_workflow: dict, output_path: str | Path) -> None:
    """
    Save an API-format workflow to a JSON file for debugging / inspection.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(api_workflow, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow_converter.py <visual_workflow.json> [output_api.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".json", "_api.json")

    api = load_and_convert(in_path)
    save_api_workflow(api, out_path)
    print(f"Converted {len(api)} nodes -> {out_path}")
