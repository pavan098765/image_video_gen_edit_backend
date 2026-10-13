#!/usr/bin/env python3
"""Flatten ComfyUI subgraph instances and inject Use-Everywhere links."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

SUBGRAPH_INPUT_ID = -10
SUBGRAPH_OUTPUT_ID = -20

WIDGET_LINK_TYPES = frozenset({"COMBO", "INT", "FLOAT", "STRING", "BOOLEAN"})


def _link_as_list(link: Any) -> Optional[list]:
    if isinstance(link, list) and len(link) >= 6:
        return [
            int(link[0]),
            int(link[1]),
            int(link[2]),
            int(link[3]),
            int(link[4]),
            link[5],
        ]
    if isinstance(link, dict):
        return [
            int(link.get("id", 0)),
            int(link.get("origin_id", link.get("from_node", 0))),
            int(link.get("origin_slot", link.get("from_slot", 0))),
            int(link.get("target_id", link.get("to_node", 0))),
            int(link.get("target_slot", link.get("to_slot", 0))),
            link.get("type", "*"),
        ]
    return None


def normalize_links(links: Any) -> List[list]:
    out: List[list] = []
    if not links:
        return out
    for link in links:
        parsed = _link_as_list(link)
        if parsed:
            out.append(parsed)
    return out


def _max_ids(visual: dict) -> Tuple[int, int]:
    max_node = 0
    for node in visual.get("nodes") or []:
        try:
            max_node = max(max_node, int(node.get("id") or 0))
        except (TypeError, ValueError):
            continue
    max_link = 0
    for link in normalize_links(visual.get("links")):
        max_link = max(max_link, int(link[0]))
    extra = visual.get("extra") or {}
    for ue in extra.get("ue_links") or []:
        if not isinstance(ue, dict):
            continue
        try:
            max_node = max(max_node, int(ue.get("upstream") or 0), int(ue.get("downstream") or 0))
        except (TypeError, ValueError):
            pass
    return max_node, max_link


def _set_input_link(node: dict, slot: int, link_id: Optional[int]) -> None:
    inputs = node.get("inputs") or []
    if 0 <= slot < len(inputs) and isinstance(inputs[slot], dict):
        inputs[slot]["link"] = link_id


def _set_output_links(node: dict, slot: int, link_id: int) -> None:
    outputs = node.get("outputs") or []
    if 0 <= slot < len(outputs) and isinstance(outputs[slot], dict):
        existing = outputs[slot].get("links") or []
        if not isinstance(existing, list):
            existing = []
        if link_id not in existing:
            existing.append(link_id)
        outputs[slot]["links"] = existing


def _apply_widget_value(node: dict, slot: int, value: Any) -> None:
    wv = node.get("widgets_values")
    if isinstance(wv, list):
        while len(wv) <= slot:
            wv.append(None)
        wv[slot] = value
        node["widgets_values"] = wv
        return
    inputs = node.get("inputs") or []
    name = None
    if 0 <= slot < len(inputs) and isinstance(inputs[slot], dict):
        name = inputs[slot].get("name")
    if isinstance(wv, dict) and name:
        wv[name] = value
        node["widgets_values"] = wv


def expand_one(
    visual: dict,
    instance: dict,
    subgraph: dict,
    next_node_id: int,
    next_link_id: int,
) -> Tuple[int, int, Dict[int, int], Dict[int, Tuple[int, int]]]:
    """Replace one subgraph instance with cloned inner nodes. Returns id maps."""
    instance_id = int(instance["id"])
    sg_name = subgraph.get("name") or subgraph.get("id")
    inner_nodes = copy.deepcopy(subgraph.get("nodes") or [])
    inner_links = subgraph.get("links") or []

    id_map: Dict[int, int] = {}
    cloned: List[dict] = []
    for node in inner_nodes:
        old_id = int(node["id"])
        new_id = next_node_id
        next_node_id += 1
        id_map[old_id] = new_id
        node["id"] = new_id
        if instance.get("mode") in (2, 4):
            node["mode"] = instance.get("mode")
        props = dict(node.get("properties") or {})
        props["_h3_subgraph"] = sg_name
        props["_h3_instance_id"] = instance_id
        props["_h3_inner_id"] = old_id
        node["properties"] = props
        cloned.append(node)

    inner_by_old = {int(n.get("_orig", 0)): n for n in cloned}  # unused
    inner_by_new = {int(n["id"]): n for n in cloned}
    old_to_cloned = {old: inner_by_new[new] for old, new in id_map.items()}

    output_map: Dict[int, Tuple[int, int]] = {}
    widget_slots: List[Tuple[int, dict, int]] = []

    root_links = normalize_links(visual.get("links"))
    incoming: Dict[int, list] = {}
    outgoing: Dict[int, List[list]] = {}
    remaining_links: List[list] = []
    for link in root_links:
        _lid, src, sslot, dst, dslot, ltype = link
        if dst == instance_id:
            incoming.setdefault(int(dslot), []).append(link)
            continue
        if src == instance_id:
            outgoing.setdefault(int(sslot), []).append(link)
            continue
        remaining_links.append(link)

    new_links: List[list] = []

    for raw in inner_links:
        parsed = _link_as_list(raw)
        if not parsed:
            continue
        _lid, origin_id, origin_slot, target_id, target_slot, ltype = parsed

        if origin_id == SUBGRAPH_INPUT_ID and target_id == SUBGRAPH_OUTPUT_ID:
            continue

        if origin_id == SUBGRAPH_INPUT_ID:
            outer_list = incoming.get(int(origin_slot), [])
            inner_target = id_map.get(int(target_id))
            if inner_target is None:
                continue
            if outer_list:
                for outer in outer_list:
                    nid = next_link_id
                    next_link_id += 1
                    new_links.append(
                        [nid, outer[1], outer[2], inner_target, int(target_slot), outer[5] or ltype]
                    )
                    _set_input_link(old_to_cloned[int(target_id)], int(target_slot), nid)
                    src_nodes = {int(n["id"]): n for n in visual.get("nodes") or []}
                    src_node = src_nodes.get(int(outer[1]))
                    if src_node:
                        _set_output_links(src_node, int(outer[2]), nid)
            elif str(ltype).upper() in WIDGET_LINK_TYPES or str(ltype) in WIDGET_LINK_TYPES:
                widget_slots.append((int(origin_slot), old_to_cloned[int(target_id)], int(target_slot)))
            continue

        if target_id == SUBGRAPH_OUTPUT_ID:
            inner_src = id_map.get(int(origin_id))
            if inner_src is None:
                continue
            output_map[int(target_slot)] = (inner_src, int(origin_slot))
            for outer in outgoing.get(int(target_slot), []):
                nid = next_link_id
                next_link_id += 1
                new_links.append(
                    [nid, inner_src, int(origin_slot), outer[3], outer[4], outer[5] or ltype]
                )
                dst_nodes = {int(n["id"]): n for n in visual.get("nodes") or []}
                dst_node = dst_nodes.get(int(outer[3]))
                if dst_node:
                    _set_input_link(dst_node, int(outer[4]), nid)
                _set_output_links(old_to_cloned[int(origin_id)], int(origin_slot), nid)
            continue

        inner_src = id_map.get(int(origin_id))
        inner_dst = id_map.get(int(target_id))
        if inner_src is None or inner_dst is None:
            continue
        nid = next_link_id
        next_link_id += 1
        new_links.append([nid, inner_src, int(origin_slot), inner_dst, int(target_slot), ltype])
        _set_input_link(old_to_cloned[int(target_id)], int(target_slot), nid)
        _set_output_links(old_to_cloned[int(origin_id)], int(origin_slot), nid)

    wv = instance.get("widgets_values")
    if isinstance(wv, list) and widget_slots:
        widget_slots.sort(key=lambda x: x[0])
        for value, (_slot, node, tslot) in zip(wv, widget_slots):
            _apply_widget_value(node, tslot, value)

    visual["nodes"] = [n for n in visual.get("nodes") or [] if int(n.get("id")) != instance_id]
    visual["nodes"].extend(cloned)
    visual["links"] = remaining_links + new_links
    return next_node_id, next_link_id, id_map, output_map


def _remap_ue_links(visual: dict, instance_id: int, output_map: Dict[int, Tuple[int, int]]) -> None:
    extra = visual.setdefault("extra", {})
    ue_links = extra.get("ue_links") or []
    remapped = []
    for ue in ue_links:
        if not isinstance(ue, dict):
            continue
        item = dict(ue)
        try:
            up = int(item.get("upstream"))
            down = int(item.get("downstream"))
        except (TypeError, ValueError):
            remapped.append(item)
            continue
        if up == instance_id:
            slot = int(item.get("upstream_slot") or 0)
            if slot in output_map:
                new_id, new_slot = output_map[slot]
                item["upstream"] = str(new_id)
                item["upstream_slot"] = new_slot
        if down == instance_id:
            # Downstream into a subgraph instance: leave; expand_one already
            # rewired graph links. Skip dangling UE into removed instance.
            remapped.append(item)
            continue
        remapped.append(item)
    extra["ue_links"] = remapped


def inject_ue_links(visual: dict) -> dict:
    """Turn extra.ue_links into real links so API convert can follow them."""
    extra = visual.get("extra") or {}
    ue_links = extra.get("ue_links") or []
    if not ue_links:
        return visual
    links = normalize_links(visual.get("links"))
    next_link = max((int(l[0]) for l in links), default=0) + 1
    nodes_by_id = {int(n["id"]): n for n in visual.get("nodes") or [] if n.get("id") is not None}
    existing = {(int(l[1]), int(l[2]), int(l[3]), int(l[4])) for l in links}
    for ue in ue_links:
        if not isinstance(ue, dict):
            continue
        try:
            src = int(ue.get("upstream"))
            sslot = int(ue.get("upstream_slot") or 0)
            dst = int(ue.get("downstream"))
            dslot = int(ue.get("downstream_slot") or 0)
        except (TypeError, ValueError):
            continue
        key = (src, sslot, dst, dslot)
        if key in existing:
            continue
        if src not in nodes_by_id or dst not in nodes_by_id:
            continue
        nid = next_link
        next_link += 1
        ltype = ue.get("type") or "*"
        links.append([nid, src, sslot, dst, dslot, ltype])
        existing.add(key)
        _set_input_link(nodes_by_id[dst], dslot, nid)
        _set_output_links(nodes_by_id[src], sslot, nid)
    visual["links"] = links
    return visual


def flatten_subgraphs(visual: dict) -> dict:
    """Expand every subgraph instance; inject UE links; return a flat visual graph."""
    visual = copy.deepcopy(visual)
    definitions = visual.get("definitions") or {}
    subgraphs = {sg["id"]: sg for sg in definitions.get("subgraphs") or [] if sg.get("id")}
    next_node, next_link = _max_ids(visual)
    next_node += 1
    next_link += 1

    safety = 0
    while safety < 32:
        safety += 1
        instance = next(
            (n for n in visual.get("nodes") or [] if n.get("type") in subgraphs),
            None,
        )
        if instance is None:
            break
        sg = subgraphs[instance["type"]]
        next_node, next_link, _id_map, output_map = expand_one(
            visual, instance, sg, next_node, next_link
        )
        _remap_ue_links(visual, int(instance["id"]), output_map)

    visual["links"] = normalize_links(visual.get("links"))
    inject_ue_links(visual)
    return visual


def collect_class_types(visual: dict) -> set:
    types = set()
    for node in visual.get("nodes") or []:
        t = node.get("type")
        if t:
            types.add(str(t))
    for sg in (visual.get("definitions") or {}).get("subgraphs") or []:
        for node in sg.get("nodes") or []:
            t = node.get("type")
            if t:
                types.add(str(t))
    return types
