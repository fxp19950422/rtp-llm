"""Independent DeepSeek-V4 Flash checkpoint inventory oracle.

This module deliberately does not import the RTP-LLM model descriptor.  The
expected source names are derived from the Hugging Face config/index contract,
so descriptor omissions and duplicate consumption cannot make the oracle pass.
Only safetensors headers and bounded byte slices are read; tensor payloads are
never materialized.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


MAIN_GLOBAL_TARGETS = {
    "embed.weight": "embedding",
    "norm.weight": "final_ln_gamma",
    "head.weight": "lm_head",
    "hc_head_base": "v4.hc.head_base",
    "hc_head_fn": "v4.hc.head_fn",
    "hc_head_scale": "v4.hc.head_scale",
}

MTP_GLOBAL_TARGETS = {
    "mtp.0.norm.weight": "final_ln_gamma",
    "mtp.0.hc_head_base": "v4.hc.head_base",
    "mtp.0.hc_head_fn": "v4.hc.head_fn",
    "mtp.0.hc_head_scale": "v4.hc.head_scale",
    "mtp.0.enorm.weight": "v4.mtp.enorm",
    "mtp.0.hnorm.weight": "v4.mtp.hnorm",
    "mtp.0.e_proj.weight": "v4.mtp.e_proj.weight",
    "mtp.0.e_proj.scale": "v4.mtp.e_proj.scale",
    "mtp.0.h_proj.weight": "v4.mtp.h_proj.weight",
    "mtp.0.h_proj.scale": "v4.mtp.h_proj.scale",
}


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    filename: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]
    slice_sha256: str


def _add(mapping: Dict[str, str], name: str, target: str) -> None:
    if name in mapping:
        raise ValueError(f"duplicate source mapping: {name}")
    mapping[name] = target


def _layer_mapping(
    prefix: str,
    *,
    ratio: int,
    hash_router: bool,
    num_experts: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for suffix, target in (
        ("attn_norm.weight", "v4.attn_norm"),
        ("attn.q_norm.weight", "v4.attn.q_norm"),
        ("attn.kv_norm.weight", "v4.attn.kv_norm"),
        ("attn.attn_sink", "v4.attn.attn_sink"),
    ):
        _add(out, f"{prefix}.{suffix}", target)

    for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        for kind in ("weight", "scale"):
            _add(
                out,
                f"{prefix}.attn.{projection}.{kind}",
                f"v4.attn.{projection}.{kind}",
            )

    if ratio in (4, 128):
        for kind in ("wkv.weight", "wgate.weight", "norm.weight", "ape"):
            _add(
                out,
                f"{prefix}.attn.compressor.{kind}",
                f"v4.compressor.{kind}",
            )

    if ratio == 4:
        for kind in ("wq_b.weight", "wq_b.scale", "weights_proj.weight"):
            _add(
                out,
                f"{prefix}.attn.indexer.{kind}",
                f"v4.indexer.{kind}",
            )
        for kind in ("wkv.weight", "wgate.weight", "norm.weight", "ape"):
            _add(
                out,
                f"{prefix}.attn.indexer.compressor.{kind}",
                f"v4.indexer.compressor.{kind}",
            )

    for block in ("attn", "ffn"):
        for kind in ("base", "fn", "scale"):
            _add(out, f"{prefix}.hc_{block}_{kind}", f"v4.hc.{block}.{kind}")

    _add(out, f"{prefix}.ffn_norm.weight", "v4.ffn_norm")
    _add(out, f"{prefix}.ffn.gate.weight", "v4.router.weight")
    router_kind = "tid2eid" if hash_router else "bias"
    _add(out, f"{prefix}.ffn.gate.{router_kind}", f"v4.router.{router_kind}")

    for projection in ("w1", "w2", "w3"):
        merged_projection = "w13" if projection in ("w1", "w3") else "w2"
        merged_slot = ":0" if projection == "w1" else ":1" if projection == "w3" else ""
        for kind in ("weight", "scale"):
            _add(
                out,
                f"{prefix}.ffn.shared_experts.{projection}.{kind}",
                f"v4.shared.{merged_projection}.{kind}{merged_slot}",
            )

    for expert_id in range(num_experts):
        for projection in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                _add(
                    out,
                    f"{prefix}.ffn.experts.{expert_id}.{projection}.{kind}",
                    f"v4.routed.{projection}.{kind}[{expert_id}]",
                )
    return out


def expected_checkpoint_mapping(config: Mapping[str, object]) -> Dict[str, str]:
    """Return every expected checkpoint source and its unique logical slot."""
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    num_hash_layers = int(config["num_hash_layers"])
    ratios = [int(value) for value in config["compress_ratios"]]
    if len(ratios) < num_layers:
        raise ValueError(
            f"compress_ratios has {len(ratios)} entries for {num_layers} layers"
        )

    out = dict(MAIN_GLOBAL_TARGETS)
    for layer_id in range(num_layers):
        layer = _layer_mapping(
            f"layers.{layer_id}",
            ratio=ratios[layer_id],
            hash_router=layer_id < num_hash_layers,
            num_experts=num_experts,
        )
        for source, target in layer.items():
            _add(out, source, f"layers.{layer_id}.{target}")

    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    if num_mtp_layers not in (0, 1):
        raise ValueError(f"only zero or one MTP layer is supported, got {num_mtp_layers}")
    if num_mtp_layers:
        mtp = _layer_mapping(
            "mtp.0", ratio=0, hash_router=False, num_experts=num_experts
        )
        for source, target in mtp.items():
            _add(out, source, f"mtp.0.{target}")
        for source, target in MTP_GLOBAL_TARGETS.items():
            _add(out, source, f"mtp.0.global.{target}")
    return out


def validate_inventory(
    config: Mapping[str, object], index: Mapping[str, object]
) -> Dict[str, str]:
    expected = expected_checkpoint_mapping(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index has no weight_map object")
    actual = set(weight_map)
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        raise ValueError(
            "checkpoint inventory mismatch: "
            f"missing={missing[:20]} ({len(missing)} total), "
            f"unexpected={unexpected[:20]} ({len(unexpected)} total)"
        )
    if len(expected) != len(actual):
        raise ValueError(
            f"checkpoint source mapping is not one-to-one: {len(expected)} != {len(actual)}"
        )
    return expected


def expert_owners(num_experts: int, ep_size: int) -> List[Tuple[int, ...]]:
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError(
            f"num_experts must be divisible by positive ep_size: {num_experts}, {ep_size}"
        )
    per_rank = num_experts // ep_size
    return [
        tuple(range(rank * per_rank, (rank + 1) * per_rank))
        for rank in range(ep_size)
    ]


def _read_header(path: Path) -> Tuple[int, Mapping[str, object]]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    return 8 + header_length, header


def inspect_tensor(
    checkpoint: Path,
    index: Mapping[str, object],
    name: str,
    slice_bytes: int = 4096,
) -> TensorMetadata:
    filename = str(index["weight_map"][name])
    path = checkpoint / filename
    data_start, header = _read_header(path)
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"{name} is indexed in {filename} but absent from its header")
    begin, end = (int(value) for value in entry["data_offsets"])
    length = min(max(0, end - begin), slice_bytes)
    with path.open("rb") as stream:
        stream.seek(data_start + begin)
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError(f"truncated tensor payload for {name}")
    return TensorMetadata(
        name=name,
        filename=filename,
        dtype=str(entry["dtype"]),
        shape=tuple(int(value) for value in entry["shape"]),
        data_offsets=(begin, end),
        slice_sha256=hashlib.sha256(payload).hexdigest(),
    )


def tensor_category(name: str) -> str:
    if ".ffn.experts." in name:
        return "routed_fp4_scale" if name.endswith(".scale") else "routed_fp4_weight"
    if ".ffn.shared_experts." in name:
        return "shared_fp8"
    if name.startswith("mtp."):
        return "mtp"
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn.compressor." in name:
        return "compressor"
    if any(part in name for part in (".wq_a.", ".wq_b.", ".wkv.", ".wo_a.", ".wo_b.")):
        return "attention_projection"
    if ".ffn.gate." in name:
        return "router"
    if ".hc_" in name or name.startswith("hc_head_"):
        return "mhc"
    return "global_or_norm"


def deterministic_samples(names: Iterable[str], seed: int = 20260821) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        grouped[tensor_category(name)].append(name)
    rng = random.Random(seed)
    chosen = {rng.choice(sorted(values)) for values in grouped.values()}
    for expert_id in (0, 31, 32, 255):
        chosen.add(f"layers.0.ffn.experts.{expert_id}.w1.weight")
        chosen.add(f"layers.0.ffn.experts.{expert_id}.w1.scale")
    chosen.update(
        {
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w1.scale",
            "mtp.0.e_proj.weight",
            "mtp.0.ffn.experts.255.w2.scale",
        }
    )
    return sorted(chosen)


def _validate_routed_fp4(metadata: TensorMetadata, config: Mapping[str, object]) -> None:
    hidden = int(config["hidden_size"])
    inter = int(config["moe_intermediate_size"])
    projection = metadata.name.rsplit(".", 2)[-2]
    is_scale = metadata.name.endswith(".scale")
    if projection in ("w1", "w3"):
        expected_shape = (inter, hidden // (32 if is_scale else 2))
    elif projection == "w2":
        expected_shape = (hidden, inter // (32 if is_scale else 2))
    else:
        raise ValueError(f"unknown routed projection: {metadata.name}")
    expected_dtype = "F8_E8M0" if is_scale else "I8"
    if metadata.shape != expected_shape or metadata.dtype != expected_dtype:
        raise ValueError(
            f"FP4 raw contract mismatch for {metadata.name}: "
            f"got {metadata.dtype}{metadata.shape}, expected "
            f"{expected_dtype}{expected_shape}"
        )


def validate_quantized_headers(
    checkpoint: Path,
    config: Mapping[str, object],
    index: Mapping[str, object],
) -> Mapping[str, int]:
    """Validate every FP4 expert and every dense/shared FP8 weight header."""
    names_by_file: Dict[str, List[str]] = defaultdict(list)
    for name, filename in index["weight_map"].items():
        names_by_file[str(filename)].append(str(name))

    counts = Counter()
    for filename, names in names_by_file.items():
        _, header = _read_header(checkpoint / filename)
        for name in names:
            entry = header.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"{name} is absent from indexed shard {filename}")
            metadata = TensorMetadata(
                name=name,
                filename=filename,
                dtype=str(entry["dtype"]),
                shape=tuple(int(value) for value in entry["shape"]),
                data_offsets=tuple(int(value) for value in entry["data_offsets"]),
                slice_sha256="",
            )
            if ".ffn.experts." in name:
                _validate_routed_fp4(metadata, config)
                counts["routed_fp4"] += 1
                continue

            quantized_prefix = any(
                marker in name
                for marker in (
                    ".attn.wq_a.",
                    ".attn.wq_b.",
                    ".attn.wkv.",
                    ".attn.wo_a.",
                    ".attn.wo_b.",
                    ".attn.indexer.wq_b.",
                    ".ffn.shared_experts.",
                )
            ) or name in {
                "mtp.0.e_proj.weight",
                "mtp.0.e_proj.scale",
                "mtp.0.h_proj.weight",
                "mtp.0.h_proj.scale",
            }
            if quantized_prefix:
                expected_dtype = "F8_E8M0" if name.endswith(".scale") else "F8_E4M3"
                if metadata.dtype != expected_dtype:
                    raise ValueError(
                        f"FP8 raw contract mismatch for {name}: "
                        f"got {metadata.dtype}, expected {expected_dtype}"
                    )
                counts["dense_or_shared_fp8"] += 1
    return dict(counts)


def current_rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def repeated_plan_rss_growth(
    config: Mapping[str, object], *, iterations: int = 8
) -> int:
    expected_checkpoint_mapping(config)
    gc.collect()
    before = current_rss_bytes()
    for _ in range(iterations):
        plan = expected_checkpoint_mapping(config)
        if not plan:
            raise AssertionError("empty load plan")
        del plan
        gc.collect()
    return max(0, current_rss_bytes() - before)


def run(checkpoint: Path, *, max_rss_growth_mb: int = 32) -> Mapping[str, object]:
    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    mapping = validate_inventory(config, index)
    quantized_header_counts = validate_quantized_headers(checkpoint, config, index)
    samples = [inspect_tensor(checkpoint, index, name) for name in deterministic_samples(mapping)]
    for metadata in samples:
        if ".ffn.experts." in metadata.name:
            _validate_routed_fp4(metadata, config)
    owners = {ep: expert_owners(int(config["n_routed_experts"]), ep) for ep in (1, 2, 4, 8)}
    rss_growth = repeated_plan_rss_growth(config)
    limit = max_rss_growth_mb * 1024 * 1024
    if rss_growth > limit:
        raise ValueError(
            f"repeated load-plan RSS growth {rss_growth} exceeds {limit} bytes"
        )
    return {
        "tensor_count": len(mapping),
        "total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "category_counts": dict(sorted(Counter(tensor_category(name) for name in mapping).items())),
        "ep_local_expert_counts": {str(ep): len(ranks[0]) for ep, ranks in owners.items()},
        "rss_growth_bytes": rss_growth,
        "quantized_header_counts": quantized_header_counts,
        "samples": [metadata.__dict__ for metadata in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-rss-growth-mb", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.checkpoint, max_rss_growth_mb=args.max_rss_growth_mb),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
