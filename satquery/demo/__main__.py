"""
satquery.demo - CLI entrypoint
Run with:
    python -m satquery.demo                       # live mode (default)
    python -m satquery.demo --mode cached         # offline cached mode (stage-day fallback)
    python -m satquery.demo --mode cached --act 2 # single act in cached mode
    SATQUERY_DEMO_MODE=cached python -m satquery.demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FALLBACK_CACHE_DIR = Path(__file__).resolve().parents[2] / "demo" / "fallback_cache"

ACT_FILES = {
    1: "act1_single_image.json",
    2: "act2_bitemporal_change.json",
    3: "act3_multimodal_investigation.json",
}

DEMO_QUERIES = {
    1: "Describe the current land cover of this region from the Cartosat-3 scene.",
    2: "Has significant deforestation occurred in this region between January 2023 and January 2024?",
    3: "Compare optical and SAR analysis - is the flooding genuine or a pseudo-change artifact?",
}


def _print_header():
    print("\n" + "=" * 70)
    print("  SatQuery AI - Stage Demo Runner  (SIH26167 / Team SIH059)")
    print("=" * 70)


def _print_act_banner(act: int, mode: str):
    tags = {1: "Scene Description", 2: "Bi-temporal Change", 3: "Multi-modal Investigation"}
    print(f"\n{'-'*70}")
    print(f"  ACT {act} - {tags.get(act, 'Unknown')}  [{mode.upper()} MODE]")
    print(f"{'-'*70}")
    print(f"  QUERY: {DEMO_QUERIES[act]}")
    print()


def _load_cached_payload(act: int) -> dict:
    fpath = FALLBACK_CACHE_DIR / ACT_FILES[act]
    if not fpath.exists():
        print(f"  [ERROR] Cached payload not found: {fpath}", file=sys.stderr)
        sys.exit(1)
    with fpath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_result(payload: dict):
    status = payload.get("status", "unknown")
    conf = payload.get("confidence_score", 0.0)
    text = payload.get("text_response", payload.get("answer", ""))
    nums = payload.get("deterministic_numbers", payload.get("measurements", {}))
    breakdown = payload.get("confidence_breakdown", {})

    print(f"  STATUS    : {status}")
    print(f"  CONFIDENCE: {conf:.3f}  ({'HIGH' if conf >= 0.80 else 'MODERATE' if conf >= 0.60 else 'LOW'})")
    print()
    print("  ANSWER:")
    for line in text.split(". "):
        if line.strip():
            print(f"    . {line.strip().rstrip('.')}.")
    print()
    if nums:
        print("  KEY MEASUREMENTS:")
        for k, v in nums.items():
            if isinstance(v, list):
                print(f"    . {k}: {v}")
            else:
                unit = "ha" if "ha" in k else "m2" if "m2" in k else "%" if "pct" in k else ""
                print(f"    . {k}: {v} {unit}".rstrip())
    if breakdown:
        print()
        print("  CONFIDENCE BREAKDOWN:")
        for factor, score in breakdown.items():
            bar = "#" * int(float(score) * 20)
            print(f"    {factor:<25} {float(score):.2f}  {bar}")
    geojson = payload.get("geojson_geometry")
    if geojson:
        feature_count = len(geojson.get("features", [])) if geojson.get("type") == "FeatureCollection" else 1
        print()
        print(f"  GEOSPATIAL: {feature_count} feature(s) available for map overlay")
    print()


def _run_cached_mode(acts: list):
    print("\n  [CACHED MODE] Loading pre-computed results - no GPU required.")
    print(f"  Cache directory: {FALLBACK_CACHE_DIR}\n")
    manifest_path = FALLBACK_CACHE_DIR / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        print(f"  Manifest: generated_at={manifest.get('generated_at', 'N/A')}  "
              f"commit={str(manifest.get('commit_hash', 'N/A'))[:8]}")
    for act in acts:
        _print_act_banner(act, mode="cached")
        payload = _load_cached_payload(act)
        print("  [Loading cached result...]")
        time.sleep(0.4)
        _render_result(payload)
        time.sleep(0.3)
    print("=" * 70)
    print("  CACHED DEMO COMPLETE - all acts served from local fixtures.")
    print("=" * 70)


def _run_live_mode(acts: list):
    print("\n  [LIVE MODE] Initiating agent orchestrator...")
    try:
        import asyncio
        from satquery.agent.orchestrator import orchestrator as _orch

        async def _exec_act(act: int):
            asset_paths = {
                "optical_t1": f"demo/assets/optical_t1_act{act}.tif",
                "optical_t2": f"demo/assets/optical_t2_act{act}.tif",
            }
            if act == 3:
                asset_paths["sar"] = f"demo/assets/sar_act{act}.tif"
            return await _orch.run_query(
                query=DEMO_QUERIES[act],
                asset_paths=asset_paths,
                mission_id=f"DEMO-ACT{act}",
            )

        for act in acts:
            _print_act_banner(act, mode="live")
            print("  [Running agent orchestrator...]")
            try:
                result = asyncio.run(_exec_act(act))
                payload = {
                    "status": result.get("status"),
                    "text_response": result.get("answer", ""),
                    "confidence_score": result.get("composite_confidence", 0.0),
                    "confidence_breakdown": result.get("confidence_breakdown", {}),
                    "deterministic_numbers": {
                        m["metric_id"]: m["value"] for m in result.get("measurements", [])
                    },
                    "geojson_geometry": result.get("geojson_geometry"),
                }
                _render_result(payload)
            except Exception as exc:
                print(f"  [WARNING] Live execution failed: {exc}")
                print("  [FALLBACK] Switching to cached payload for this act...\n")
                payload = _load_cached_payload(act)
                _render_result(payload)
    except ImportError as e:
        print(f"  [ERROR] Cannot import orchestrator: {e}", file=sys.stderr)
        print("  TIP: Use --mode cached for offline demo.", file=sys.stderr)
        sys.exit(1)
    print("=" * 70)
    print("  LIVE DEMO COMPLETE.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m satquery.demo",
        description="SatQuery AI - 3-Act Stage Demo Runner (SIH26167)",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "cached"],
        default=os.environ.get("SATQUERY_DEMO_MODE", "live"),
        help="cached serves pre-computed fixtures (no GPU needed).",
    )
    parser.add_argument(
        "--act",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific act (1, 2, or 3). Default: all three.",
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List cached payload files and exit.",
    )
    args = parser.parse_args()

    if args.list_cache:
        print(f"Cached payloads in: {FALLBACK_CACHE_DIR}")
        for act, fname in ACT_FILES.items():
            fpath = FALLBACK_CACHE_DIR / fname
            status = "OK" if fpath.exists() else "MISSING"
            size = f"{fpath.stat().st_size} bytes" if fpath.exists() else ""
            print(f"  Act {act}: {fname}  {status}  {size}")
        sys.exit(0)

    acts = [args.act] if args.act else [1, 2, 3]
    _print_header()
    if args.mode == "cached":
        _run_cached_mode(acts)
    else:
        _run_live_mode(acts)


if __name__ == "__main__":
    main()
