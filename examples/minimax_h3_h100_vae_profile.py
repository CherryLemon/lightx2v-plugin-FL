"""Reproduce the opt-in H100 MiniMax-H3 VAE profile on a single GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

os.environ["LIGHTX2V_FL_GLOBAL_GEMS"] = "0"

import torch
import flag_gems
from loguru import logger
from lightx2v import LightX2VPipeline
from lightx2v_fl.h3_vae_profile import H3VAEGemsProfile, VAE_EXCLUSIONS
from lightx2v_fl.ops import attn, norm

BASE_EXCLUSIONS = [
    "mm", "mm_out", "rms_norm", "conv1d", "conv_transpose1d",
    "index_select", "mul", "add", "neg", "cat", "silu", "to_copy", "copy_",
]
PROMPT = (
    "integrated_multimodal_description: A cinematic fox walks through a snowy "
    "pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and "
    "distant birds. non_diegetic_music: Quiet warm strings."
)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        with torch._C._AutoDispatchBelowAutograd():
            return value.detach().to(device="cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/MiniMax-H3")
    parser.add_argument("--config", default="/work/h3-flagos.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--validate-current", action="store_true")
    parser.add_argument("--paired-control", action="store_true",
                        help="interleave current/candidate measured requests")
    args = parser.parse_args()
    if args.warmups < 1 or args.runs < 1:
        parser.error("warmups and runs must be positive")
    torch.set_grad_enabled(False)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    pipe = LightX2VPipeline(
        task="t2av", model_path=args.model_path,
        model_cls="minimax_h3", model_variant="fl2av")
    pipe.create_generator(config_json=args.config)
    results = {
        "base_exclusions": BASE_EXCLUSIONS,
        "vae_extra_exclusions": list(VAE_EXCLUSIONS),
        "preserved_gems_module": "video_vae.decoder.proj_in",
        "observations": [], "audits": {},
    }
    stages, payload = [], {}
    capture = False
    candidate = True
    rms_stats = {"success": 0, "error": 0}
    original_rms = norm._GEMS_RMS

    def count_rms(*a, **kw):
        try:
            value = original_rms(*a, **kw)
        except Exception:
            rms_stats["error"] += 1
            raise
        rms_stats["success"] += 1
        return value

    norm._GEMS_RMS = count_rms
    base_keys = None
    removed_keys = {
        entry[0] for entry in flag_gems._FULL_CONFIG
        if entry[1].__name__ in VAE_EXCLUSIONS
    }
    for name in ("video_vae", "audio_vae"):
        owner = getattr(pipe.runner, name)
        original = owner.decode

        def decode(*a, _name=name, _fn=original, **kw):
            actual = set(flag_gems.all_registered_keys())
            expected = base_keys - removed_keys if candidate else base_keys
            assert actual == expected, (_name, sorted(actual ^ expected))
            results["audits"]["candidate" if candidate else "current"] = {
                "entries": len(actual),
                "removed_vs_current": sorted(base_keys - actual),
            }
            if capture:
                payload[_name + "_input"] = snapshot((a, kw))
            value = _fn(*a, **kw)
            if capture:
                payload[_name + "_output"] = snapshot(value)
            return value
        owner.decode = decode

    def sink(message):
        text = message.record["message"]
        match = re.search(r" - (.+) cost ([\d.]+) seconds", text)
        if text.startswith("[Profile]") and match:
            stages.append({"name": match[1], "seconds": float(match[2])})

    logger.add(sink, level="INFO")

    def run(label, measured=False, save_tensors=False, use_candidate=True):
        nonlocal capture, candidate
        capture, candidate = save_tensors, use_candidate
        payload.clear()
        stages.clear()
        rms_before, attn_before = dict(rms_stats), dict(attn._ATTN_STATS)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        path = out / (label + ".mp4")
        pipe.generate(task="t2av", seed=args.seed, prompt=args.prompt,
                      save_result_path=str(path))
        torch.cuda.synchronize()
        row = {
            "label": label, "measured": measured, "capture": capture,
            "profile": "candidate" if candidate else "current",
            "seconds": time.perf_counter() - start, "stages": list(stages),
            "rms_success": rms_stats["success"] - rms_before["success"],
            "rms_errors": rms_stats["error"] - rms_before["error"],
            "attention_gems": attn._ATTN_STATS["gems"] - attn_before["gems"],
            "attention_fallback": attn._ATTN_STATS["fallback"] - attn_before["fallback"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        assert row["rms_success"] > 0
        assert row["rms_errors"] == row["attention_fallback"] == 0
        assert path.is_file()
        if capture:
            torch.save(payload, out / (label + ".pt"))
        results["observations"].append(row)
        (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        print("H3_VAE_PROFILE_RESULT", json.dumps(row), flush=True)

    try:
        with H3VAEGemsProfile(pipe.runner, base_exclusions=BASE_EXCLUSIONS) as profile:
            base_keys = set(flag_gems.all_registered_keys())
            for index in range(args.warmups):
                if args.paired_control:
                    with profile.current_only():
                        run("warmup-current-" + str(index + 1), use_candidate=False)
                run("warmup-" + str(index + 1))
            for index in range(args.runs):
                if args.paired_control and index % 2 == 0:
                    with profile.current_only():
                        run("current-" + str(index + 1), measured=True, use_candidate=False)
                run("candidate-" + str(index + 1), measured=True)
                if args.paired_control and index % 2 == 1:
                    with profile.current_only():
                        run("current-" + str(index + 1), measured=True, use_candidate=False)
            if args.validate_current:
                with profile.current_only():
                    run("current-validation", save_tensors=True, use_candidate=False)
                run("candidate-validation", save_tensors=True)
                with profile.current_only():
                    run("current-repeat", save_tensors=True, use_candidate=False)
            measured = [row["seconds"] for row in results["observations"]
                        if row["measured"] and row["profile"] == "candidate"]
            results["mean_seconds"] = statistics.mean(measured)
            results["median_seconds"] = statistics.median(measured)
            control = [row["seconds"] for row in results["observations"]
                       if row["measured"] and row["profile"] == "current"]
            if control:
                results["current_mean_seconds"] = statistics.mean(control)
                results["matched_reduction_pct"] = (
                    1 - results["mean_seconds"] / results["current_mean_seconds"]) * 100
            results["media_sha256"] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in out.glob("*.mp4")
            }
            (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        norm._GEMS_RMS = original_rms
        pipe.runner.close()


if __name__ == "__main__":
    main()
