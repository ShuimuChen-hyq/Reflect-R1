#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


RELATIVE_VIDEO_ROOTS = ("short/", "long/", "video_r1/", "panda/")


def _is_jsonl(path):
    return str(path).endswith(".jsonl")


def _json_dump(obj, fp):
    json.dump(obj, fp, ensure_ascii=False)


def _normalize_prefix(prefix):
    if not prefix:
        return None
    return prefix if prefix.endswith("/") else prefix + "/"


def _init_stats():
    return {
        "records": 0,
        "short": 0,
        "long": 0,
        "video_r1": 0,
        "panda": 0,
        "absolute": 0,
        "missing_video_path": 0,
        "unknown": 0,
    }


def _classify_path(path):
    if isinstance(path, str):
        if path.startswith("short/"):
            return "short"
        if path.startswith("long/"):
            return "long"
        if path.startswith("video_r1/"):
            return "video_r1"
        if path.startswith("panda/"):
            return "panda"
        if os.path.isabs(path):
            return "absolute"
    return "unknown"


def _update_stats(stats, path):
    kind = _classify_path(path)
    stats[kind] += 1


def _transform_video_path(path, args):
    if not isinstance(path, str):
        raise ValueError(f"video_path must be a string, got {type(path).__name__}")

    if args.mode == "strip":
        if path.startswith(RELATIVE_VIDEO_ROOTS):
            return path

        short_prefix = _normalize_prefix(args.short_prefix or args.video_r1_prefix)
        long_prefix = _normalize_prefix(args.long_prefix or args.panda_prefix)
        if short_prefix and path.startswith(short_prefix):
            return "short/" + path[len(short_prefix):]
        if long_prefix and path.startswith(long_prefix):
            return "long/" + path[len(long_prefix):]
        if os.path.isabs(path):
            raise ValueError(f"unknown absolute video_path prefix: {path}")
        raise ValueError(f"unknown relative video_path root: {path}")

    if args.mode == "localize":
        if path.startswith("short/"):
            root = args.short_dir or args.video_r1_dir
            if not root:
                raise ValueError("SHORT_VIDEO_DIR or --short-dir is required for short/ paths")
            return os.path.join(root, path[len("short/"):])
        if path.startswith("long/"):
            root = args.long_dir or args.panda_dir
            if not root:
                raise ValueError("LONG_VIDEO_DIR or --long-dir is required for long/ paths")
            return os.path.join(root, path[len("long/"):])
        if path.startswith("video_r1/"):
            root = args.video_r1_dir or args.short_dir
            if not root:
                raise ValueError("VIDEO_R1_DIR or --video-r1-dir is required for video_r1/ paths")
            return os.path.join(root, path[len("video_r1/"):])
        if path.startswith("panda/"):
            root = args.panda_dir or args.long_dir
            if not root:
                raise ValueError("PANDA_DIR or --panda-dir is required for panda/ paths")
            return os.path.join(root, path[len("panda/"):])
        raise ValueError(f"expected a video_path under short/ or long/, got: {path}")

    raise ValueError(f"unsupported transform mode: {args.mode}")


def _walk_and_transform(obj, args, stats):
    if isinstance(obj, dict):
        out = {}
        saw_video_path = False
        for key, value in obj.items():
            if key == "video_path":
                saw_video_path = True
                value = _transform_video_path(value, args)
                _update_stats(stats, value)
            else:
                value = _walk_and_transform(value, args, stats)
            out[key] = value
        if not saw_video_path and "messages" in obj:
            stats["missing_video_path"] += 1
        return out
    if isinstance(obj, list):
        return [_walk_and_transform(item, args, stats) for item in obj]
    return obj


def _walk_and_validate(obj, stats, must_exist=False):
    if isinstance(obj, dict):
        if "video_path" in obj:
            path = obj["video_path"]
            _update_stats(stats, path)
            if must_exist and (not isinstance(path, str) or not os.path.exists(path)):
                raise FileNotFoundError(f"missing video_path target: {path}")
        elif "messages" in obj:
            stats["missing_video_path"] += 1
        for value in obj.values():
            _walk_and_validate(value, stats, must_exist=must_exist)
    elif isinstance(obj, list):
        for item in obj:
            _walk_and_validate(item, stats, must_exist=must_exist)


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc


def _write_stats(stats):
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


def transform_file(args):
    stats = _init_stats()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_jsonl(input_path):
        with open(output_path, "w", encoding="utf-8") as out_fp:
            for record in _iter_jsonl(input_path):
                stats["records"] += 1
                transformed = _walk_and_transform(record, args, stats)
                _json_dump(transformed, out_fp)
                out_fp.write("\n")
    else:
        with open(input_path, "r", encoding="utf-8") as in_fp:
            data = json.load(in_fp)
        records = data if isinstance(data, list) else [data]
        stats["records"] = len(records)
        transformed = _walk_and_transform(data, args, stats)
        with open(output_path, "w", encoding="utf-8") as out_fp:
            _json_dump(transformed, out_fp)
            out_fp.write("\n")

    _write_stats(stats)


def sample_file(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.limit < 1:
        raise ValueError("--limit must be positive")

    if _is_jsonl(input_path):
        count = 0
        with open(output_path, "w", encoding="utf-8") as out_fp:
            for record in _iter_jsonl(input_path):
                _json_dump(record, out_fp)
                out_fp.write("\n")
                count += 1
                if count >= args.limit:
                    break
        print(json.dumps({"records": count}, sort_keys=True))
        return

    with open(input_path, "r", encoding="utf-8") as in_fp:
        data = json.load(in_fp)
    if isinstance(data, list):
        sampled = data[: args.limit]
    else:
        sampled = data
    with open(output_path, "w", encoding="utf-8") as out_fp:
        _json_dump(sampled, out_fp)
        out_fp.write("\n")
    print(json.dumps({"records": len(sampled) if isinstance(sampled, list) else 1}, sort_keys=True))


def validate_file(args):
    stats = _init_stats()
    input_path = Path(args.input)
    if _is_jsonl(input_path):
        for record in _iter_jsonl(input_path):
            stats["records"] += 1
            _walk_and_validate(record, stats, must_exist=args.must_exist)
    else:
        with open(input_path, "r", encoding="utf-8") as in_fp:
            data = json.load(in_fp)
        records = data if isinstance(data, list) else [data]
        stats["records"] = len(records)
        _walk_and_validate(data, stats, must_exist=args.must_exist)
    _write_stats(stats)


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare Reflect-R1 public training JSONs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    strip = subparsers.add_parser("strip", help="Replace source video roots with relative paths.")
    strip.add_argument("--input", required=True)
    strip.add_argument("--output", required=True)
    strip.add_argument("--short-prefix", default=os.environ.get("SHORT_SOURCE_PREFIX"))
    strip.add_argument("--long-prefix", default=os.environ.get("LONG_SOURCE_PREFIX"))
    strip.add_argument("--video-r1-prefix", default=os.environ.get("VIDEO_R1_SOURCE_PREFIX"))
    strip.add_argument("--panda-prefix", default=os.environ.get("PANDA_SOURCE_PREFIX"))
    strip.set_defaults(func=transform_file, mode="strip")

    localize = subparsers.add_parser("localize", help="Replace relative paths with local video roots.")
    localize.add_argument("--input", required=True)
    localize.add_argument("--output", required=True)
    localize.add_argument("--short-dir", default=os.environ.get("SHORT_VIDEO_DIR"))
    localize.add_argument("--long-dir", default=os.environ.get("LONG_VIDEO_DIR"))
    localize.add_argument("--video-r1-dir", default=os.environ.get("VIDEO_R1_DIR"))
    localize.add_argument("--panda-dir", default=os.environ.get("PANDA_DIR"))
    localize.set_defaults(func=transform_file, mode="localize")

    sample = subparsers.add_parser("sample", help="Write the first N records.")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--limit", type=int, required=True)
    sample.set_defaults(func=sample_file)

    validate = subparsers.add_parser("validate", help="Validate and count video_path fields.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--must-exist", action="store_true")
    validate.set_defaults(func=validate_file)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
