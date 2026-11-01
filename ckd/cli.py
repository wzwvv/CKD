# Author: Ziwei Wang

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from .data import DATASETS, load_dataset, split_loso
from .models import CKDModel, ModelConfig, build_modules
from .training import evaluate_model, initialize_adaptation, make_loader, seed_everything, train_student, train_teacher


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def _load(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    return value


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _model_config(dataset, args):
    return ModelConfig(channels=int(dataset.x.shape[1]), samples=int(dataset.x.shape[2]),
                       sampling_rate=int(dataset.sampling_rate), f1=args.f1, d=args.d, f2=args.f2)


def _teacher_path(root, scenario, source, target, seed):
    dataset = source if scenario == "single" else target
    return root / "teachers" / scenario / dataset / f"seed_{seed}" / "teacher.pt"


def _teacher_state(args, scenario, source, target, seed):
    replacements = {"source": source or "all", "target": target, "seed": seed}
    if args.teacher_encoder_checkpoint or args.teacher_classifier_checkpoint:
        if not (args.teacher_encoder_checkpoint and args.teacher_classifier_checkpoint) or args.teacher_checkpoint:
            raise ValueError("Supply both encoder/classifier checkpoints, or one teacher checkpoint.")
        return {
            "encoder": _load(Path(args.teacher_encoder_checkpoint.format(**replacements))),
            "classifier": _load(Path(args.teacher_classifier_checkpoint.format(**replacements))),
        }
    path = Path(args.teacher_checkpoint.format(**replacements)) if args.teacher_checkpoint else _teacher_path(args.output_root, scenario, source, target, seed)
    state = _load(path)
    if "source_datasets" in state and target in state["source_datasets"]:
        raise ValueError("The target dataset must not be included in teacher pretraining.")
    if "scenario" in state and state["scenario"] != scenario:
        raise ValueError("Teacher checkpoint scenario does not match the requested scenario.")
    if "source_datasets" in state:
        expected = [source] if scenario == "single" else [name for name in DATASETS if name != target]
        if set(state["source_datasets"]) != set(expected):
            raise ValueError("Teacher source datasets do not match this transfer task.")
    if "alignment" in state and state["alignment"] != args.align:
        raise ValueError("Teacher checkpoint and target data must use the same alignment setting.")
    if "model_config" in state:
        for name in ("samples", "sampling_rate", "f1", "d", "f2", "num_classes"):
            expected = {"samples": 1000, "sampling_rate": 250, "f1": args.f1, "d": args.d, "f2": args.f2, "num_classes": 2}[name]
            if state["model_config"][name] != expected:
                raise ValueError(f"Teacher model setting does not match: {name}")
    return state


def _parser(stage):
    parser = argparse.ArgumentParser(description="CKD EEGNet training and evaluation")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--align", choices=("online", "none"), default="online")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32 if stage == "teacher_single" else 64)
    parser.add_argument("--epochs", type=int, default=200 if stage == "teacher_multi" else 100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--f1", type=int, default=4)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--f2", type=int, default=8)
    if stage == "teacher_single":
        parser.add_argument("--sources", choices=DATASETS, nargs="+", default=list(DATASETS))
    else:
        parser.add_argument("--targets", choices=DATASETS, nargs="+", default=list(DATASETS))
    if stage.startswith("student"):
        if stage == "student_single":
            parser.add_argument("--sources", choices=DATASETS, nargs="+")
        parser.add_argument("--subjects", type=int, nargs="+")
        parser.add_argument("--kd-weight", type=float, default=4.0)
        parser.add_argument("--contrastive-weight", type=float, default=0.1)
        parser.add_argument("--kd-temperature", type=float, default=4.0)
        parser.add_argument("--contrastive-temperature", type=float, default=0.2)
        parser.add_argument("--teacher-checkpoint")
        parser.add_argument("--teacher-encoder-checkpoint")
        parser.add_argument("--teacher-classifier-checkpoint")
    return parser


def _validate(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Epochs, batch size, learning rate, and worker count must be valid.")
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32).")
    for name in ("sources", "targets", "subjects"):
        values = getattr(args, name, None)
        if values is not None and len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicates.")
    for name in ("kd_weight", "contrastive_weight", "kd_temperature", "contrastive_temperature"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0 or ("temperature" in name and value == 0)):
            raise ValueError(f"Invalid {name}.")


def _args_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _logger(folder, label):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "training.jsonl"
    if path.exists():
        raise FileExistsError(f"Select another output root: {path}")
    def log(row):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(label, json.dumps(row, allow_nan=False), flush=True)
    return log


def _pretrain(args, scenario, device):
    tasks = [(source, [source]) for source in args.sources] if scenario == "single" else [
        (target, [name for name in DATASETS if name != target]) for target in args.targets
    ]
    for key, names in tasks:
        datasets = {name: load_dataset(args.data_root, name, align=args.align == "online") for name in names}
        configs = {name: _model_config(dataset, args) for name, dataset in datasets.items()}
        for seed in args.seeds:
            path = _teacher_path(args.output_root, scenario, key if scenario == "single" else None, key, seed)
            if path.exists():
                raise FileExistsError(f"Teacher checkpoint already exists: {path}")
            _json(path.parent / "config.json", _args_json(args))
            seed_everything(seed)
            embeddings, encoder, classifier, _ = train_teacher(
                datasets, configs, args.epochs, args.batch_size, args.lr, device, args.workers,
                _logger(path.parent, f"teacher/{scenario}/{key}/seed{seed}"),
            )
            _checkpoint(path, {
                "stage": "teacher", "scenario": scenario, "source_datasets": names, "seed": seed,
                "alignment": args.align, "model_config": asdict(next(iter(configs.values()))),
                "embeddings": {name: module.state_dict() for name, module in embeddings.items()},
                "encoder": encoder.state_dict(), "classifier": classifier.state_dict(),
            })


def _adapt(args, scenario, device):
    for target in args.targets:
        dataset = load_dataset(args.data_root, target, align=args.align == "online")
        config = _model_config(dataset, args)
        subjects = sorted(int(s) for s in np.unique(dataset.subjects))
        selected = args.subjects or subjects
        if not set(selected).issubset(subjects):
            raise ValueError(f"Requested subjects are not present in {target}: {selected}")
        sources = (args.sources if args.sources is not None else [name for name in DATASETS if name != target]) if scenario == "single" else [None]
        if target in sources:
            raise ValueError("Source and target datasets must differ.")
        for source in sources:
            task = f"{source or 'all'}_to_{target}"
            result_path = args.output_root / "results" / f"{scenario}_{task}.json"
            if result_path.exists():
                raise FileExistsError(f"Results already exist: {result_path}")
            rows = []
            for seed in args.seeds:
                teacher_state = _teacher_state(args, scenario, source, target, seed)
                for subject in selected:
                    folder = args.output_root / "students" / scenario / task / f"seed_{seed}" / f"subject_{subject}"
                    if (folder / "student.pt").exists():
                        raise FileExistsError(f"Student checkpoint already exists: {folder}")
                    x_train, y_train, x_test, y_test = split_loso(dataset, subject)
                    _json(folder / "config.json", {**_args_json(args), "target": target, "source": source,
                          "held_out_subject": subject, "seed": seed, "train_trials": len(y_train), "test_trials": len(y_test),
                          "model_config": asdict(config)})
                    seed_everything(seed)
                    modules = initialize_adaptation(config, teacher_state, device)
                    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.workers)
                    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.workers)
                    train_student(
                        modules, train_loader, args.epochs, args.lr, device, args.kd_weight,
                        args.contrastive_weight, args.kd_temperature, args.contrastive_temperature,
                        _logger(folder, f"student/{scenario}/{task}/seed{seed}/subject{subject}"),
                    )
                    model = modules.student_model()
                    metrics, labels, probabilities = evaluate_model(model, test_loader, device)
                    _checkpoint(folder / "student.pt", {
                        "stage": "student", "scenario": scenario, "source_datasets": [source] if source else [name for name in DATASETS if name != target],
                        "target_dataset": target, "held_out_subject": subject, "seed": seed, "alignment": args.align,
                        "model_config": asdict(config), "embedding": modules.embedding.state_dict(),
                        "encoder": modules.student_encoder.state_dict(), "classifier": modules.student_classifier.state_dict(),
                    })
                    np.savez(folder / "predictions.npz", labels=labels, probabilities=probabilities,
                             subject=np.full(len(labels), subject, dtype=np.int64))
                    row = {"seed": seed, "subject": subject, **metrics}
                    _json(folder / "metrics.json", row)
                    rows.append(row)
                    print(task, json.dumps(row), flush=True)
            seed_means = {seed: mean(row["accuracy"] for row in rows if row["seed"] == seed) for seed in args.seeds}
            summary = {"task": task, "scenario": scenario, "seed_results": seed_means,
                       "accuracy_mean": mean(seed_means.values()),
                       "accuracy_std": stdev(seed_means.values()) if len(seed_means) > 1 else 0.0,
                       "subject_results": rows}
            _json(result_path, summary)
            print(json.dumps(summary), flush=True)


def main(stage, argv=None):
    if stage not in {"teacher_single", "teacher_multi", "student_single", "student_cross"}:
        raise ValueError("Unknown training stage.")
    args = _parser(stage).parse_args(argv)
    _validate(args)
    device = _device(args.device)
    scenario = "single" if stage.endswith("single") else "multi"
    if stage.startswith("teacher"):
        _pretrain(args, scenario, device)
    else:
        _adapt(args, scenario, device)


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a CKD student checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    state = _load(args.checkpoint)
    if state.get("stage") != "student":
        raise ValueError("Evaluation requires a student checkpoint.")
    device = _device(args.device)
    config = ModelConfig(**state["model_config"])
    embedding, encoder, classifier = build_modules(config)
    for name, module in (("embedding", embedding), ("encoder", encoder), ("classifier", classifier)):
        module.load_state_dict(state[name], strict=True)
    model = CKDModel(embedding, encoder, classifier).to(device)
    dataset = load_dataset(args.data_root, state["target_dataset"], align=state["alignment"] == "online")
    _, _, x, y = split_loso(dataset, state["held_out_subject"])
    metrics, _, _ = evaluate_model(model, make_loader(x, y, args.batch_size, False), device)
    print(json.dumps(metrics, indent=2))
