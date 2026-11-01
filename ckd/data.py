# Author: Ziwei Wang

import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import fractional_matrix_power


DATASETS = ("BNCI2014001", "BNCI2014004", "MI1", "HighGamma", "Weibo2014")
RAW_COUNTS = {
    "BNCI2014001": [[288, 288]] * 9,
    "BNCI2014004": [
        [120, 120, 160, 160, 160],
        [120, 120, 160, 120, 160],
        [120, 120, 160, 160, 160],
        [120, 140, 160, 160, 160],
        [120, 140, 160, 160, 160],
        [120, 120, 160, 160, 160],
        [120, 120, 160, 160, 160],
        [160, 120, 160, 160, 160],
        [120, 120, 160, 160, 160],
    ],
    "MI1": [[200]] * 7,
    "HighGamma": [[n] for n in (320, 813, 880, 897, 720, 880, 880, 654, 880, 880, 880, 880, 800, 880)],
    "Weibo2014": [[237]] * 10,
}
LEGACY_COUNTS = {
    "BNCI2014001": (288,) * 9,
    "BNCI2014004": (600, 600, 600, 600, 600, 600, 600, 800, 600),
    "MI1": (200,) * 5,
    "HighGamma": (160, 406, 440, 448, 360, 440, 440, 328, 440, 440, 440, 440, 400, 440),
    "Weibo2014": (158, 156, 160, 159, 158, 157, 156, 160, 158, 158),
}
CHANNELS = {"BNCI2014001": 22, "BNCI2014004": 3, "MI1": 59, "HighGamma": 128, "Weibo2014": 60}
RAW_RATES = {"BNCI2014001": 250, "BNCI2014004": 250, "MI1": 1000, "HighGamma": 500, "Weibo2014": 200}


@dataclass
class EEGDataset:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    name: str
    sampling_rate: int = 250
    sessions: np.ndarray | None = None
    trial_ids: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = _integer_vector(self.y, "y")
        self.subjects = _integer_vector(self.subjects, "subjects")
        n = len(self.y)
        if self.x.ndim != 3 or self.x.shape[0] != n or self.x.shape[-1] != 1000:
            raise ValueError("Prepared EEG must have shape [trials, channels, 1000].")
        if len(self.subjects) != n or np.any(self.subjects < 1):
            raise ValueError("subjects must contain one natural subject ID (>= 1) per trial.")
        if self.sampling_rate != 250:
            raise ValueError("Prepared sampling_rate must be 250 Hz.")
        if not np.isfinite(self.x).all() or not np.isin(self.y, [0, 1]).all():
            raise ValueError("EEG must be finite and labels must be 0/1 (left/right hand).")
        if n == 0:
            raise ValueError("Dataset contains no trials.")
        if self.sessions is None:
            self.sessions = np.zeros(n, dtype=np.int64)
        else:
            self.sessions = _integer_vector(self.sessions, "sessions")
        if self.trial_ids is None:
            self.trial_ids = np.arange(1, n + 1, dtype=np.int64)
        else:
            self.trial_ids = _integer_vector(self.trial_ids, "trial_ids")
        if len(self.sessions) != n or len(self.trial_ids) != n:
            raise ValueError("Session and trial metadata must match the number of trials.")
        if np.any(self.sessions < 0) or np.any(self.trial_ids < 1):
            raise ValueError("Sessions must be nonnegative and trial identifiers must be positive.")
        keys = np.column_stack((self.subjects, self.sessions, self.trial_ids))
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError("Duplicate subject/session/trial identifiers.")


def _integer_vector(values, name):
    arr = np.asarray(values).reshape(-1)
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite integers.")
    converted = arr.astype(np.int64)
    if not np.array_equal(arr, converted):
        raise ValueError(f"{name} must contain integers.")
    return converted


def _load_array(path):
    return np.load(path, allow_pickle=False)


def _json_metadata(folder):
    path = folder / "metadata.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a JSON object in {path}.")
        return metadata
    return {}


def _trial_metadata(folder, n, metadata):
    arrays = []
    for key, aliases in (
        ("subjects", ("subjects.npy", "subject_ids.npy")),
        ("sessions", ("sessions.npy", "session_ids.npy")),
        ("trial_ids", ("trial_ids.npy",)),
    ):
        existing = next((folder / name for name in aliases if (folder / name).is_file()), None)
        value = _load_array(existing) if existing else metadata.get(key)
        if value is not None:
            value = _integer_vector(value, key)
            if len(value) != n:
                raise ValueError(f"{folder}: {key} length does not match {n} trials.")
        arrays.append(value)
    csv_path = folder / "meta.csv"
    if arrays[0] is None and csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != n or not rows or "subject" not in rows[0] or "session" not in rows[0]:
            raise ValueError(f"{csv_path} must contain subject/session columns and one row per trial.")
        subjects, sessions, mappings = [], [], {}
        for row in rows:
            try:
                subject = int(row["subject"])
            except ValueError as error:
                raise ValueError(f"{csv_path}: subjects must be natural integer IDs.") from error
            mapping = mappings.setdefault(subject, {})
            label = row["session"]
            if label not in mapping:
                mapping[label] = len(mapping) + 1
            subjects.append(subject)
            sessions.append(mapping[label])
        arrays = [np.asarray(subjects, dtype=np.int64), np.asarray(sessions, dtype=np.int64), np.arange(1, n + 1, dtype=np.int64)]
    if arrays[0] is None and any(value is not None for value in arrays):
        raise ValueError(f"{folder}: subject metadata is required with session/trial metadata.")
    return arrays


def _ordered_metadata(counts, subject_ids=None):
    subjects, sessions, trials = [], [], []
    subject_ids = subject_ids if subject_ids is not None else range(1, len(counts) + 1)
    for subject, session_counts in zip(subject_ids, counts):
        for session, count in enumerate(session_counts, start=1):
            subjects.extend([subject] * count)
            sessions.extend([session] * count)
            trials.extend(range(1, count + 1))
    return tuple(np.asarray(values, dtype=np.int64) for values in (subjects, sessions, trials))


def align_online(x, subjects=None):
    x = np.asarray(x)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("EA input must be finite [trials, channels, samples].")
    subject_ids = np.ones(len(x), dtype=np.int64) if subjects is None else _integer_vector(subjects, "subjects")
    if len(subject_ids) != len(x):
        raise ValueError("EA subject metadata length mismatch.")
    output = np.empty(x.shape, dtype=np.float32)
    for subject in np.unique(subject_ids):
        reference = 0
        for count, index in enumerate(np.flatnonzero(subject_ids == subject)):
            current = np.asarray(x[index], dtype=np.float64)
            reference = (reference * count + np.cov(current)) / (count + 1)
            whitening = np.real_if_close(fractional_matrix_power(reference, -0.5))
            if np.iscomplexobj(whitening) or not np.isfinite(whitening).all():
                raise ValueError(f"EA covariance is not numerically invertible for subject {subject}, trial {count + 1}.")
            output[index] = whitening @ current
    if not np.isfinite(output).all():
        raise ValueError("EA produced non-finite values.")
    return output


def _cache_folder(data_root, name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}.")
    root = Path(data_root).expanduser()
    candidates = (root / "BNCI" / name / "seta", root / name / "seta", root / name, root)
    for folder in candidates:
        if (folder / "y.npy").is_file() or (name == "HighGamma" and (folder / "y0.npy").is_file()):
            return folder
    raise FileNotFoundError(f"No prepared seta cache for {name} under {root}.")


def _legacy_subjects(name, n, per_subject=None):
    counts = tuple(per_subject) if per_subject is not None else LEGACY_COUNTS[name]
    if per_subject is None and name == "BNCI2014001" and n == 1296:
        counts = (144,) * 9
    if per_subject is None and name == "BNCI2014004":
        first_counts = tuple(counts[0] for counts in RAW_COUNTS[name])
        all_counts = tuple(sum(counts) for counts in RAW_COUNTS[name])
        if n == sum(first_counts):
            counts = first_counts
        elif n == sum(all_counts):
            counts = all_counts
    if name == "HighGamma" and per_subject is not None and len(counts) != 14:
        raise ValueError("HighGamma legacy cache must contain all 14 subject files.")
    if n != sum(counts) or any(count <= 0 for count in counts):
        raise ValueError(f"{name}: legacy cache has {n} trials; expected {sum(counts)}. Supply explicit subjects.npy for other valid selections.")
    subject_ids = np.array([2, 3, 4, 5, 7], dtype=np.int64) if name == "MI1" else np.arange(1, len(counts) + 1, dtype=np.int64)
    subjects = np.repeat(subject_ids, counts)
    warnings.warn(
        f"{name}: loading cached subject blocks without trial metadata; prepare_data.py can generate explicit identifiers.",
        UserWarning,
        stacklevel=3,
    )
    return subjects


def load_dataset(data_root, name, align=True):
    folder = _cache_folder(data_root, name)
    metadata = _json_metadata(folder)
    per_subject = None
    x_path = folder / ("X-EA.npy" if align else "X.npy")
    already_aligned = align and x_path.is_file()
    if not x_path.is_file() and align:
        x_path = folder / "X.npy"
    if (folder / "y.npy").is_file():
        if not x_path.is_file():
            raise FileNotFoundError(f"Missing EEG cache {x_path}.")
        x, y = _load_array(x_path), _load_array(folder / "y.npy")
    elif name == "HighGamma":
        parts, labels, per_subject = [], [], []
        already_aligned = bool(align)
        for index in range(14):
            file_x = folder / f"{'X-EA' if align else 'X'}{index}.npy"
            aligned_part = align and file_x.is_file()
            if not file_x.is_file() and align:
                file_x = folder / f"X{index}.npy"
            file_y = folder / f"y{index}.npy"
            if not file_x.is_file() or not file_y.is_file():
                raise FileNotFoundError(f"HighGamma requires paired files for subject {index + 1}: {file_x}, {file_y}.")
            part, label = _load_array(file_x), _load_array(file_y).reshape(-1)
            if len(part) != len(label):
                raise ValueError(f"HighGamma subject {index + 1}: EEG/label length mismatch.")
            if align and not aligned_part:
                part = align_online(part)
            parts.append(part)
            labels.append(label)
            per_subject.append(len(label))
        x, y = np.concatenate(parts), np.concatenate(labels)
    else:
        raise FileNotFoundError(f"Missing prepared data in {folder}.")
    subjects, sessions, trials = _trial_metadata(folder, len(y), metadata)
    if subjects is None:
        subjects = _legacy_subjects(name, len(y), per_subject)
        metadata = dict(metadata, provenance="legacy_shape_only")
    else:
        metadata = dict(metadata, provenance=metadata.get("provenance", "provided_metadata"))
    dataset = EEGDataset(x, y, subjects, name, int(metadata.get("sampling_rate", 250)), sessions, trials, metadata)
    if dataset.x.shape[1] != CHANNELS[name]:
        raise ValueError(f"{name}: expected {CHANNELS[name]} channels, got {dataset.x.shape[1]}.")
    if align and not already_aligned:
        dataset.x = align_online(dataset.x, dataset.subjects)
    return dataset


def split_loso(dataset, subject):
    test = dataset.subjects == int(subject)
    if not test.any() or test.all():
        raise ValueError(f"LOSO needs subject {subject} and at least one other subject.")
    return dataset.x[~test], dataset.y[~test], dataset.x[test], dataset.y[test]


def _raw_folder(raw_root, name):
    root = Path(raw_root).expanduser()
    for folder in (root / name, root):
        if (folder / "X.npy").is_file() or (name == "HighGamma" and (folder / "X0.npy").is_file()):
            return folder
    raise FileNotFoundError(f"Provide {name}/X.npy and labels.npy (or HighGamma X0..X13.npy/y0..y13.npy) under {root}.")


def _read_raw(raw_root, name):
    folder = _raw_folder(raw_root, name)
    metadata = _json_metadata(folder)
    if (folder / "X.npy").is_file():
        label_path = folder / "labels.npy"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing original labels file {label_path}.")
        x, y = _load_array(folder / "X.npy"), _load_array(label_path).reshape(-1)
        subjects, sessions, trials = _trial_metadata(folder, len(y), metadata)
    else:
        parts, labels, subject_parts = [], [], []
        for index in range(14):
            file_x, file_y = folder / f"X{index}.npy", folder / f"y{index}.npy"
            if not file_x.is_file() or not file_y.is_file():
                raise FileNotFoundError(f"Missing HighGamma pair {file_x}, {file_y}.")
            part, label = _load_array(file_x), _load_array(file_y).reshape(-1)
            if len(part) != len(label):
                raise ValueError(f"HighGamma subject {index + 1}: EEG/label length mismatch.")
            parts.append(part)
            labels.append(label)
            subject_parts.append(np.full(len(label), index + 1, dtype=np.int64))
        x, y = np.concatenate(parts), np.concatenate(labels)
        subjects, sessions, trials = _trial_metadata(folder, len(y), metadata)
        if subjects is None:
            subjects = np.concatenate(subject_parts)
    if x.ndim != 3 or len(x) != len(y) or x.shape[1] != CHANNELS[name]:
        raise ValueError(f"{name}: raw EEG must be [trials, {CHANNELS[name]}, samples] and match labels.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: raw EEG contains non-finite values.")
    if subjects is None:
        counts = RAW_COUNTS[name]
        expected = sum(map(sum, counts))
        if len(y) != expected:
            raise ValueError(f"{name}: expected {expected} ordered raw trials without metadata, got {len(y)}. Provide subjects.npy and sessions.npy for another valid selection.")
        subjects, sessions, trials = _ordered_metadata(counts)
        metadata["raw_order"] = "subject_then_session_then_trial"
    else:
        if np.any(subjects < 1):
            raise ValueError("Raw subject identifiers must be natural IDs (>= 1).")
        if sessions is None:
            if name in ("BNCI2014001", "BNCI2014004"):
                raise ValueError(f"{name}: sessions.npy is required when supplying subjects.npy.")
            sessions = np.ones(len(y), dtype=np.int64)
        if trials is None:
            trials = np.arange(1, len(y) + 1, dtype=np.int64)
    if np.any(sessions < 1):
        raise ValueError("Raw session identifiers must be natural IDs (>= 1).")
    identities = np.column_stack((subjects, sessions, trials))
    if len(np.unique(identities, axis=0)) != len(y):
        raise ValueError("Duplicate raw subject/session/trial identifiers.")
    return x, y, subjects, sessions, trials, metadata


def prepare_dataset(raw_root, name, sessions="first", align="online", mi1_exclude_subjects=None):
    if name not in DATASETS or sessions not in ("first", "all") or align not in ("online", "none"):
        raise ValueError("Invalid dataset, session selection, or alignment mode.")
    x, labels, subject_ids, session_ids, trial_ids, metadata = _read_raw(raw_root, name)
    selected = np.ones(len(labels), dtype=bool)
    if name == "MI1":
        exclusions = ([1, 6] if len(np.unique(subject_ids)) == 7 else []) if mi1_exclude_subjects is None else [int(value) for value in mi1_exclude_subjects]
        if exclusions:
            if len(set(exclusions)) != 2 or not set(exclusions).issubset(set(subject_ids.tolist())):
                raise ValueError("--mi1-exclude-subjects must give two distinct natural IDs present in the raw data.")
            selected &= ~np.isin(subject_ids, exclusions)
        elif len(np.unique(subject_ids)) != 5:
            raise ValueError("MI1 requires seven original subjects or an explicitly selected five-subject raw dataset.")
        metadata["excluded_subjects"] = exclusions
    if sessions == "first" and name in ("BNCI2014001", "BNCI2014004"):
        for subject in np.unique(subject_ids):
            mask = subject_ids == subject
            selected[mask] &= session_ids[mask] == session_ids[mask].min()
    if name in ("MI1", "HighGamma"):
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError(f"{name}: expected original numeric labels with left=0 and right=1.")
        selected &= np.isin(labels, [0, 1])
        y = labels[selected].astype(np.int64)
    else:
        if labels.dtype.kind not in ("U", "S"):
            raise ValueError(f"{name}: original labels.npy must contain left_hand/right_hand strings; prepared binary caches should be loaded with load_dataset.")
        labels = labels.astype(str)
        selected &= np.isin(labels, ["left_hand", "right_hand"])
        y = (labels[selected] == "right_hand").astype(np.int64)
    if not selected.any():
        raise ValueError(f"{name}: no left/right-hand trials remain.")
    x = np.asarray(x[selected], dtype=np.float64)
    subject_ids, session_ids, trial_ids = (values[selected] for values in (subject_ids, session_ids, trial_ids))
    sampling_rate = int(metadata.get("sampling_rate", RAW_RATES[name]))
    if sampling_rate <= 0 or x.shape[-1] < sampling_rate * 4:
        raise ValueError(f"{name}: raw epochs must contain at least four seconds at {sampling_rate} Hz.")
    if sampling_rate != 250:
        from mne.filter import resample

        x = resample(x, up=250, down=sampling_rate, axis=-1, verbose=False)
    x = np.asarray(x[..., :1000], dtype=np.float32)
    metadata = {
        "dataset": name,
        "sampling_rate": 250,
        "raw_sampling_rate": sampling_rate,
        "label_mapping": {"left_hand": 0, "right_hand": 1},
        "session_selection": sessions,
        "alignment": align,
        "provenance": "raw_trial_metadata",
        "excluded_subjects": metadata.get("excluded_subjects", []),
    }
    raw = EEGDataset(x, y, subject_ids, name, 250, session_ids, trial_ids, metadata)
    aligned = align_online(raw.x, raw.subjects) if align == "online" else None
    return raw, aligned


def save_dataset(dataset, data_root, aligned=None):
    folder = Path(data_root).expanduser() / "BNCI" / dataset.name / "seta"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {folder}. Choose a new --data-root.")
    if aligned is not None and (np.asarray(aligned).shape != dataset.x.shape or not np.isfinite(aligned).all()):
        raise ValueError("Aligned EEG shape/values do not match the raw prepared EEG.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("X.npy", dataset.x),
        ("y.npy", dataset.y),
        ("subjects.npy", dataset.subjects),
        ("sessions.npy", dataset.sessions),
        ("trial_ids.npy", dataset.trial_ids),
    ):
        np.save(folder / name, values, allow_pickle=False)
    if aligned is not None:
        np.save(folder / "X-EA.npy", aligned, allow_pickle=False)
    metadata = dict(dataset.metadata, sampling_rate=dataset.sampling_rate)
    metadata.pop("subjects", None)
    metadata.pop("sessions", None)
    metadata.pop("trial_ids", None)
    with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--sessions", choices=("first", "all"), default="first")
    parser.add_argument("--align", choices=("online", "none"), default="online")
    parser.add_argument("--mi1-exclude-subjects", type=int, nargs=2, help="Excluded natural subject IDs for MI1; default: 1 6 for seven-subject raw data.")
    args = parser.parse_args(argv)
    for name in args.datasets:
        dataset, aligned = prepare_dataset(args.raw_root, name, args.sessions, args.align, args.mi1_exclude_subjects)
        folder = save_dataset(dataset, args.data_root, aligned)
        print(f"{name}: trials={len(dataset.y)}, subjects={np.unique(dataset.subjects).tolist()}, shape={list(dataset.x.shape)}, output={folder}")
