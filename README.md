<div align="center">
<h1>CKD</h1>
<h3>Contrastive Knowledge Distillation for Cross-Dataset EEG Classification</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, [Xingyi He](https://scholar.google.com/citations?user=nNPi7lcAAAAJ&hl=en)<sup>1</sup>, [Hongbin Wang](https://scholar.google.com/citations?user=1F5ERP8AAAAJ&hl=en)<sup>1</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>1</sup> Hubei Key Laboratory of Brain-inspired Intelligent Systems, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(<sup>:email:</sup>) Corresponding Author

[Paper](https://doi.org/10.1109/TBME.2026.3701548) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/42258675/) | [Citation](#citation)

</div>

> This repository contains the implementation of our paper [**"CKD: Contrastive Knowledge Distillation for Cross-Dataset EEG Classification"**](https://doi.org/10.1109/TBME.2026.3701548), accepted for publication in **IEEE Transactions on Biomedical Engineering (IEEE TBME)**.

## Overview

**CKD** transfers knowledge across motor imagery EEG datasets through a two-stage framework:

1. **Cross-dataset teacher pretraining:** Train a teacher on one or multiple auxiliary datasets. In the multi-source setting, each auxiliary dataset has its own embedding module, while the encoder and classifier are shared.
2. **Cross-subject adaptation:** On the target dataset's training subjects, adapt the teacher and train an independent student using logit-level knowledge distillation and feature-level contrastive alignment. The target embedding is shared by the two branches. Only the student branch is retained for inference.

This release provides the **pipeline** for five MI datasets under single-source and multi-source transfer settings. Target-dataset evaluation follows **leave-one-subject-out (LOSO)**: labeled trials from the remaining target subjects are used for training, and the held-out subject is used only for evaluation. This is not label-free transfer to an entirely unseen target dataset or test-time gradient adaptation.

## Features

- **Two-stage knowledge transfer** across datasets and subjects.
- **Logit-level distillation and feature-level InfoNCE** with independently configurable weights and temperatures.
- **Dataset-specific embeddings** for different channel configurations.
- **Single-source and multi-source training** with subject-wise online Euclidean Alignment (EA).
- **Reproducible LOSO evaluation** with configurable seeds, saved configurations, training logs, student checkpoints, and per-subject predictions.

## Code Structure

```text
CKD/
├── prepare_data.py              # Prepare epoch arrays and subject metadata
├── train_teacher_single.py      # Single-source teacher pretraining
├── train_teacher_multi.py       # Multi-source teacher pretraining
├── train_student_single.py      # Single-source transfer with target LOSO
├── train_student_cross.py       # Multi-source transfer with target LOSO
├── evaluate.py                  # Evaluate a saved student checkpoint
├── ckd/
│   ├── models.py                # EEGNet embedding, encoder, and classifier
│   ├── losses.py                # Knowledge distillation and InfoNCE
│   ├── data.py                  # Data preparation, online EA, and LOSO
│   ├── training.py              # Teacher and student training
│   └── cli.py                   # Command-line interfaces and output handling
├── tests/                      # Unit and integration tests
├── requirements.txt
└── README.md
```

## Environment

Python 3.10 or later is required. Run the following commands from this directory:

```bash
python -m pip install -r requirements.txt
```

Use `--device cuda` for GPU training or `--device cpu` for CPU execution. With multiple GPUs, select a device using `CUDA_VISIBLE_DEVICES`.

## Datasets and Data Preparation

The input is the epoch-array format used by the project, rather than continuous EDF or MATLAB recordings. Apply the dataset's original filtering and cue-relative epoch extraction before exporting these arrays. Keep trials in subject, session, and acquisition order.

Supported datasets: `BNCI2014001`, `BNCI2014004`, `MI1`, `HighGamma`, and `Weibo2014`. `MI1` denotes the Blankertz dataset.

For each dataset, place the exported arrays and available metadata under:

```text
raw/
  BNCI2014001/
    X.npy
    labels.npy
    meta.csv
  BNCI2014004/
    X.npy
    labels.npy
    meta.csv
  MI1/
    X.npy
    labels.npy
  HighGamma/
    X0.npy ... X13.npy
    y0.npy ... y13.npy
  Weibo2014/
    X.npy
    labels.npy
    meta.csv
```

- `X` has shape `[trials, channels, samples]` and contains the source EEG epochs.
- BNCI and Weibo labels use `left_hand` and `right_hand`; other classes are filtered during preparation. MI1 and HighGamma use numeric labels with left/right hand encoded as `0/1`.
- HighGamma may alternatively use concatenated `X.npy` and `labels.npy` with subject metadata.
- `meta.csv` is the original export metadata, with `subject` and `session` columns. Alternatively, use `subjects.npy`, `sessions.npy`, and optional `trial_ids.npy`; IDs start at 1.
- `metadata.json` can specify the raw `sampling_rate`. Defaults are 250 Hz for BNCI, 1000 Hz for MI1, 500 Hz for HighGamma, and 200 Hz for Weibo.
- Without metadata, arrays must follow the original complete subject/session ordering and trial counts.

Prepare all datasets:

```bash
python prepare_data.py \
  --raw-root /path/to/raw \
  --data-root /path/to/prepared \
  --datasets BNCI2014001 BNCI2014004 MI1 HighGamma Weibo2014 \
  --sessions first \
  --align online
```

`--sessions first` selects the first session of BNCI2014001 and BNCI2014004; the other datasets retain their supplied sessions. Use `--sessions all` to retain every supplied session. MI1 excludes original subject IDs 1 and 6 by default and retains the remaining original IDs. Set `--mi1-exclude-subjects ID1 ID2` to specify another selection.

Preparation selects left/right-hand trials, resamples to 250 Hz, retains 1000 samples, and applies subject-wise online Euclidean Alignment. Each trial updates its subject's cumulative covariance reference before whitening. Use `--align none` to omit EA.

Prepared output:

```text
prepared/BNCI/DATASET/seta/
  X.npy
  X-EA.npy
  y.npy
  subjects.npy
  sessions.npy
  trial_ids.npy
  metadata.json
```

`X-EA.npy` is present when EA is enabled. Existing prepared layouts `DATA_ROOT/DATASET/seta/` and `DATA_ROOT/DATASET/` are also accepted. Use a new destination for each data-preparation run.

For a standard five-subject MI1 cache without metadata, subject blocks correspond to original IDs `2, 3, 4, 5, 7`. Supply `subjects.npy` for any other selection.

## Experimental Scenarios

Both scenarios first pretrain the teacher on auxiliary datasets, then train the target embedding and student using the target dataset's training subjects. The test subject is excluded from supervised training.

### Single-Source Transfer

First train the auxiliary-dataset teacher:

```bash
python train_teacher_single.py \
  --data-root /path/to/prepared \
  --output-root runs/single \
  --sources BNCI2014004 \
  --seeds 1 2 3 \
  --device cuda
```

Then train and evaluate the student on the target dataset under LOSO:

```bash
python train_student_single.py \
  --data-root /path/to/prepared \
  --output-root runs/single \
  --sources BNCI2014004 \
  --targets BNCI2014001 \
  --seeds 1 2 3 \
  --device cuda
```

The student command evaluates every target subject by default. For a subset, add `--subjects 1 2`. Subjects are identified by their original natural IDs. Source and target datasets must differ.

### Multi-Source Transfer

The target dataset is excluded from teacher pretraining. For target BNCI2014001, the teacher uses the other four datasets:

```bash
python train_teacher_multi.py \
  --data-root /path/to/prepared \
  --output-root runs/multi \
  --targets BNCI2014001 \
  --seeds 1 2 3 \
  --device cuda

python train_student_cross.py \
  --data-root /path/to/prepared \
  --output-root runs/multi \
  --targets BNCI2014001 \
  --seeds 1 2 3 \
  --device cuda
```

Omitting dataset-selection options runs all datasets or transfer pairs supported by that command.

## Training Settings

| Stage | Epochs | Batch size | Optimizer | Learning rate |
|---|---:|---:|---|---:|
| Single-source teacher | 100 | 32 | Adam | 0.001 |
| Multi-source teacher | 200 | 64 per source | Adam | 0.001 |
| LOSO student | 100 | 64 | Adam | 0.001 |

Multi-source steps concatenate one batch from each auxiliary dataset after their respective embeddings; shorter loaders cycle until the longest loader completes an epoch. Training loaders shuffle and drop the final incomplete batch. Evaluation retains all test trials.

The teacher encoder/classifier are initialized from pretraining. The target embedding and student encoder/classifier are randomly initialized. The shared embedding is optimized by both branch objectives: teacher CE and student CE plus KD and contrastive alignment. Only the student branch is used for inference.

Student defaults: `--kd-weight 4`, `--contrastive-weight 0.1`, `--kd-temperature 4`, and `--contrastive-temperature 0.2`. For another trade-off, supply the weights explicitly:

```bash
python train_student_single.py \
  --data-root /path/to/prepared \
  --output-root runs/weight_setting \
  --teacher-checkpoint 'runs/single/teachers/single/BNCI2014004/seed_{seed}/teacher.pt' \
  --sources BNCI2014004 --targets BNCI2014001 \
  --kd-weight 1 --contrastive-weight 0.01 \
  --seeds 1 2 3 --device cuda
```

Set `--contrastive-weight 0` for the KD-only objective. `--epochs`, `--batch-size`, `--lr`, `--f1`, `--d`, and `--f2` are configurable. All teacher and student commands must use the same alignment setting and compatible model settings.

To load separately saved encoder and classifier state dictionaries, use both `--teacher-encoder-checkpoint` and `--teacher-classifier-checkpoint`. Checkpoint path templates accept `{source}`, `{target}`, and `{seed}`.

## Outputs and Evaluation

```text
runs/
  teachers/single/SOURCE/seed_1/teacher.pt
  teachers/multi/TARGET/seed_1/teacher.pt
  students/single/SOURCE_to_TARGET/seed_1/subject_1/
    student.pt
    config.json
    training.jsonl
    metrics.json
    predictions.npz
  results/single_SOURCE_to_TARGET.json
```

Multi-source student outputs use `students/multi/all_to_TARGET/`. Result files report each held-out subject and seed, then the mean subject accuracy per seed and mean/sample standard deviation across seeds. Accuracy is in percent.

Evaluate a saved student checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/single/students/single/BNCI2014004_to_BNCI2014001/seed_1/subject_1/student.pt \
  --data-root /path/to/prepared \
  --device cuda
```

Each checkpoint records the target subject, model settings, and alignment setting. Existing output files are not overwritten; select another output root for a new run.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{wang2026ckd,
  author  = {Ziwei Wang and Xingyi He and Hongbin Wang and Dongrui Wu},
  title   = {{CKD}: Contrastive Knowledge Distillation for Cross-Dataset {EEG} Classification},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2026},
  doi     = {10.1109/TBME.2026.3701548},
  url     = {https://doi.org/10.1109/TBME.2026.3701548},
  note    = {Early access}
}
```

## License

This project is released under the [PolyForm Noncommercial License 1.0.0](https://spdx.org/licenses/PolyForm-Noncommercial-1.0.0.html) for academic and non-commercial research use.

For commercial licensing inquiries, please contact Ziwei Wang at [vivi@hust.edu.cn](mailto:vivi@hust.edu.cn).

## Acknowledgments

We thank the developers of [EEGNet](https://github.com/vlawhern/arl-eegmodels) and [MNE-Python](https://mne.tools/) for their work and software.
