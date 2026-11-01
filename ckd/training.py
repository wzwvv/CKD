# Author: Ziwei Wang

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .losses import contrastive_loss, distillation_loss
from .models import CKDModel, ModelConfig, build_modules, build_multisource_modules


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unique_parameters(*modules):
    parameters = [p for module in modules for p in module.parameters() if p.requires_grad]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Each optimizer must contain each parameter once.")
    return parameters


def make_loader(x, y, batch_size, train, workers=0):
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid loader settings.")
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Inputs and labels must have the same positive length.")
    if train and len(x) < batch_size:
        raise ValueError("Training set is smaller than batch size; select a smaller batch size.")
    dataset = TensorDataset(
        torch.as_tensor(np.asarray(x), dtype=torch.float32).unsqueeze(1),
        torch.as_tensor(np.asarray(y), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=workers)


@dataclass
class AdaptationModules:
    embedding: nn.Module
    teacher_encoder: nn.Module
    teacher_classifier: nn.Module
    student_encoder: nn.Module
    student_classifier: nn.Module

    def student_model(self):
        return CKDModel(self.embedding, self.student_encoder, self.student_classifier)


def initialize_adaptation(config: ModelConfig, teacher_state, device="cpu"):
    embedding, teacher_encoder, teacher_classifier = build_modules(config)
    _, student_encoder, student_classifier = build_modules(config)
    teacher_encoder.load_state_dict(teacher_state["encoder"], strict=True)
    teacher_classifier.load_state_dict(teacher_state["classifier"], strict=True)
    modules = AdaptationModules(embedding, teacher_encoder, teacher_classifier, student_encoder, student_classifier)
    for module in vars(modules).values():
        module.to(device)
    teacher_ids = {id(p) for p in unique_parameters(teacher_encoder, teacher_classifier)}
    student_ids = {id(p) for p in unique_parameters(student_encoder, student_classifier)}
    if teacher_ids & student_ids:
        raise ValueError("Teacher and student encoders/classifiers must be independent.")
    return modules


def adaptation_step(
    embedding, teacher_encoder, teacher_classifier, student_encoder, student_classifier,
    teacher_optimizer, student_optimizer, x, y,
    kd_weight=4.0, ct_weight=0.1, kd_temperature=4.0, ct_temperature=0.2,
):
    if not all(math.isfinite(value) and value >= 0 for value in (kd_weight, ct_weight)):
        raise ValueError("Loss weights must be finite and nonnegative.")
    teacher_parameters = unique_parameters(embedding, teacher_encoder, teacher_classifier)
    student_parameters = unique_parameters(embedding, student_encoder, student_classifier)
    shared_ids = {id(p) for p in unique_parameters(embedding)}
    if {id(p) for p in teacher_parameters} & {id(p) for p in student_parameters} != shared_ids:
        raise ValueError("Only the embedding may be shared between teacher and student.")
    for optimizer, parameters in ((teacher_optimizer, teacher_parameters), (student_optimizer, student_parameters)):
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {id(p) for p in parameters}:
            raise ValueError("Optimizer parameters do not match the corresponding branch.")
    e = embedding(x)
    teacher_features, teacher_logits = teacher_classifier(teacher_encoder(e))
    student_features, student_logits = student_classifier(student_encoder(e))
    teacher_ce = F.cross_entropy(teacher_logits, y)
    student_ce = F.cross_entropy(student_logits, y)
    kd = distillation_loss(student_logits, teacher_logits.detach(), kd_temperature)
    ct = contrastive_loss(student_features, teacher_features.detach(), ct_temperature)
    student_total = student_ce + kd_weight * kd + ct_weight * ct
    teacher_gradients = torch.autograd.grad(teacher_ce, teacher_parameters, retain_graph=True)
    student_gradients = torch.autograd.grad(student_total, student_parameters)
    teacher_optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(teacher_parameters, teacher_gradients):
        parameter.grad = gradient.detach()
    teacher_optimizer.step()
    student_optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(student_parameters, student_gradients):
        parameter.grad = gradient.detach()
    student_optimizer.step()
    return {
        "teacher_ce": float(teacher_ce.detach()), "student_ce": float(student_ce.detach()),
        "kd": float(kd.detach()), "contrastive": float(ct.detach()),
        "student_total": float(student_total.detach()),
    }


def train_student(modules, loader, epochs=100, lr=0.001, device="cpu", kd_weight=4.0,
                  ct_weight=0.1, kd_temperature=4.0, ct_temperature=0.2, callback=None):
    if epochs < 1 or not math.isfinite(lr) or lr <= 0 or len(loader) == 0:
        raise ValueError("Positive epochs, learning rate, and nonempty training loader are required.")
    teacher_optimizer = torch.optim.Adam(unique_parameters(modules.embedding, modules.teacher_encoder, modules.teacher_classifier), lr=lr)
    student_optimizer = torch.optim.Adam(unique_parameters(modules.embedding, modules.student_encoder, modules.student_classifier), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        for module in vars(modules).values():
            module.train()
        sums, count = {}, 0
        for x, y in loader:
            metrics = adaptation_step(
                modules.embedding, modules.teacher_encoder, modules.teacher_classifier,
                modules.student_encoder, modules.student_classifier, teacher_optimizer, student_optimizer,
                x.to(device), y.to(device), kd_weight, ct_weight, kd_temperature, ct_temperature,
            )
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * len(y)
            count += len(y)
        row = {"epoch": epoch, **{name: value / count for name, value in sums.items()}}
        history.append(row)
        if callback is not None:
            callback(row)
    return history


def train_teacher(datasets, configs, epochs, batch_size, lr=0.001, device="cpu", workers=0, callback=None):
    if not datasets or set(datasets) != set(configs) or epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Teacher datasets, models, and training settings must be valid.")
    embeddings, encoder, classifier = build_multisource_modules(configs)
    embeddings, encoder, classifier = embeddings.to(device), encoder.to(device), classifier.to(device)
    optimizer = torch.optim.Adam(unique_parameters(embeddings, encoder, classifier), lr=lr)
    loaders = {name: make_loader(d.x, d.y, batch_size, True, workers) for name, d in datasets.items()}
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    history = []
    for epoch in range(1, epochs + 1):
        embeddings.train()
        encoder.train()
        classifier.train()
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        total_loss, count = 0.0, 0
        for _ in range(steps_per_epoch):
            features, labels = [], []
            for name, loader in loaders.items():
                try:
                    x, y = next(iterators[name])
                except StopIteration:
                    iterators[name] = iter(loader)
                    x, y = next(iterators[name])
                features.append(embeddings[name](x.to(device)))
                labels.append(y.to(device))
            y = torch.cat(labels)
            _, logits = classifier(encoder(torch.cat(features)))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
        row = {"epoch": epoch, "teacher_ce": total_loss / count}
        history.append(row)
        if callback is not None:
            callback(row)
    return embeddings, encoder, classifier, history


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for x, y in loader:
            _, output = model(x.to(device))
            logits.append(output.cpu())
            labels.append(y)
    if not labels:
        raise ValueError("Evaluation loader must not be empty.")
    y = torch.cat(labels).numpy()
    probabilities = torch.cat(logits).softmax(1).numpy()
    predictions = probabilities.argmax(1)
    return {"accuracy": float(np.mean(predictions == y) * 100), "n_trials": len(y)}, y, probabilities
