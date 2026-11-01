# Author: Ziwei Wang

import math

import torch
import torch.nn.functional as F


def _validate_pair(student: torch.Tensor, teacher: torch.Tensor, temperature: float):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student.ndim != 2 or teacher.ndim != 2:
        raise ValueError("student and teacher must both have shape [batch, features]")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher must have identical shapes")
    if student.shape[0] < 1 or student.shape[1] < 1:
        raise ValueError("student and teacher must have non-empty batches and features")
    if student.device != teacher.device:
        raise ValueError("student and teacher must be on the same device")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("student and teacher must be floating-point tensors")


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0):
    _validate_pair(student_logits, teacher_logits, temperature)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * temperature ** 2


def contrastive_loss(student_features: torch.Tensor, teacher_features: torch.Tensor, temperature: float = 0.2):
    _validate_pair(student_features, teacher_features, temperature)
    student = F.normalize(student_features, dim=1)
    teacher = F.normalize(teacher_features.detach(), dim=1)
    logits = student @ teacher.transpose(0, 1) / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    return F.cross_entropy(logits, labels)
