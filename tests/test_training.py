# Author: Ziwei Wang

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ckd.models import CKDModel, ModelConfig, build_modules
from ckd.training import adaptation_step, initialize_adaptation


class CountingEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 5)
        self.norm = nn.BatchNorm1d(5)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.norm(self.linear(x)).tanh()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return x, self.linear(x)


def make_modules():
    return (
        CountingEmbedding(),
        nn.Sequential(nn.Linear(5, 3), nn.Tanh()),
        Classifier(),
        nn.Sequential(nn.Linear(5, 3), nn.Tanh()),
        Classifier(),
    )


def make_optimizers(modules, optimizer_type=torch.optim.Adam):
    embedding, teacher_encoder, teacher_classifier, student_encoder, student_classifier = modules
    teacher = optimizer_type(list(embedding.parameters()) + list(teacher_encoder.parameters()) + list(teacher_classifier.parameters()), lr=0.03)
    student = optimizer_type(list(embedding.parameters()) + list(student_encoder.parameters()) + list(student_classifier.parameters()), lr=0.03)
    return teacher, student


def parameter_snapshot(modules):
    return [[p.detach().clone() for p in module.parameters()] for module in modules]


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.modules = make_modules()
        self.x = torch.randn(7, 4)
        self.y = torch.tensor([0, 1, 0, 1, 1, 0, 1])

    def test_single_embedding_forward_and_all_branches_update(self):
        before = parameter_snapshot(self.modules)
        teacher_optimizer, student_optimizer = make_optimizers(self.modules)
        metrics = adaptation_step(*self.modules, teacher_optimizer, student_optimizer, self.x, self.y)
        self.assertEqual(self.modules[0].calls, 1)
        self.assertEqual(self.modules[0].norm.num_batches_tracked.item(), 1)
        self.assertEqual(set(metrics), {"teacher_ce", "student_ce", "kd", "contrastive", "student_total"})
        for name, value in metrics.items():
            self.assertIsInstance(value, float, name)
            self.assertTrue(torch.isfinite(torch.tensor(value)), name)
        self.assertAlmostEqual(metrics["student_total"], metrics["student_ce"] + 4.0 * metrics["kd"] + 0.1 * metrics["contrastive"], places=5)
        for old_parameters, module in zip(before, self.modules):
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(old_parameters, module.parameters())))
        shared_ids = {id(p) for p in self.modules[0].parameters()}
        optimizer_ids = []
        for optimizer in (teacher_optimizer, student_optimizer):
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            optimizer_ids.append(set(ids))
            for parameter in self.modules[0].parameters():
                self.assertEqual(optimizer.state[parameter]["step"].item(), 1)
        self.assertEqual(optimizer_ids[0] & optimizer_ids[1], shared_ids)
        self.assertIsNot(teacher_optimizer.state[self.modules[0].linear.weight], student_optimizer.state[self.modules[0].linear.weight])

    def test_teacher_update_independent_of_student_loss_weights(self):
        zero_modules = copy.deepcopy(self.modules)
        weighted_modules = copy.deepcopy(self.modules)
        zero_optimizers = make_optimizers(zero_modules, torch.optim.SGD)
        weighted_optimizers = make_optimizers(weighted_modules, torch.optim.SGD)
        adaptation_step(*zero_modules, *zero_optimizers, self.x, self.y, kd_weight=0.0, ct_weight=0.0)
        adaptation_step(*weighted_modules, *weighted_optimizers, self.x, self.y, kd_weight=6.0, ct_weight=0.7)
        for zero_branch, weighted_branch in zip(zero_modules[1:3], weighted_modules[1:3]):
            for zero_parameter, weighted_parameter in zip(zero_branch.parameters(), weighted_branch.parameters()):
                torch.testing.assert_close(zero_parameter, weighted_parameter, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(first, second) for first, second in zip(zero_modules[3].parameters(), weighted_modules[3].parameters())))

    def test_exact_cached_gradient_two_optimizer_reference(self):
        actual = copy.deepcopy(self.modules)
        expected = copy.deepcopy(self.modules)
        actual_optimizers = make_optimizers(actual)
        expected_optimizers = make_optimizers(expected)
        metrics = adaptation_step(*actual, *actual_optimizers, self.x, self.y, kd_weight=1.0, ct_weight=0.01)
        embedding, teacher_encoder, teacher_classifier, student_encoder, student_classifier = expected
        shared_features = embedding(self.x)
        teacher_features, teacher_logits = teacher_classifier(teacher_encoder(shared_features))
        student_features, student_logits = student_classifier(student_encoder(shared_features))
        teacher_ce = F.cross_entropy(teacher_logits, self.y)
        student_ce = F.cross_entropy(student_logits, self.y)
        p_teacher = F.softmax(teacher_logits.detach() / 4.0, dim=1)
        kd = (p_teacher * (p_teacher.log() - F.log_softmax(student_logits / 4.0, dim=1))).sum() / len(self.y) * 16.0
        contrastive_logits = F.normalize(student_features, dim=1) @ F.normalize(teacher_features.detach(), dim=1).T / 0.2
        contrastive = F.cross_entropy(contrastive_logits, torch.arange(len(self.y)))
        student_total = student_ce + kd + 0.01 * contrastive
        teacher_parameters = [p for module in expected[:3] for p in module.parameters()]
        student_parameters = [p for module in (embedding, student_encoder, student_classifier) for p in module.parameters()]
        teacher_gradients = torch.autograd.grad(teacher_ce, teacher_parameters, retain_graph=True)
        student_gradients = torch.autograd.grad(student_total, student_parameters)
        for optimizer, parameters, gradients in zip(expected_optimizers, (teacher_parameters, student_parameters), (teacher_gradients, student_gradients)):
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient.detach().clone()
            optimizer.step()
        for actual_module, expected_module in zip(actual, expected):
            for actual_parameter, expected_parameter in zip(actual_module.parameters(), expected_module.parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-6, atol=1e-7)
        for name, value in (("teacher_ce", teacher_ce), ("student_ce", student_ce), ("kd", kd), ("contrastive", contrastive), ("student_total", student_total)):
            self.assertAlmostEqual(metrics[name], value.item(), places=5)

    def test_pretrained_teacher_random_independent_student(self):
        config = ModelConfig(channels=3)
        _, encoder, classifier = build_modules(config)
        with torch.no_grad():
            for module in (encoder, classifier):
                for parameter in module.parameters():
                    parameter.fill_(0.125)
        teacher_state = {"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}
        modules = initialize_adaptation(config, teacher_state)
        self.assertTrue(callable(modules.student_model))
        student = modules.student_model()
        self.assertIsInstance(student, CKDModel)
        self.assertIs(student.embedding, modules.embedding)
        self.assertIs(student.encoder, modules.student_encoder)
        self.assertIs(student.classifier, modules.student_classifier)
        teacher_ids = {id(p) for module in (modules.teacher_encoder, modules.teacher_classifier) for p in module.parameters()}
        student_ids = {id(p) for module in (modules.student_encoder, modules.student_classifier) for p in module.parameters()}
        self.assertFalse(teacher_ids & student_ids)
        for key, tensor in encoder.state_dict().items():
            torch.testing.assert_close(modules.teacher_encoder.state_dict()[key], tensor, rtol=0, atol=0)
        for key, tensor in classifier.state_dict().items():
            torch.testing.assert_close(modules.teacher_classifier.state_dict()[key], tensor, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(first, second) for first, second in zip(modules.teacher_encoder.parameters(), modules.student_encoder.parameters())))
        self.assertTrue(any(not torch.equal(first, second) for first, second in zip(modules.teacher_classifier.parameters(), modules.student_classifier.parameters())))

    def test_repeated_steps_release_graph_and_replace_old_gradients(self):
        clean = copy.deepcopy(self.modules)
        dirty = copy.deepcopy(self.modules)
        clean_optimizers = make_optimizers(clean)
        dirty_optimizers = make_optimizers(dirty)
        for step in range(5):
            for module in dirty:
                for parameter in module.parameters():
                    parameter.grad = torch.full_like(parameter, 100.0 + step)
            x = self.x + 0.1 * step
            clean_metrics = adaptation_step(*clean, *clean_optimizers, x, self.y)
            dirty_metrics = adaptation_step(*dirty, *dirty_optimizers, x, self.y)
            self.assertEqual(clean_metrics, dirty_metrics)
            for clean_module, dirty_module in zip(clean, dirty):
                for clean_parameter, dirty_parameter in zip(clean_module.parameters(), dirty_module.parameters()):
                    torch.testing.assert_close(clean_parameter, dirty_parameter, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(clean_parameter).all())
                    self.assertIsNone(clean_parameter.grad.grad_fn)
                    self.assertFalse(clean_parameter.grad.requires_grad)
            for optimizer in clean_optimizers:
                for parameter in clean[0].parameters():
                    self.assertEqual(optimizer.state[parameter]["step"].item(), step + 1)
        self.assertEqual(clean[0].calls, 5)
        self.assertEqual(clean[0].norm.num_batches_tracked.item(), 5)


if __name__ == "__main__":
    unittest.main()
