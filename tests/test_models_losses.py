# Author: Ziwei Wang

import ast
import os
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ckd.losses import contrastive_loss, distillation_loss
from ckd.models import CKDModel, ModelConfig, build_modules, build_multisource_modules


class ModelTests(unittest.TestCase):
    def test_shapes(self):
        for samples in (1000, 1001, 1125, 1008):
            config = ModelConfig(channels=3, samples=samples)
            model = CKDModel(*build_modules(config)).eval()
            features, logits = model(torch.randn(2, 1, config.channels, samples))
            self.assertEqual(features.shape, (2, config.f2 * (samples // 4 // 8)))
            self.assertEqual(logits.shape, (2, config.num_classes))

    def test_construction_preserves_initial_batch_norm(self):
        model = CKDModel(*build_modules(ModelConfig()))
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual(module.num_batches_tracked.item(), 0)
                torch.testing.assert_close(module.running_mean, torch.zeros_like(module.running_mean))
                torch.testing.assert_close(module.running_var, torch.ones_like(module.running_var))

    def test_invalid_configuration(self):
        for kwargs in ({"channels": 0}, {"samples": 31}, {"sampling_rate": 3}, {"f1": 0}, {"num_classes": 1}):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_multisource_shapes(self):
        configs = {"source_a": ModelConfig(channels=3), "source_b": ModelConfig(channels=22)}
        embeddings, encoder, classifier = build_multisource_modules(configs)
        self.assertEqual(list(embeddings), list(configs))
        for name, config in configs.items():
            features, logits = classifier(encoder(embeddings[name](torch.randn(2, 1, config.channels, 1000))))
            self.assertEqual(features.shape, (2, 248))
            self.assertEqual(logits.shape, (2, 2))
        with self.assertRaises(ValueError):
            build_multisource_modules({})
        with self.assertRaises(ValueError):
            build_multisource_modules({"a": ModelConfig(), "b": ModelConfig(f2=16)})

    @unittest.skipUnless(os.environ.get("CKD_ORIGINAL_SOURCE"), "CKD_ORIGINAL_SOURCE is not set")
    def test_original_model_equivalence(self):
        source = Path(os.environ["CKD_ORIGINAL_SOURCE"])
        scope = {"torch": torch, "nn": nn}
        for relative, selected in (("models/EEGNet.py", {"EEGNet_block1", "EEGNet_block2"}), ("models/FC.py", {"FC_xy"})):
            tree = ast.parse((source / relative).read_text(encoding="utf-8"))
            nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in selected]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / relative), "exec"), scope)
        config = ModelConfig(channels=3, samples=1000)
        kwargs = dict(n_classes=2, Chans=3, Samples=1000, kernLenght=125, F1=4, D=2, F2=8, dropoutRate=0.25, norm_rate=0.5)
        torch.manual_seed(47)
        old_embedding = scope["EEGNet_block1"](**kwargs)
        kwargs["Chans"] = -1
        old_encoder = scope["EEGNet_block2"](**kwargs)
        old_classifier = scope["FC_xy"](248, 2)
        old_rng = torch.get_rng_state()
        torch.manual_seed(47)
        modules = build_modules(config)
        torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
        for old_module, new_module in zip((old_embedding, old_encoder, old_classifier), modules):
            self.assertEqual(tuple(old_module.state_dict()), tuple(new_module.state_dict()))
            for key, tensor in old_module.state_dict().items():
                torch.testing.assert_close(tensor, new_module.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 1, 3, 1000)
        for training in (False, True):
            for module in (old_embedding, old_encoder, old_classifier, *modules):
                module.train(training)
            torch.manual_seed(151)
            old_features, old_logits = old_classifier(old_encoder(old_embedding(x)))
            torch.manual_seed(151)
            new_features, new_logits = CKDModel(*modules)(x)
            torch.testing.assert_close(old_features, new_features, rtol=0, atol=0)
            torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
        torch.manual_seed(83)
        kwargs["Chans"] = 3
        old_embeddings = [scope["EEGNet_block1"](**kwargs)]
        kwargs["Chans"] = 22
        old_embeddings.append(scope["EEGNet_block1"](**kwargs))
        kwargs["Chans"] = -1
        old_encoder = scope["EEGNet_block2"](**kwargs)
        old_classifier = scope["FC_xy"](248, 2)
        old_rng = torch.get_rng_state()
        torch.manual_seed(83)
        embeddings, encoder, classifier = build_multisource_modules({"a": config, "b": ModelConfig(channels=22)})
        torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
        for old_module, new_module in zip((*old_embeddings, old_encoder, old_classifier), (*embeddings.values(), encoder, classifier)):
            self.assertEqual(tuple(old_module.state_dict()), tuple(new_module.state_dict()))
            for key, tensor in old_module.state_dict().items():
                torch.testing.assert_close(tensor, new_module.state_dict()[key], rtol=0, atol=0)


class LossTests(unittest.TestCase):
    def test_kd_formula_and_gradient_direction(self):
        torch.manual_seed(19)
        student = torch.randn(5, 2, requires_grad=True)
        teacher = torch.randn(5, 2, requires_grad=True)
        temperature = 4.0
        loss = distillation_loss(student, teacher, temperature)
        p_teacher = F.softmax(teacher.detach() / temperature, dim=1)
        expected = (p_teacher * (p_teacher.log() - F.log_softmax(student / temperature, dim=1))).sum() / 5 * temperature ** 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_contrastive_formula_and_gradient_direction(self):
        torch.manual_seed(21)
        student = torch.randn(5, 7, requires_grad=True)
        teacher = torch.randn(5, 7, requires_grad=True)
        loss = contrastive_loss(student, teacher, 0.2)
        expected = F.cross_entropy(F.normalize(student, dim=1) @ F.normalize(teacher.detach(), dim=1).T / 0.2, torch.arange(5))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_positive_pair_permutation(self):
        features = torch.eye(6)
        matched = contrastive_loss(features, features)
        shuffled = contrastive_loss(features, features.roll(1, dims=0))
        self.assertLess(matched.item(), shuffled.item())

    def test_single_sample_batch(self):
        features = torch.randn(1, 4, requires_grad=True)
        teacher = torch.randn(1, 4)
        self.assertEqual(contrastive_loss(features, teacher).item(), 0.0)
        self.assertTrue(torch.isfinite(distillation_loss(features, teacher)))

    def test_invalid_shapes_and_temperatures(self):
        for loss in (distillation_loss, contrastive_loss):
            for temperature in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    loss(torch.randn(2, 3), torch.randn(2, 3), temperature)
            for first, second in ((torch.randn(2, 3, 1), torch.randn(2, 3, 1)), (torch.randn(2, 3), torch.randn(2, 4)), (torch.empty(0, 3), torch.empty(0, 3)), (torch.empty(2, 0), torch.empty(2, 0))):
                with self.assertRaises(ValueError):
                    loss(first, second)


if __name__ == "__main__":
    unittest.main()
