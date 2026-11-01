# Author: Ziwei Wang

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ckd import cli
from ckd.data import CHANNELS, DATASETS, load_dataset
from ckd.training import evaluate_model


class PipelineTests(unittest.TestCase):
    def test_training_evaluation_and_output_safety(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, output_root = root / "data", root / "runs"
            rng = np.random.default_rng(1729)
            for name in DATASETS:
                folder = data_root / name / "seta"
                folder.mkdir(parents=True)
                np.save(folder / "X.npy", rng.standard_normal((4, CHANNELS[name], 1000)).astype(np.float32))
                np.save(folder / "y.npy", np.asarray([0, 1, 0, 1], dtype=np.int64))
                np.save(folder / "subjects.npy", np.asarray([1, 1, 2, 2], dtype=np.int64))
                np.save(folder / "sessions.npy", np.ones(4, dtype=np.int64))
                np.save(folder / "trial_ids.npy", np.asarray([1, 2, 1, 2], dtype=np.int64))
                (folder / "metadata.json").write_text(json.dumps({"sampling_rate": 250}), encoding="utf-8")
            hashes_before = {str(path.relative_to(data_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in data_root.rglob("*") if path.is_file()}
            common = ["--data-root", str(data_root), "--output-root", str(output_root), "--device", "cpu", "--seeds", "1", "--epochs", "1", "--batch-size", "2", "--align", "none"]
            source, target = "BNCI2014001", "BNCI2014004"
            commands = {
                "teacher_single": common + ["--sources", source],
                "teacher_multi": common + ["--targets", target],
                "student_single": common + ["--sources", source, "--targets", target, "--subjects", "1"],
                "student_cross": common + ["--targets", target, "--subjects", "1"],
            }
            self.assertEqual(cli._parser("teacher_single").parse_args(common).sources, list(DATASETS))
            self.assertEqual(cli._parser("teacher_multi").parse_args(common).targets, list(DATASETS))
            for stage in ("teacher_single", "teacher_multi"):
                with self.subTest(stage=stage), redirect_stdout(io.StringIO()):
                    with patch("ckd.cli.load_dataset", wraps=load_dataset) as loader:
                        cli.main(stage, commands[stage])
                    loaded = [call.args[1] for call in loader.call_args_list]
                    expected = [source] if stage == "teacher_single" else [name for name in DATASETS if name != target]
                    self.assertEqual(loaded, expected)
                    self.assertNotIn(target, loaded)
                    scenario, key = ("single", source) if stage == "teacher_single" else ("multi", target)
                    state = torch.load(output_root / "teachers" / scenario / key / "seed_1" / "teacher.pt", weights_only=True)
                    self.assertEqual(state["source_datasets"], expected)
                    self.assertEqual(list(state["embeddings"]), expected)
                    self.assertNotIn(target, state["embeddings"])
            for stage, scenario, task in (("student_single", "single", f"{source}_to_{target}"), ("student_cross", "multi", f"all_to_{target}")):
                with self.subTest(stage=stage), redirect_stdout(io.StringIO()):
                    cli.main(stage, commands[stage])
                    folder = output_root / "students" / scenario / task / "seed_1" / "subject_1"
                    checkpoint = folder / "student.pt"
                    state = torch.load(checkpoint, weights_only=True)
                    self.assertEqual(state["stage"], "student")
                    self.assertEqual(state["target_dataset"], target)
                    self.assertEqual(state["held_out_subject"], 1)
                    self.assertNotIn(target, state["source_datasets"])
                    self.assertEqual({key for key, value in state.items() if isinstance(value, dict)}, {"model_config", "embedding", "encoder", "classifier"})
                    self.assertFalse(any(key.startswith("teacher") for key in state))
                    self.assertTrue(all(state[name] for name in ("embedding", "encoder", "classifier")))
                    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(metrics["n_trials"], 2)
                    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                        labels = saved["labels"].copy()
                        probabilities = saved["probabilities"].copy()
                        np.testing.assert_array_equal(saved["subject"], [1, 1])
                    np.testing.assert_array_equal(labels, [0, 1])
                    self.assertEqual(probabilities.shape, (2, 2))
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)
                    self.assertEqual(float((probabilities.argmax(axis=1) == labels).mean() * 100), metrics["accuracy"])
                    summary = json.loads((output_root / "results" / f"{scenario}_{task}.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["accuracy_mean"], metrics["accuracy"])
                    self.assertEqual(summary["accuracy_std"], 0)
                    self.assertEqual(summary["subject_results"], [metrics])
                    captured = []
                    def evaluate(*args, **kwargs):
                        result = evaluate_model(*args, **kwargs)
                        captured.append(result)
                        return result
                    stdout = io.StringIO()
                    with patch("ckd.cli.evaluate_model", side_effect=evaluate), redirect_stdout(stdout):
                        cli.evaluate_main(["--checkpoint", str(checkpoint), "--data-root", str(data_root), "--device", "cpu", "--batch-size", "2"])
                    restored_metrics = json.loads(stdout.getvalue())
                    self.assertEqual(restored_metrics, {"accuracy": metrics["accuracy"], "n_trials": metrics["n_trials"]})
                    self.assertEqual(len(captured), 1)
                    np.testing.assert_array_equal(captured[0][1], labels)
                    np.testing.assert_allclose(captured[0][2], probabilities, rtol=0, atol=0)
            tainted_path = root / "tainted_teacher.pt"
            tainted = torch.load(output_root / "teachers" / "single" / source / "seed_1" / "teacher.pt", weights_only=True)
            tainted["source_datasets"] = [source, target]
            torch.save(tainted, tainted_path)
            rejected_output = root / "rejected_runs"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "target dataset must not be included"):
                cli.main("student_single", commands["student_single"] + ["--teacher-checkpoint", str(tainted_path), "--output-root", str(rejected_output)])
            self.assertFalse(rejected_output.exists())
            for stage, arguments in commands.items():
                with self.subTest(repeated_stage=stage), redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileExistsError):
                        cli.main(stage, arguments)
            hashes_after = {str(path.relative_to(data_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in data_root.rglob("*") if path.is_file()}
            self.assertEqual(hashes_before, hashes_after)


if __name__ == "__main__":
    unittest.main()
