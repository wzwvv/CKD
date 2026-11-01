# Author: Ziwei Wang

import json
import csv
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import fractional_matrix_power

from ckd.data import EEGDataset, _legacy_subjects, _ordered_metadata, align_online, load_dataset, prepare_dataset, save_dataset, split_loso


class DataTests(unittest.TestCase):
    def test_original_moabb_csv_metadata_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "BNCI2014004"
            folder.mkdir()
            np.save(folder / "X.npy", np.ones((4, 3, 1000), dtype=np.float32))
            np.save(folder / "labels.npy", np.array(["left_hand", "right_hand"] * 2))
            with (folder / "meta.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject", "session", "run"])
                writer.writeheader()
                writer.writerows({"subject": subject, "session": session, "run": "0"} for subject, session in [(2, "0train"), (2, "1test"), (7, "0train"), (7, "1test")])
            dataset, _ = prepare_dataset(temporary, "BNCI2014004", sessions="first", align="none")
            np.testing.assert_array_equal(dataset.subjects, [2, 7])
            np.testing.assert_array_equal(dataset.sessions, [1, 1])

    def test_ordered_metadata_tracks_each_session_once(self):
        subjects, sessions, trials = _ordered_metadata([[2, 3], [1, 2]])
        np.testing.assert_array_equal(subjects, [1, 1, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(sessions, [1, 1, 2, 2, 2, 1, 2, 2])
        np.testing.assert_array_equal(trials, [1, 2, 1, 2, 3, 1, 1, 2])

    def test_sessions_are_unique_and_first_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "BNCI2014004"
            folder.mkdir()
            subjects = np.repeat([1, 2], 4)
            sessions = np.tile([1, 1, 2, 2], 2)
            x = np.repeat(np.arange(8, dtype=np.float32)[:, None, None], 3, axis=1)
            x = np.repeat(x, 1000, axis=2)
            np.save(folder / "X.npy", x)
            np.save(folder / "labels.npy", np.tile(["left_hand", "right_hand"], 4))
            np.save(folder / "subjects.npy", subjects)
            np.save(folder / "sessions.npy", sessions)
            all_data, _ = prepare_dataset(root, "BNCI2014004", sessions="all", align="none")
            first_data, _ = prepare_dataset(root, "BNCI2014004", sessions="first", align="none")
            np.testing.assert_array_equal(all_data.x[:, 0, 0], np.arange(8))
            np.testing.assert_array_equal(first_data.x[:, 0, 0], [0, 1, 4, 5])
            self.assertEqual(len(np.unique(np.column_stack((all_data.subjects, all_data.sessions, all_data.trial_ids)), axis=0)), 8)

    def test_loso_excludes_the_complete_subject(self):
        rng = np.random.default_rng(4)
        dataset = EEGDataset(rng.normal(size=(6, 3, 1000)), [0, 1] * 3, [1, 1, 2, 2, 3, 3], "BNCI2014004")
        train_x, train_y, test_x, test_y = split_loso(dataset, 2)
        np.testing.assert_array_equal(train_x, dataset.x[[0, 1, 4, 5]])
        np.testing.assert_array_equal(test_x, dataset.x[[2, 3]])
        self.assertEqual((len(train_y), len(test_y)), (4, 2))
        with self.assertRaises(ValueError):
            split_loso(dataset, 9)

    def test_ea_online_matches_running_covariance_and_subject_reset(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(6, 3, 1000))
        subjects = np.repeat([1, 2], 3)
        expected = []
        for subject in [1, 2]:
            reference = 0
            for count, trial in enumerate(x[subjects == subject]):
                reference = (reference * count + np.cov(trial)) / (count + 1)
                expected.append(fractional_matrix_power(reference, -0.5) @ trial)
        result = align_online(x, subjects)
        np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-6)
        changed = x.copy()
        changed[2] *= 9
        np.testing.assert_allclose(align_online(changed, subjects)[:2], result[:2])
        np.testing.assert_allclose(align_online(changed, subjects)[3:], result[3:])

    def test_metadata_cache_round_trip(self):
        rng = np.random.default_rng(2)
        dataset = EEGDataset(rng.normal(size=(4, 3, 1000)), [0, 1, 0, 1], [2, 2, 7, 7], "BNCI2014004", sessions=[1, 2, 1, 2], trial_ids=[1, 1, 1, 1], metadata={"provenance": "raw_trial_metadata"})
        with tempfile.TemporaryDirectory() as temporary:
            save_dataset(dataset, temporary)
            loaded = load_dataset(temporary, "BNCI2014004", align=False)
            np.testing.assert_array_equal(loaded.x, dataset.x)
            np.testing.assert_array_equal(loaded.subjects, [2, 2, 7, 7])
            np.testing.assert_array_equal(loaded.sessions, dataset.sessions)
            self.assertEqual(loaded.metadata["provenance"], "raw_trial_metadata")
            with self.assertRaises(FileExistsError):
                save_dataset(dataset, temporary)

    def test_legacy_cache_requires_matching_length_and_reports_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "BNCI2014004" / "seta"
            folder.mkdir(parents=True)
            np.save(folder / "X.npy", np.zeros((5600, 3, 1000), dtype=np.float32))
            np.save(folder / "y.npy", np.tile([0, 1], 2800))
            with warnings.catch_warnings(record=True) as caught:
                loaded = load_dataset(temporary, "BNCI2014004", align=False)
            self.assertEqual(loaded.metadata["provenance"], "legacy_shape_only")
            self.assertEqual(len(caught), 1)
            self.assertTrue(np.all(loaded.sessions == 0))
            np.save(folder / "y.npy", np.zeros(10, dtype=np.int64))
            with self.assertRaises(ValueError):
                load_dataset(temporary, "BNCI2014004", align=False)

    def test_raw_order_length_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "BNCI2014001"
            folder.mkdir()
            np.save(folder / "X.npy", np.zeros((3, 22, 1000), dtype=np.float32))
            np.save(folder / "labels.npy", np.array(["left_hand", "right_hand", "left_hand"]))
            with self.assertRaises(ValueError):
                prepare_dataset(temporary, "BNCI2014001", align="none")

    def test_duplicate_trial_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            EEGDataset(np.zeros((2, 3, 1000)), [0, 1], [1, 1], "BNCI2014004", sessions=[1, 1], trial_ids=[2, 2])

    def test_mi1_default_and_explicit_exclusion_retain_natural_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "MI1"
            folder.mkdir()
            np.save(folder / "X.npy", np.zeros((14, 59, 1000), dtype=np.float32))
            np.save(folder / "labels.npy", np.tile([0, 1], 7))
            np.save(folder / "subjects.npy", np.repeat(np.arange(1, 8), 2))
            with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump({"sampling_rate": 250}, handle)
            dataset, _ = prepare_dataset(temporary, "MI1", align="none")
            np.testing.assert_array_equal(np.unique(dataset.subjects), [2, 3, 4, 5, 7])
            dataset, _ = prepare_dataset(temporary, "MI1", align="none", mi1_exclude_subjects=[1, 5])
            np.testing.assert_array_equal(np.unique(dataset.subjects), [2, 3, 4, 6, 7])
            with self.assertRaises(ValueError):
                prepare_dataset(temporary, "MI1", align="none", mi1_exclude_subjects=[1, 9])

    def test_mi1_legacy_ids_follow_default_selection(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subjects = _legacy_subjects("MI1", 1000)
        np.testing.assert_array_equal(subjects, np.repeat([2, 3, 4, 5, 7], 200))

    def test_weibo_resampling_and_left_right_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "Weibo2014"
            folder.mkdir()
            np.save(folder / "X.npy", np.ones((3, 60, 800), dtype=np.float32))
            np.save(folder / "labels.npy", np.array(["left_hand", "right_hand", "feet"]))
            np.save(folder / "subjects.npy", [1, 1, 1])
            np.save(folder / "sessions.npy", [1, 2, 2])
            dataset, _ = prepare_dataset(temporary, "Weibo2014", align="none")
            self.assertEqual(dataset.x.shape, (2, 60, 1000))
            np.testing.assert_array_equal(dataset.y, [0, 1])
            np.testing.assert_array_equal(dataset.sessions, [1, 2])
            np.testing.assert_allclose(dataset.x, 1, atol=1e-6)

    def test_highgamma_subject_files_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "HighGamma" / "seta"
            folder.mkdir(parents=True)
            for index in range(14):
                np.save(folder / f"X{index}.npy", np.full((2, 128, 1000), index + 1, dtype=np.float32))
                np.save(folder / f"y{index}.npy", [0, 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dataset = load_dataset(temporary, "HighGamma", align=False)
            np.testing.assert_array_equal(dataset.subjects, np.repeat(np.arange(1, 15), 2))
            np.testing.assert_array_equal(dataset.x[:, 0, 0], dataset.subjects)


if __name__ == "__main__":
    unittest.main()
