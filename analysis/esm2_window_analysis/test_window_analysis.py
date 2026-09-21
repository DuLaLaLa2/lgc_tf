"""ESM2 窗口分组分析的边界与一致性测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.esm2_window_analysis.analyze_window_groups import (
    align_and_validate,
    bootstrap_delta_mcc,
    build_group_comparison,
    build_metadata,
    build_per_protein_comparison,
    compute_per_protein,
    correlation_analysis,
    get_esm2_window_count,
    read_predictions,
    run_boundary_tests,
    summarize_groups,
    write_scientific_summary,
)


class WindowBoundaryTests(unittest.TestCase):
    """验证原始滑窗循环的关键边界。"""

    def test_required_boundaries(self) -> None:
        run_boundary_tests()

    def test_expected_counts(self) -> None:
        expected = {
            500: 1,
            1000: 1,
            1022: 1,
            1023: 2,
            1203: 2,
            1343: 2,
            1500: 2,
            1922: 2,
            1923: 3,
            2000: 3,
            2485: 3,
            2822: 3,
            2823: 4,
        }
        self.assertEqual({length: get_esm2_window_count(length) for length in expected}, expected)


class PredictionValidationTests(unittest.TestCase):
    """验证阈值、标签和跨模型对齐检查。"""

    def make_frame(self, probability_shift: float = 0.0) -> pd.DataFrame:
        rows = []
        for protein_id, length in (("p1", 3), ("p2", 4)):
            for residue_index in range(1, length + 1):
                label = residue_index % 2
                rows.append(
                    {
                        "sample_id": protein_id,
                        "protein_length": length,
                        "residue_index": residue_index,
                        "label": label,
                        "probability": float(np.clip(0.2 + 0.6 * label + probability_shift, 0, 1)),
                        "threshold": 0.5,
                    }
                )
        return pd.DataFrame(rows)

    def test_alignment_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer_path = Path(directory) / "transformer.csv"
            bimamba_path = Path(directory) / "bimamba.csv"
            self.make_frame().to_csv(transformer_path, index=False)
            self.make_frame(0.05).to_csv(bimamba_path, index=False)
            transformer, _ = read_predictions(transformer_path, "Transformer", None)
            bimamba, _ = read_predictions(bimamba_path, "BiMamba", None)
            transformer, bimamba = align_and_validate(transformer, bimamba)
            metadata = build_metadata(transformer)
            self.assertEqual(metadata["protein_id"].tolist(), ["p1", "p2"])
            self.assertEqual(metadata["esm2_window_count"].tolist(), [1, 1])

    def test_label_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer_path = Path(directory) / "transformer.csv"
            bimamba_path = Path(directory) / "bimamba.csv"
            transformer_raw = self.make_frame()
            bimamba_raw = self.make_frame()
            bimamba_raw.loc[0, "label"] = 1 - bimamba_raw.loc[0, "label"]
            transformer_raw.to_csv(transformer_path, index=False)
            bimamba_raw.to_csv(bimamba_path, index=False)
            transformer, _ = read_predictions(transformer_path, "Transformer", None)
            bimamba, _ = read_predictions(bimamba_path, "BiMamba", None)
            with self.assertRaises(AssertionError):
                align_and_validate(transformer, bimamba)

    def test_complete_statistical_pipeline(self) -> None:
        """用覆盖 1/2/3/4 窗口的合成数据验证完整统计链路。"""
        rows_t = []
        rows_m = []
        for protein_id, length in (("p1", 8), ("p2", 1023), ("p3", 1923), ("p4", 2823)):
            for residue_index in range(1, length + 1):
                label = residue_index % 2
                base = 0.7 if label else 0.3
                common = {
                    "sample_id": protein_id,
                    "protein_length": length,
                    "residue_index": residue_index,
                    "label": label,
                    "threshold": 0.5,
                }
                rows_t.append({**common, "probability": base})
                rows_m.append({**common, "probability": min(max(base + (0.05 if label else -0.05), 0), 1)})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer_path = root / "transformer.csv"
            bimamba_path = root / "bimamba.csv"
            pd.DataFrame(rows_t).to_csv(transformer_path, index=False)
            pd.DataFrame(rows_m).to_csv(bimamba_path, index=False)
            transformer, transformer_threshold = read_predictions(transformer_path, "Transformer", None)
            bimamba, bimamba_threshold = read_predictions(bimamba_path, "BiMamba", None)
            transformer, bimamba = align_and_validate(transformer, bimamba)
            metadata = build_metadata(transformer)
            warning_messages: list[str] = []
            transformer_per = compute_per_protein(
                transformer, transformer_threshold, "transformer", warning_messages
            )
            bimamba_per = compute_per_protein(
                bimamba, bimamba_threshold, "bimamba", warning_messages
            )
            per_protein = build_per_protein_comparison(metadata, transformer_per, bimamba_per)
            transformer_groups = summarize_groups(
                transformer, metadata, transformer_per, transformer_threshold, "Transformer", "transformer"
            )
            bimamba_groups = summarize_groups(
                bimamba, metadata, bimamba_per, bimamba_threshold, "BiMamba", "bimamba"
            )
            comparison = build_group_comparison(transformer_groups, bimamba_groups)
            bootstrap = bootstrap_delta_mcc(
                metadata,
                transformer,
                bimamba,
                transformer_threshold,
                bimamba_threshold,
                n_bootstrap=20,
                seed=42,
            )
            correlations = correlation_analysis(per_protein)
            summary_path = root / "summary.md"
            write_scientific_summary(
                summary_path,
                comparison,
                bootstrap,
                correlations,
                transformer_threshold,
                bimamba_threshold,
            )
            self.assertEqual(metadata["window_group"].tolist(), ["1-window", "2-window", "3-window", ">=4-window"])
            self.assertEqual(len(comparison), 4)
            self.assertEqual(len(bootstrap), 4)
            self.assertEqual(len(correlations), 3)
            self.assertTrue(summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
