"""Check recorded ledgers, hashes, plot/video/PDF files and model reproducibility."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pymupdf

from garl_lander.evaluation import evaluate
from garl_lander.experiment import load_agent, suite_runs
from garl_lander.io_utils import read_json, write_json
from garl_lander.seeding import episode_seeds, seed_everything


def verify_suite(suite: Path) -> dict:
    cfg = read_json(suite / "manifest.json")["config"]
    checked = []
    for directory, summary in suite_runs(suite):
        records = pd.read_csv(directory / "evaluation.csv")
        assert len(records) == cfg["experiment"]["final_eval_episodes"]
        assert summary["optimization_steps"] <= summary["allocated_steps"]
        assert summary["ga_steps"] + summary["ppo_steps"] == summary["optimization_steps"]
        assert summary["test_steps"] == records.length.sum()
        assert summary["total_steps"] == summary["optimization_steps"] + summary["selection_steps"] + summary["test_steps"]
        assert np.isclose(summary["mean_return"], records["return"].mean())
        seed_everything(summary["seed"], cfg["experiment"]["threads"])
        # A complete second evaluation is independent of training, and separately costed.
        repeated = evaluate(load_agent(directory, summary["agent"]), cfg, episode_seeds(cfg, "final"))
        np.testing.assert_allclose(records["return"], repeated["return"], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(records.length, repeated.length)
        checked.append({"agent": summary["agent"], "seed": summary["seed"],
                        "reevaluation_steps": int(repeated.length.sum()), "reproduced": True})
    ledger = {"suite": str(suite), "models": checked,
              "additional_verification_interactions": sum(r["reevaluation_steps"] for r in checked)}
    write_json(suite / "verification.json", ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_suite(args.run)
    if args.report:
        document = pymupdf.open(args.report)
        assert len(document) > 5
        assert all(len(p.get_text().strip()) > 0 for p in document)
        result["pdf_pages"] = len(document)
        destination = Path("tmp/pdfs/final")
        destination.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(document):
            page.get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(destination / f"page_{index+1:02d}.png")
        document.close()
    videos = []
    for path in Path("videos").glob("*/*.mp4"):
        reader = imageio.get_reader(path)
        try:
            frame = reader.get_data(0)
            assert frame.ndim == 3 and frame.shape[2] == 3
            videos.append({"path": str(path), "first_frame_shape": list(frame.shape)})
        finally:
            reader.close()
    result["decoded_videos"] = videos
    write_json(Path("results/artifact_verification.json"), result)
    print(f"Verified {len(result['models'])} models and {len(videos)} videos.")


if __name__ == "__main__":
    main()
