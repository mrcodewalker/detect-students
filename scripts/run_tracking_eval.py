"""CLI tool to evaluate multi-object trackers against ground-truth using official TrackEval metrics."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tabulate import tabulate

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.eval.mot_export import read_mot_format
from src.eval.tracking_evaluator import evaluate_sequences


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate multi-object tracking outputs against ground truth using TrackEval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt-folder",
        "-g",
        type=str,
        required=True,
        help="Path to folder containing ground truth MOTChallenge txt files.",
    )
    parser.add_argument(
        "--trackers-folder",
        "-t",
        type=str,
        required=True,
        help="Path to folder containing tracker outputs (either tracker subfolders or tracker txt files).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/eval",
        help="Directory to save evaluation summaries and comparison table.",
    )
    return parser.parse_args()


def discover_ground_truths(gt_folder: Path) -> Dict[str, Path]:
    """Discover all ground-truth sequence files in the given folder."""
    gt_map: Dict[str, Path] = {}
    for p in gt_folder.rglob("*.txt"):
        # Sequence name is parent folder name if named gt.txt, else file stem
        seq_name = p.parent.name if p.stem.lower() == "gt" else p.stem
        gt_map[seq_name] = p
    return gt_map


def discover_tracker_outputs(trackers_folder: Path, available_seqs: List[str]) -> Dict[str, Dict[str, Path]]:
    """Discover tracker predictions organized by tracker name and sequence."""
    tracker_map: Dict[str, Dict[str, Path]] = {}

    # Check for tracker subfolders (e.g. trackers_folder/sort/seq1.txt)
    subdirs = [d for d in trackers_folder.iterdir() if d.is_dir()]
    if subdirs:
        for d in subdirs:
            tracker_name = d.name
            tracker_map[tracker_name] = {}
            for f in d.glob("*.txt"):
                tracker_map[tracker_name][f.stem] = f

    # Check for direct files in trackers_folder (e.g. trackers_folder/bytetrack_tracks_mot.txt or trackers_folder/sort.txt)
    direct_files = [f for f in trackers_folder.glob("*.txt") if f.is_file()]
    if direct_files:
        for f in direct_files:
            # Derive clean tracker name
            stem = f.stem
            for suffix in ["_tracks_mot", "_tracks", "_mot"]:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            tracker_name = stem
            if tracker_name not in tracker_map:
                tracker_map[tracker_name] = {}

            # If there's only one GT sequence or file matches sequence
            if len(available_seqs) == 1:
                tracker_map[tracker_name][available_seqs[0]] = f
            elif f.stem in available_seqs:
                tracker_map[tracker_name][f.stem] = f

    return tracker_map


def main() -> int:
    """Execute evaluation routine."""
    args = parse_args()
    gt_folder = Path(args.gt_folder).resolve()
    trackers_folder = Path(args.trackers_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_folder.is_dir():
        print(f"Error: Ground truth directory not found: {gt_folder}", file=sys.stderr)
        return 1
    if not trackers_folder.is_dir():
        print(f"Error: Trackers directory not found: {trackers_folder}", file=sys.stderr)
        return 2

    gt_map = discover_ground_truths(gt_folder)
    if not gt_map:
        print(f"Error: No .txt ground-truth files found in {gt_folder}", file=sys.stderr)
        return 3

    available_seqs = sorted(list(gt_map.keys()))
    tracker_map = discover_tracker_outputs(trackers_folder, available_seqs)

    if not tracker_map:
        print(f"Error: No tracker output files found in {trackers_folder}", file=sys.stderr)
        return 4

    print("==================================================")
    print("Multi-Object Tracking Evaluation (TrackEval)")
    print("==================================================")
    print(f"Ground Truth Folder : {gt_folder} ({len(gt_map)} sequence(s))")
    print(f"Trackers Folder     : {trackers_folder} ({len(tracker_map)} tracker(s))")
    print(f"Output Directory    : {output_dir}")
    print("--------------------------------------------------")

    comparison_rows: List[Dict[str, Any]] = []

    for tracker_name, seq_files in sorted(tracker_map.items()):
        seq_metric_rows: List[Dict[str, Any]] = []

        all_gt_records = []
        all_trk_records = []

        for seq_name in available_seqs:
            if seq_name not in seq_files:
                continue

            gt_file = gt_map[seq_name]
            trk_file = seq_files[seq_name]

            gt_recs = read_mot_format(gt_file, to_zero_based=False)
            trk_recs = read_mot_format(trk_file, to_zero_based=False)

            metrics = evaluate_sequences(gt_recs, trk_recs)
            metrics_entry = {"Sequence": seq_name, **metrics}
            seq_metric_rows.append(metrics_entry)

            all_gt_records.extend(gt_recs)
            all_trk_records.extend(trk_recs)

        if not seq_metric_rows:
            print(f"Warning: No matching sequences found for tracker '{tracker_name}'.")
            continue

        # Save individual tracker summary CSV
        tracker_csv = output_dir / f"{tracker_name}_summary.csv"
        with open(tracker_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(seq_metric_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seq_metric_rows)
        print(f"Saved tracker summary: {tracker_csv.name}")

        # Compute overall / aggregated metrics for this tracker
        overall_metrics = evaluate_sequences(all_gt_records, all_trk_records)
        comparison_entry = {
            "Tracker": tracker_name,
            "HOTA (%)": f"{overall_metrics['HOTA'] * 100:.2f}",
            "MOTA (%)": f"{overall_metrics['MOTA'] * 100:.2f}",
            "IDF1 (%)": f"{overall_metrics['IDF1'] * 100:.2f}",
            "IDSW": int(overall_metrics["IDSW"]),
            "Frag": int(overall_metrics["Frag"]),
            "DetA (%)": f"{overall_metrics['DetA'] * 100:.2f}",
            "AssA (%)": f"{overall_metrics['AssA'] * 100:.2f}",
        }
        comparison_rows.append(comparison_entry)

    # Save and display combined comparison table
    if comparison_rows:
        comp_csv = output_dir / "tracking_comparison.csv"
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(comparison_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        print("\n==================================================")
        print("Combined Tracker Benchmark Comparison")
        print("==================================================")
        print(tabulate(comparison_rows, headers="keys", tablefmt="github"))
        print("==================================================")
        print(f"Combined comparison saved to: {comp_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
