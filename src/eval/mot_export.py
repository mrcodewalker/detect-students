"""Export and read utilities for MOTChallenge evaluation format."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union


@dataclass
class MOTRecord:
    """Single tracklet record in MOTChallenge format.

    Attributes:
        frame: 1-based frame number.
        track_id: Integer tracking identity.
        bbox: Bounding box in (x1, y1, x2, y2) format.
        conf: Confidence score.
    """

    frame: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    conf: float


def export_mot_format(
    records: Sequence[Tuple[int, int, Tuple[float, float, float, float], float]],
    output_path: Union[str, Path],
    is_zero_based: bool = True,
) -> None:
    """Write tracker predictions to a MOTChallenge standard text file.

    Format specification (comma-delimited):
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1

    Note:
        MOTChallenge requires 1-based frame indexing. If input records use 0-based
        indices (default in video decoders), set is_zero_based=True to add 1.

    Args:
        records: Sequence of (frame_index, track_id, (x1, y1, x2, y2), conf).
        output_path: File path to save the MOTChallenge txt file.
        is_zero_based: Whether input frame indices are 0-based.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for frame_idx, track_id, (x1, y1, x2, y2), conf in records:
            frame_num = int(frame_idx + 1) if is_zero_based else int(frame_idx)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            f.write(
                f"{frame_num},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
            )


def read_mot_format(
    input_path: Union[str, Path],
    to_zero_based: bool = False,
) -> List[MOTRecord]:
    """Parse a MOTChallenge format text file.

    Args:
        input_path: Path to MOTChallenge formatted txt file.
        to_zero_based: If True, subtracts 1 from frame numbers to convert to 0-based indices.

    Returns:
        List of MOTRecord instances with (x1, y1, x2, y2) bounding boxes.
    """
    in_file = Path(input_path).resolve()
    if not in_file.is_file():
        raise FileNotFoundError(f"MOT file not found: {in_file}")

    records: List[MOTRecord] = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue

            parts = [p.strip() for p in clean_line.split(",")]
            if len(parts) < 7:
                continue

            raw_frame = int(float(parts[0]))
            frame = raw_frame - 1 if to_zero_based else raw_frame
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])

            bbox = (x, y, x + w, y + h)
            records.append(MOTRecord(frame=frame, track_id=track_id, bbox=bbox, conf=conf))

    return records

