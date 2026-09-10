#!/usr/bin/env python3
"""Build a distance/angle/approach/effort success map from stair diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _bin(value: float, width: float) -> float:
    return math.floor(value / width) * width


def _mean(total: float, count: int) -> float:
    return total / count if count else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_csv", help="stair_attempts.csv produced by training")
    parser.add_argument("--output", help="output CSV (default: next to attempt CSV)")
    parser.add_argument("--distance-bin", type=_positive_float, default=0.05, help="distance bin width [m]")
    parser.add_argument("--angle-bin", type=_positive_float, default=5.0, help="absolute angle bin width [deg]")
    parser.add_argument("--speed-bin", type=_positive_float, default=0.10, help="approach speed bin width [m/s]")
    parser.add_argument("--torque-bin", type=_positive_float, default=2.0, help="peak shoulder torque bin [Nm]")
    parser.add_argument("--min-attempts", type=int, default=5, help="minimum samples shown in terminal ranking")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.attempt_csv)), "stair_capability_map.csv")
    groups: dict[tuple[float, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    accepted = 0
    skipped = 0

    with open(args.attempt_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            distance = float(row["oracle_distance_along_heading_m"])
            angle = abs(float(row["oracle_alignment_error_deg"]))
            speed = float(row["start_vx_b_mps"])
            shoulder_peak = max(
                float(row["peak_abs_left_shoulder_joint_torque_nm"]),
                float(row["peak_abs_right_shoulder_joint_torque_nm"]),
            )
            terrain_level = int(row["terrain_level"])
            if not all(math.isfinite(v) for v in (distance, angle, speed, shoulder_peak)):
                skipped += 1
                continue
            key = (
                terrain_level,
                _bin(distance, args.distance_bin),
                _bin(angle, args.angle_bin),
                _bin(speed, args.speed_bin),
                _bin(shoulder_peak, args.torque_bin),
            )
            group = groups[key]
            group["attempts"] += 1
            group["successes"] += int(row["success"])
            group["climbed"] += int(row["climbed"])
            group["stable"] += int(row["stable_landing"])
            group["step_height"] += float(row["terrain_step_height_m"])
            group["step_width"] += float(row["terrain_step_width_m"])
            group["distance"] += distance
            group["angle"] += angle
            group["speed"] += speed
            group["shoulder_peak"] += shoulder_peak
            group["shoulder_impulse"] += (
                float(row["abs_left_shoulder_joint_torque_impulse_nms"])
                + float(row["abs_right_shoulder_joint_torque_impulse_nms"])
            )
            group["contact_impulse"] += float(row["wheel_contact_impulse_ns"])
            group["max_vz"] += float(row["max_base_vz_w_mps"])
            group["ground_gain"] += float(row["max_ground_gain_m"])
            accepted += 1

    fieldnames = [
        "terrain_level",
        "distance_bin_start_m",
        "angle_abs_bin_start_deg",
        "approach_speed_bin_start_mps",
        "peak_shoulder_torque_bin_start_nm",
        "attempts",
        "successes",
        "success_rate",
        "climb_rate",
        "stable_landing_rate",
        "mean_step_height_m",
        "mean_step_width_m",
        "mean_distance_m",
        "mean_abs_angle_deg",
        "mean_approach_speed_mps",
        "mean_peak_shoulder_torque_nm",
        "mean_total_shoulder_torque_impulse_nms",
        "mean_wheel_contact_impulse_ns",
        "mean_max_base_vz_mps",
        "mean_max_ground_gain_m",
    ]
    ranked = []
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            group = groups[key]
            count = int(group["attempts"])
            result = {
                "terrain_level": int(key[0]),
                "distance_bin_start_m": key[1],
                "angle_abs_bin_start_deg": key[2],
                "approach_speed_bin_start_mps": key[3],
                "peak_shoulder_torque_bin_start_nm": key[4],
                "attempts": count,
                "successes": int(group["successes"]),
                "success_rate": group["successes"] / count,
                "climb_rate": group["climbed"] / count,
                "stable_landing_rate": group["stable"] / count,
                "mean_step_height_m": _mean(group["step_height"], count),
                "mean_step_width_m": _mean(group["step_width"], count),
                "mean_distance_m": _mean(group["distance"], count),
                "mean_abs_angle_deg": _mean(group["angle"], count),
                "mean_approach_speed_mps": _mean(group["speed"], count),
                "mean_peak_shoulder_torque_nm": _mean(group["shoulder_peak"], count),
                "mean_total_shoulder_torque_impulse_nms": _mean(group["shoulder_impulse"], count),
                "mean_wheel_contact_impulse_ns": _mean(group["contact_impulse"], count),
                "mean_max_base_vz_mps": _mean(group["max_vz"], count),
                "mean_max_ground_gain_m": _mean(group["ground_gain"], count),
            }
            writer.writerow(result)
            if count >= args.min_attempts:
                ranked.append(result)

    ranked.sort(key=lambda row: (row["success_rate"], row["attempts"]), reverse=True)
    print(f"attempts accepted={accepted}, skipped_non_finite={skipped}, bins={len(groups)}")
    print(f"capability map: {output}")
    print("top bins (level, height/width[m], distance[m], |angle|[deg], vx[m/s], torque[Nm], success, n):")
    for row in ranked[:10]:
        print(
            f"  L{row['terrain_level']}, {row['mean_step_height_m']:.3f}/{row['mean_step_width_m']:.3f}, "
            f"{row['mean_distance_m']:.3f}, {row['mean_abs_angle_deg']:.1f}, "
            f"{row['mean_approach_speed_mps']:.2f}, {row['mean_peak_shoulder_torque_nm']:.1f}, "
            f"{row['success_rate']:.3f}, {row['attempts']}"
        )


if __name__ == "__main__":
    main()
