"""Recompute common-success cost differences from published aggregate counts."""
import csv
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1] / "reports/paired-cost-groups.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["success_only_success"] == row["cost_aware_success"] == "1":
                count = int(row["tasks"])
                difference = int(row["cost_aware_cost"]) - int(row["success_only_cost"])
                print(f"{row['split']}: common_success={count}, mean_cost_delta={difference / count:+.3f}")


if __name__ == "__main__":
    main()
