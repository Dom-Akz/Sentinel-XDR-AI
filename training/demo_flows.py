from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training import data as D


def stratified_sample(
    df: pd.DataFrame, n: int, min_per_class: int, seed: int
) -> pd.DataFrame:
    counts = df["attack_cat"].value_counts()
    quota = {
        c: min(k, max(min_per_class, round(n * k / len(df)))) for c, k in counts.items()
    }
    # trim the largest classes if the minimums pushed us over n
    while sum(quota.values()) > n:
        big = max(quota, key=quota.get)
        if quota[big] <= min_per_class:
            break
        quota[big] -= 1
    parts = [
        df[df["attack_cat"] == c].sample(q, random_state=seed)
        for c, q in quota.items()
        if q > 0
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--test", default=D.DEFAULT_TEST)
    ap.add_argument("--out", default="data/demo/demo_flows.csv")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    test = D.load_table(a.test)
    demo = stratified_sample(test, a.n, a.min_per_class, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(out, index=False)
    print(f"Wrote {len(demo)} flows to {out}")
    print(demo["attack_cat"].value_counts().to_string())


if __name__ == "__main__":
    main()
