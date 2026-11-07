from __future__ import annotations

import argparse

from backend.preprocessing import build_schema
from training import data as D

OFFICIAL_ROWS = {"train": 175341, "test": 82332}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=D.DEFAULT_TRAIN)
    ap.add_argument("--test", default=D.DEFAULT_TEST)
    a = ap.parse_args(argv)
    train, test = D.load_table(a.train), D.load_table(a.test)
    schema = build_schema(train)
    report, _ = D.audit(train, test, schema)
    print(f"train: {train.shape}  (official file: {OFFICIAL_ROWS['train']} rows)")
    print(f"test : {test.shape}  (official file: {OFFICIAL_ROWS['test']} rows)")
    print(
        f"features: {len(schema.features)} -> numeric {len(schema.numeric_features)}, "
        f"categorical {schema.categorical_features}"
    )
    print("class counts (train):", report["class_counts_train"])
    print("class counts (test) :", report["class_counts_test"])
    print(
        "duplicate feature rows  train/test:",
        report["duplicate_rows_train_features"],
        "/",
        report["duplicate_rows_test_features"],
    )
    print(
        "test rows identical (features) to a train row:",
        report["test_rows_with_features_also_in_train"],
        f"({report['test_rows_with_features_also_in_train_pct']:.1f}%)",
    )
    print(
        "train feature vectors with conflicting labels:",
        report["feature_vectors_with_conflicting_labels_in_train"],
    )
    if (len(train), len(test)) != (OFFICIAL_ROWS["train"], OFFICIAL_ROWS["test"]):
        print(
            "NOTE: row counts differ from the official partition. That is expected for a cleaned mirror,\n"
            "      but make sure you know which version you are using and say so in your report."
        )


if __name__ == "__main__":
    main()
