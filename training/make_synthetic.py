from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]
CATEGORICAL = ["proto", "service", "state"]
FEATURES = ["dur", "proto", "service", "state"] + NUMERIC[
    1:
]  # official column order is close to this

# Typical magnitude of each numeric feature for ordinary traffic.
SCALE = {
    "dur": 1.0,
    "spkts": 12,
    "dpkts": 10,
    "sbytes": 1500,
    "dbytes": 2500,
    "rate": 400,
    "sttl": 62,
    "dttl": 252,
    "sload": 3e5,
    "dload": 4e5,
    "sloss": 1,
    "dloss": 1,
    "sinpkt": 30,
    "dinpkt": 25,
    "sjit": 400,
    "djit": 300,
    "swin": 255,
    "stcpb": 1e9,
    "dtcpb": 1e9,
    "dwin": 255,
    "tcprtt": 0.05,
    "synack": 0.03,
    "ackdat": 0.02,
    "smean": 120,
    "dmean": 180,
    "trans_depth": 0.5,
    "response_body_len": 800,
    "ct_srv_src": 6,
    "ct_state_ttl": 1,
    "ct_dst_ltm": 5,
    "ct_src_dport_ltm": 3,
    "ct_dst_sport_ltm": 2,
    "ct_dst_src_ltm": 5,
    "is_ftp_login": 0.02,
    "ct_ftp_cmd": 0.02,
    "ct_flw_http_mthd": 0.5,
    "ct_src_ltm": 5,
    "ct_srv_dst": 6,
    "is_sm_ips_ports": 0.01,
}
INTEGER = {
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
}

# How each class departs from ordinary traffic (multipliers on the scale; "=" sets a fixed value).
EFFECTS = {
    "Normal": {},
    "Generic": {"dur": 0.01, "rate": 40, "sttl": "=254", "ct_srv_src": 3},
    "Exploits": {"sbytes": 3, "dpkts": 2, "sttl": "=254", "ct_dst_ltm": 2, "sload": 3},
    "Fuzzers": {"sloss": 4, "sinpkt": 3, "sttl": "=254", "ct_srv_dst": 3, "sjit": 3},
    "DoS": {"spkts": 8, "sbytes": 6, "rate": 10, "ct_dst_src_ltm": 4, "sttl": "=254"},
    "Reconnaissance": {
        "dur": 0.2,
        "spkts": 0.5,
        "ct_srv_src": 3,
        "ct_dst_sport_ltm": 3,
        "dbytes": 0.2,
    },
    "Analysis": {"ct_src_dport_ltm": 3, "dur": 2, "dbytes": 0.5, "ct_dst_ltm": 1.5},
    "Backdoor": {
        "dur": 3,
        "ct_src_ltm": 3,
        "smean": 0.7,
        "dwin": 0.5,
        "ct_dst_ltm": 1.5,
    },
    "Shellcode": {"sbytes": 2, "smean": 2, "response_body_len": 3, "ct_state_ttl": 3},
    "Worms": {
        "sbytes": 4,
        "spkts": 4,
        "ct_srv_dst": 4,
        "sttl": "=254",
        "ct_src_ltm": 4,
    },
}
PROTO = {"tcp": 0.6, "udp": 0.3, "icmp": 0.05, "arp": 0.03, "ospf": 0.02}
SERVICE = {"-": 0.5, "http": 0.2, "dns": 0.12, "ftp": 0.06, "smtp": 0.06, "ssh": 0.06}
STATE = {"FIN": 0.5, "CON": 0.25, "INT": 0.1, "REQ": 0.1, "RST": 0.05}
CAT_BIAS = {
    "Generic": {"proto": {"udp": 4}, "service": {"dns": 6}},
    "DoS": {"state": {"INT": 4}, "proto": {"tcp": 1.5}},
    "Reconnaissance": {"state": {"REQ": 3}, "service": {"-": 1.5}},
    "Exploits": {"service": {"http": 3, "ftp": 2}},
    "Fuzzers": {"service": {"http": 3}, "state": {"RST": 3}},
}
SHARE = {
    "Normal": 0.32,
    "Generic": 0.22,
    "Exploits": 0.19,
    "Fuzzers": 0.10,
    "DoS": 0.07,
    "Reconnaissance": 0.06,
    "Analysis": 0.015,
    "Backdoor": 0.012,
    "Shellcode": 0.008,
    "Worms": 0.005,
}
MIN_PER_CLASS = 40


def _categorical(rng, table: dict[str, float], bias: dict[str, float], n: int):
    keys = list(table)
    w = np.array([table[k] * bias.get(k, 1.0) for k in keys])
    return rng.choice(keys, size=n, p=w / w.sum())


def _sample_class(rng, cls: str, n: int, overlap: float) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    eff = EFFECTS[cls]
    for f in NUMERIC:
        base = SCALE[f] * rng.lognormal(mean=0.0, sigma=0.9, size=n)
        e = eff.get(f)
        if isinstance(e, str):
            vals = np.full(n, float(e[1:])) + rng.normal(0, 1, n)
        elif e is not None:
            vals = base * e
        else:
            vals = base
        cols[f] = vals
    df = pd.DataFrame(cols)
    bias = CAT_BIAS.get(cls, {})
    df["proto"] = _categorical(rng, PROTO, bias.get("proto", {}), n)
    df["service"] = _categorical(rng, SERVICE, bias.get("service", {}), n)
    df["state"] = _categorical(rng, STATE, bias.get("state", {}), n)
    if cls != "Normal" and overlap > 0:
        # Some attack flows look like ordinary traffic (hard cases -> imperfect scores).
        mask = rng.random(n) < overlap
        if mask.any():
            normal_like = _sample_class(rng, "Normal", int(mask.sum()), 0.0)
            df.loc[mask, :] = normal_like.to_numpy()
    return df


def generate(n_rows: int, seed: int, overlap: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for cls, share in SHARE.items():
        n = max(MIN_PER_CLASS, int(round(n_rows * share)))
        part = _sample_class(rng, cls, n, overlap)
        part["attack_cat"] = cls
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    for f in NUMERIC:
        df[f] = df[f].clip(lower=0)
        if f in INTEGER:
            df[f] = df[f].round().astype("int64")
        else:
            df[f] = df[f].round(6)
    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df[["id", *FEATURES, "attack_cat", "label"]]


def write_synthetic(
    out_dir: str | Path, n_train: int = 6000, n_test: int = 2500, seed: int = 7
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = generate(n_train, seed)
    test = generate(n_test, seed + 1)
    # Plant a few exact duplicates / train-test overlaps so the audit has something to find.
    train = pd.concat([train, train.sample(25, random_state=seed)], ignore_index=True)
    leak = train.sample(30, random_state=seed + 2).copy()
    test = pd.concat([test, leak], ignore_index=True)
    train["id"] = np.arange(1, len(train) + 1)
    test["id"] = np.arange(1, len(test) + 1)
    tr_path = out / "UNSW_NB15_training-set.csv"
    te_path = out / "UNSW_NB15_testing-set.csv"
    train.to_csv(tr_path, index=False)
    test.to_csv(te_path, index=False)
    (out / "SYNTHETIC.txt").write_text(
        "These files are SYNTHETIC test data generated by training/make_synthetic.py.\n"
        "They are not real network traffic and results on them are meaningless.\n"
    )
    return tr_path, te_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    tr, te = write_synthetic(a.out, a.n_train, a.n_test, a.seed)
    print(
        f"SYNTHETIC data written:\n  {tr}\n  {te}\n(not real traffic - for plumbing tests only)"
    )


if __name__ == "__main__":
    main()
