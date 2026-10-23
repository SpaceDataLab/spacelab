#!/usr/bin/env -S uv run --quiet --with numpy --with scikit-learn python
"""Phase-6, supervised: can we PREDICT the sensor from coverage features, and does it generalise?

The unsupervised gate (experiments/phase6_real_engine.py) showed sensor-aware coverage *clusters* by
sensor while the orbital elements do not. This turns that into a decision procedure: train a classifier
to predict the sensor from the coverage/gap features, evaluated with a **geo-holdout** — a
RAAN-sector-grouped split, so train and test orbits never share an orbit orientation. That rules out
memorising a specific geometry: the classifier must recognise the sensor's coverage *signature* across
unseen orbit orientations. The element-feature classifier is the null (the sensor is invisible to
`a, e, i, RAAN`, so it should sit at chance, 1/3).

Reads the Rust example's CSV; uses only the `LEO-family` rows (one orbit family, three sensors).

Usage:
  cargo run --release -p spacelab --example phase6_features > /tmp/phase6.csv
  uv run --no-project --with numpy --with scikit-learn python experiments/phase6_classifier.py /tmp/phase6.csv
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

ELEMENT_COLS = ["a_km", "ecc", "inc_deg", "raan_deg"]


def load(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    rows = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    col = {name: i for i, name in enumerate(header)}
    regime = np.array([str(r[col["regime"]]) for r in rows])
    sensor = np.array([str(r[col["sensor"]]) for r in rows])
    raan = np.array([float(r[col["raan_deg"]]) for r in rows])
    elem = np.array([[float(r[col[c]]) for c in ELEMENT_COLS] for r in rows])
    cov_cols = [i for name, i in col.items() if name.startswith("c") and name[1:].isdigit()]
    cov = np.array([[float(r[i]) for i in cov_cols] for r in rows])
    return regime, sensor, raan, elem, cov


def geo_holdout_accuracy(X, y, groups, n_splits):
    """Held-out accuracy under a RAAN-sector-grouped split (train/test never share a sector)."""
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits))
    return accuracy_score(y, pred), pred


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase6.csv"
    regime, sensor, raan, elem, cov = load(path)
    fam = regime == "LEO-family"
    y, raan, elem, cov = sensor[fam], raan[fam], elem[fam], cov[fam]
    classes = sorted(set(y))
    # RAAN sectors as holdout groups (six 60-degree bins).
    groups = (raan // 60.0).astype(int)
    n_splits = min(5, len(set(groups)))
    chance = 1.0 / len(classes)

    print("=" * 80)
    print(f"Phase-6 supervised sensor recognition — {len(y)} LEO-family sats, "
          f"{len(classes)} sensors, geo-holdout by RAAN sector ({n_splits}-fold)")
    print("=" * 80)

    acc_cov, pred_cov = geo_holdout_accuracy(cov, y, groups, n_splits)
    acc_elem, _ = geo_holdout_accuracy(elem, y, groups, n_splits)
    print(f"\n  held-out accuracy from COVERAGE features : {acc_cov:.3f}")
    print(f"  held-out accuracy from ELEMENT  features : {acc_elem:.3f}   (chance = {chance:.3f})")

    print("\n  confusion (coverage classifier), rows = truth, cols = predicted:")
    cm = confusion_matrix(y, pred_cov, labels=classes)
    print("            " + "".join(f"{c:>8}" for c in classes))
    for c, row in zip(classes, cm, strict=True):
        print(f"  {c:>8}  " + "".join(f"{v:>8}" for v in row))

    print("\n" + "=" * 80)
    if acc_cov > 0.9 and acc_elem < chance + 0.15:
        print(f"  Sensor is PREDICTABLE from coverage across unseen orbit orientations ({acc_cov:.2f}),")
        print(f"  and remains invisible to the orbital elements ({acc_elem:.2f} ~ chance). The coverage")
        print("  signature generalises — not a memorised geometry. Phase-6 layer confirmed supervised.")
    else:
        print(f"  Coverage accuracy {acc_cov:.2f}, element accuracy {acc_elem:.2f} — inconclusive.")


if __name__ == "__main__":
    main()
