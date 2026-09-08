"""
Fate-prediction model.

Trains a RandomForest to predict a three-body system's eventual fate
(periodic / chaotic / collision / escape) from quantities known at t=0. This is
a *surrogate model*: once trained, it predicts in microseconds what would
otherwise need a full simulation.

Only t=0 features are used - initial position, velocity, energy, angular
momentum, and closest initial approach. Outcome quantities (lifetime, max
separation, ...) are deliberately excluded: using them would be data leakage.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from survey import DEFAULT_SCENARIO, run_samples, initial_features

FEATURES = ["init_position", "init_velocity", "init_energy",
            "init_ang_mom", "init_min_dist"]
CLASSES = ["periodic", "chaotic", "collision", "escape"]


def build_dataset(n_samples, steps, seed=0, scenario=DEFAULT_SCENARIO):
    """Generate a labelled dataset by simulating randomly sampled initial conditions."""
    return run_samples(scenario, n_samples, steps, seed=seed)


def train_fate_model(df, seed=0):
    """Train + evaluate a RandomForest. Returns (model, metrics dict)."""
    X = df[FEATURES].values
    y = df["class"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)

    labels = [c for c in CLASSES if c in np.unique(y)]
    metrics = {
        "accuracy": accuracy_score(yte, yp),
        "report": classification_report(yte, yp, labels=labels,
                                        output_dict=True, zero_division=0),
        "confusion": confusion_matrix(yte, yp, labels=labels),
        "labels": labels,
        "importances": dict(zip(FEATURES, clf.feature_importances_)),
        "n_train": len(ytr),
        "n_test": len(yte),
    }
    return clf, metrics


def predict_grid(clf, n_grid, scenario=DEFAULT_SCENARIO):
    """Predict fate across the (position, velocity) plane, for a decision map.

    Uses initial_features() to build the model inputs from t=0 physics - no
    simulation, so this is instant.
    """
    pa = np.linspace(*scenario["pos_range"], n_grid)
    va = np.linspace(*scenario["vel_range"], n_grid)
    PY, VX = np.meshgrid(pa, va, indexing="ij")
    feats = initial_features(scenario, PY.ravel(), VX.ravel())
    pred = clf.predict(feats[FEATURES].values)
    return pred.reshape(n_grid, n_grid), pa, va


def predict_one(clf, position, velocity, scenario=DEFAULT_SCENARIO):
    """Predict fate for a single (position, velocity). Returns (label, prob dict)."""
    feats = initial_features(scenario, [position], [velocity])
    x = feats[FEATURES].values
    label = clf.predict(x)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(x)[0]))
    return label, proba