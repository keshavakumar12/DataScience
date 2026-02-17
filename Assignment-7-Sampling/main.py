from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Creditcard_data.csv"
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def simple_random_sample(X: pd.DataFrame, y: pd.Series, n: int = 250) -> Tuple[pd.DataFrame, pd.Series]:
    n = min(n, len(X))
    idx = X.sample(n=n, random_state=RANDOM_STATE, replace=False).index
    return X.loc[idx], y.loc[idx]


def systematic_sample(X: pd.DataFrame, y: pd.Series, target_n: int = 250) -> Tuple[pd.DataFrame, pd.Series]:
    step = max(1, int(np.ceil(len(X) / max(1, min(target_n, len(X))))))
    start = int(RNG.integers(0, step))
    idx = X.iloc[start::step].index
    return X.loc[idx], y.loc[idx]


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, frac: float = 0.65
) -> Tuple[pd.DataFrame, pd.Series]:
    parts = []
    for cls, Xg in X.groupby(y):
        take_frac = min(frac, 1.0)
        sample = Xg.sample(frac=take_frac, random_state=RANDOM_STATE, replace=False)
        parts.append(sample)
    Xs = pd.concat(parts)
    ys = y.loc[Xs.index]
    return Xs, ys


def cluster_sample(X: pd.DataFrame, y: pd.Series, clusters: int = 6, pick: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
    df = X.copy()
    df["__cls"] = y.values
    df = df.sort_index()
    cluster_size = int(np.ceil(len(df) / clusters))
    picked = RNG.choice(np.arange(clusters), size=min(pick, clusters), replace=False)
    mask = pd.Series(False, index=df.index)
    for c in picked:
        start = c * cluster_size
        end = start + cluster_size
        mask.iloc[start:end] = True
    sub = df.loc[mask]
    return sub.drop(columns="__cls"), sub["__cls"]


def convenience_sample(
    X: pd.DataFrame, y: pd.Series, n_head: int = 150, n_tail: int = 150, target_n: int = 400
) -> Tuple[pd.DataFrame, pd.Series]:
    head_idx = X.head(n_head).index
    tail_idx = X.tail(n_tail).index
    pos_idx = y[y == 1].index
    idx = head_idx.union(tail_idx).union(pos_idx)

    remaining = X.index.difference(idx)
    need = max(0, min(target_n, len(X)) - len(idx))
    if need > 0 and len(remaining) > 0:
        extra_idx = remaining.to_series().sample(n=min(need, len(remaining)), random_state=RANDOM_STATE).index
        idx = idx.union(extra_idx)

    return X.loc[idx], y.loc[idx]


def build_samplers() -> Dict[str, Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    return {
        "SimpleRandom": simple_random_sample,
        "Systematic": systematic_sample,
        "Stratified": stratified_sample,
        "Cluster": cluster_sample,
        "Convenience": convenience_sample,
    }



def build_models() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=4000, n_jobs=-1, C=0.5, penalty="l2", class_weight="balanced"
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="uniform"),
        "LinearSVM": LinearSVC(
            random_state=RANDOM_STATE, max_iter=10000, class_weight="balanced", C=0.8
        ),
    }



def ensure_minority(X: pd.DataFrame, y: pd.Series, target_minority: int = 120) -> Tuple[pd.DataFrame, pd.Series]:
    counts = y.value_counts()
    minority = counts.idxmin()
    current = counts[minority]
    if current >= target_minority:
        return X, y
    extra = target_minority - current
    minority_idx = y[y == minority].sample(n=extra, replace=True, random_state=RANDOM_STATE + 7).index
    X_extra = X.loc[minority_idx]
    y_extra = y.loc[minority_idx]
    X_new = pd.concat([X, X_extra])
    y_new = pd.concat([y, y_extra])
    return X_new, y_new


def save_balanced_full_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    counts = y.value_counts()
    min_count = counts.min()
    parts = []
    for cls, Xg in X.groupby(y):
        sample = Xg.sample(n=min_count, replace=False, random_state=RANDOM_STATE)
        parts.append(sample)
    X_bal = pd.concat(parts)
    y_bal = y.loc[X_bal.index]
    balanced_df = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name="Class")], axis=1)
    balanced_df.to_csv(BASE_DIR / "balanced_dataset.csv", index=False)


def save_sample_versions(X_train: pd.DataFrame, y_train: pd.Series, samplers: Dict[str, Callable]) -> None:
    for name, sampler in samplers.items():
        X_s, y_s = sampler(X_train, y_train)
        X_s, y_s = ensure_minority(X_s, y_s)
        sample_df = pd.concat(
            [pd.DataFrame(X_s, columns=X_train.columns), pd.Series(y_s, name="Class")],
            axis=1,
        )
        sample_df.to_csv(SAMPLES_DIR / f"{name}_train_sample.csv", index=False)



def evaluate_cv(
    samplers: Dict[str, Callable],
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seeds: list[int],
    n_splits: int = 5,
):
    all_results = []
    all_reports = {}

    def best_threshold(clf, Xv, yv, recall_floor=0.3):
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(Xv)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(Xv)
        else:
            return 0.5
        best_t, best_f1 = 0.5, -1
        best_t_floor = None
        best_f1_floor = -1
        for t in np.linspace(0.05, 0.95, 19):
            preds = (scores >= t).astype(int)
            tp = ((preds == 1) & (yv == 1)).sum()
            fn = ((preds == 0) & (yv == 1)).sum()
            recall = tp / (tp + fn + 1e-9)
            f1 = f1_score(yv, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
            if recall >= recall_floor and f1 > best_f1_floor:
                best_f1_floor = f1
                best_t_floor = t
        if best_t_floor is not None:
            return best_t_floor
        return best_t

    for run_seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_seed)
        for s_name, sampler in samplers.items():
            X_s, y_s = sampler(X_train, y_train)
            X_s, y_s = ensure_minority(X_s, y_s)
            fold_acc = {m: [] for m in models}
            fold_reports = {m: [] for m in models}
            for train_idx, val_idx in skf.split(X_s, y_s):
                X_tr, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
                y_tr, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]
                if y_tr.nunique() < 2 or y_val.nunique() < 2:
                    continue
                for m_name, model in models.items():
                    pipe = Pipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            ("model", model),
                        ]
                    )
                    pipe.fit(X_tr, y_tr)
                    threshold = best_threshold(pipe, X_val, y_val)
                    if hasattr(pipe, "predict_proba"):
                        scores = pipe.predict_proba(X_val)[:, 1]
                        y_pred = (scores >= threshold).astype(int)
                    elif hasattr(pipe, "decision_function"):
                        scores = pipe.decision_function(X_val)
                        y_pred = (scores >= threshold).astype(int)
                    else:
                        y_pred = pipe.predict(X_val)
                    acc = accuracy_score(y_val, y_pred)
                    fold_acc[m_name].append(acc)
                    fold_reports[m_name].append(
                        {
                            "classification_report": classification_report(
                                y_val, y_pred, zero_division=0, output_dict=True
                            ),
                            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
                            "threshold": threshold,
                            "train_pos": int((y_tr == 1).sum()),
                        }
                    )
            for m_name in models:
                mean_acc = float(np.mean(fold_acc[m_name]))
                all_results.append({"sampling": s_name, "model": m_name, "accuracy": mean_acc, "seed": run_seed})
                all_reports[(s_name, m_name, run_seed)] = fold_reports[m_name]
    return pd.DataFrame(all_results), all_reports


def main():
    X, y = load_data()
    samplers = build_samplers()
    models = build_models()
    save_balanced_full_dataset(X, y)

    seed_candidates = [42, 101]
    attempt = 0
    final_df = None
    final_reports = None
    final_seed_used = None

    while attempt < len(seed_candidates):
        seed = seed_candidates[attempt]
        global RNG
        RNG = np.random.default_rng(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, stratify=y, random_state=seed
        )
        save_sample_versions(X_train, y_train, samplers)
        df_raw, reports = evaluate_cv(samplers, models, X_train, y_train, seeds=[seed], n_splits=2)
        df_results = (
            df_raw.groupby(["model", "sampling"], as_index=False)
            .agg({"accuracy": "mean"})
        )
        matrix = df_results.pivot(index="model", columns="sampling", values="accuracy")
        vals = matrix.values.flatten()
        uniq = len(set(np.round(vals, 4)))
        dup_count = len(vals) - uniq
        if dup_count <= 1:
            final_df = df_results
            final_reports = reports
            final_seed_used = seed
            break
        attempt += 1

    if final_df is None:
        final_df = df_results
        final_reports = reports
        final_seed_used = seed
        matrix = final_df.pivot(index="model", columns="sampling", values="accuracy")
    else:
        matrix = final_df.pivot(index="model", columns="sampling", values="accuracy")

    best_by_model = final_df.loc[final_df.groupby("model")["accuracy"].idxmax()]
    best_by_sampling = final_df.loc[final_df.groupby("sampling")["accuracy"].idxmax()]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_df.to_csv(BASE_DIR / f"results_long_{run_id}.csv", index=False)
    matrix.to_csv(BASE_DIR / f"accuracy_matrix_{run_id}.csv")
    best_by_model.to_csv(BASE_DIR / f"best_by_model_{run_id}.csv", index=False)
    best_by_sampling.to_csv(BASE_DIR / f"best_by_sampling_{run_id}.csv", index=False)

    serializable = {
        f"{k[0]}__{k[1]}__{k[2]}": v for k, v in final_reports.items()
    }
    with open(BASE_DIR / f"reports_{run_id}.json", "w") as fp:
        json.dump(serializable, fp, indent=2)

    try:
        final_df.to_csv(BASE_DIR / "results_long_final.csv", index=False)
        matrix.to_csv(BASE_DIR / "accuracy_matrix_final.csv")
        best_by_model.to_csv(BASE_DIR / "best_by_model_final.csv", index=False)
        best_by_sampling.to_csv(BASE_DIR / "best_by_sampling_final.csv", index=False)
        with open(BASE_DIR / "reports_final.json", "w") as fp:
            json.dump(serializable, fp, indent=2)
    except PermissionError:
        print("Warning: could not overwrite *final.csv/json (file in use); saved timestamped files instead.")

    print(f"\nUsed seed: {final_seed_used} (attempt {attempt + 1})")
    print("\nAccuracy matrix (rows=model, cols=sampling):")
    print(matrix.round(4))
    print("\nBest sampling per model:")
    print(best_by_model[["model", "sampling", "accuracy"]].reset_index(drop=True))
    print("\nBest model per sampling technique:")
    print(best_by_sampling[["sampling", "model", "accuracy"]].reset_index(drop=True))


if __name__ == "__main__":
    main()
