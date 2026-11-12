"""
Quantum Kernel Learning with Low-Rank Feature Embeddings for Intelligent Classification
Datasets: Wisconsin Diagnostic Breast Cancer (WDBC, UCI) and UCI Dry Bean.

Pipeline (fitted inside every training fold - no leakage):
    standardise -> low-rank embedding (truncated SVD / PCA, rank r = #qubits)
    -> min-max scale to [0, pi] -> IQP/ZZ fidelity quantum kernel -> SVM (precomputed kernel)

Run:  python src/run_experiments.py            (writes everything to ../results)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC

from quantum_kernel import (FidelityQuantumKernel, effective_dimension,
                            kernel_target_alignment, verify_against_pennylane)

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DRYBEAN_SUBSAMPLE = 3000          # kernel methods scale O(n^2); stratified subsample


# ----------------------------------------------------------------------------- data
def load_datasets():
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target

    xlsx = next(DATA.rglob("Dry_Bean_Dataset.xlsx"))
    df = pd.read_excel(xlsx)
    X_db = df.drop(columns=["Class"]).values.astype(float)
    y_db_names = df["Class"].values
    classes = np.unique(y_db_names)
    y_db = np.searchsorted(classes, y_db_names)
    X_db, _, y_db, _ = train_test_split(X_db, y_db, train_size=DRYBEAN_SUBSAMPLE,
                                        stratify=y_db, random_state=SEED)
    return {
        "BreastCancer": dict(X=X_bc, y=y_bc, classes=list(bc.target_names)),
        "DryBean": dict(X=X_db, y=y_db, classes=list(classes)),
    }


# ----------------------------------------------------------------------------- models
class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM on a fidelity quantum kernel; input must already be r-dimensional."""

    def __init__(self, layers=2, gamma=0.5, C=1.0, entanglement="linear"):
        self.layers, self.gamma, self.C, self.entanglement = layers, gamma, C, entanglement

    def fit(self, X, y):
        self.qk_ = FidelityQuantumKernel(X.shape[1], self.layers, self.gamma, self.entanglement)
        self.Xtr_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced")
        self.svc_.fit(self.qk_(X), y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        return self.svc_.predict(self.qk_(X, self.Xtr_))


def embedding(kind: str, r: int):
    """Low-rank embedding block. Output is scaled to [0, pi] (rotation angles)."""
    proj = PCA(n_components=r, random_state=SEED) if kind == "pca" else \
        GaussianRandomProjection(n_components=r, random_state=SEED)
    return [("std", StandardScaler()), ("lowrank", proj), ("angle", MinMaxScaler((0, np.pi)))]


QK_GRID = {"clf__gamma": [0.05, 0.1, 0.2, 0.4, 0.8], "clf__C": [0.1, 1, 10, 100]}
# main model: circuit depth and entanglement are also selected in the inner CV loop
QK_GRID_FULL = dict(QK_GRID, clf__layers=[1, 2], clf__entanglement=["linear", "full"])


def model_zoo(r: int):
    """name -> (pipeline, param_grid). All baselines see the SAME r-dim embedding,
    except 'RBF-SVM (full features)' which is the classical upper reference."""
    emb = embedding("pca", r)
    return {
        "QK-SVM (ours)": (Pipeline(emb + [("clf", QuantumKernelSVC())]), QK_GRID_FULL),
        "RBF-SVM": (Pipeline(emb + [("clf", SVC(kernel="rbf", class_weight="balanced"))]),
                    {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}),
        "Linear-SVM": (Pipeline(emb + [("clf", SVC(kernel="linear", class_weight="balanced"))]),
                       {"clf__C": [0.01, 0.1, 1, 10]}),
        "LogReg": (Pipeline(emb + [("clf", LogisticRegression(max_iter=3000))]),
                   {"clf__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (Pipeline(emb + [("clf", RandomForestClassifier(300, random_state=SEED))]),
                         {"clf__max_depth": [None, 8]}),
        "MLP": (Pipeline(emb + [("clf", MLPClassifier((64, 32), max_iter=2000, random_state=SEED))]),
                {"clf__alpha": [1e-4, 1e-2]}),
        "RBF-SVM (full features)": (Pipeline([("std", StandardScaler()),
                                              ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
                                    {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]}),
    }


def scores(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                f1_macro=f1_score(y, p, average="macro"))


def nested_cv(pipe, grid, X, y, n_outer=5, n_inner=3):
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=SEED)
    rows, y_all, p_all, best = [], [], [], []
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        t0 = time.time()
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="f1_macro", n_jobs=-1)
        gs.fit(X[tr], y[tr])
        p = gs.predict(X[te])
        rows.append(dict(fold=fold, **scores(y[te], p), seconds=time.time() - t0))
        y_all.append(y[te]); p_all.append(p); best.append(gs.best_params_)
    return pd.DataFrame(rows), np.concatenate(y_all), np.concatenate(p_all), best


# ----------------------------------------------------------------------------- experiments
def main_comparison(dsets, r=8):
    per_fold, summary, preds = [], [], {}
    for dname, d in dsets.items():
        for mname, (pipe, grid) in model_zoo(r).items():
            df, yt, yp, best = nested_cv(pipe, grid, d["X"], d["y"])
            df.insert(0, "model", mname); df.insert(0, "dataset", dname)
            per_fold.append(df)
            s = {"dataset": dname, "model": mname, "rank_r": r}
            for m in ["accuracy", "balanced_accuracy", "f1_macro"]:
                s[f"{m}_mean"], s[f"{m}_std"] = df[m].mean(), df[m].std()
            s["seconds_per_fold"] = df["seconds"].mean()
            s["best_params_fold0"] = json.dumps(best[0], default=str)
            summary.append(s); preds[(dname, mname)] = (yt, yp)
            print(f"[main] {dname:12s} {mname:26s} acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} "
                  f"F1={s['f1_macro_mean']:.4f}")
    per_fold = pd.concat(per_fold)
    summary = pd.DataFrame(summary)

    # paired t-test (QK vs each baseline, fold-wise accuracy)
    tests = []
    for dname in dsets:
        q = per_fold[(per_fold.dataset == dname) & (per_fold.model == "QK-SVM (ours)")]["accuracy"].values
        for mname in summary.model.unique():
            if mname == "QK-SVM (ours)":
                continue
            b = per_fold[(per_fold.dataset == dname) & (per_fold.model == mname)]["accuracy"].values
            t, pval = stats.ttest_rel(q, b) if np.any(q != b) else (0.0, 1.0)
            tests.append(dict(dataset=dname, baseline=mname, mean_diff=(q - b).mean(), t=t, p_value=pval))
    return per_fold, summary, pd.DataFrame(tests), preds


def rank_ablation(dsets, ranks=(2, 4, 6, 8)):
    rows = []
    for dname, d in dsets.items():
        for r in ranks:
            for kind in ["pca", "random"]:
                pipe = Pipeline(embedding(kind, r) + [("clf", QuantumKernelSVC(layers=2))])
                df, *_ = nested_cv(pipe, QK_GRID, d["X"], d["y"])
                rows.append(dict(dataset=dname, model="QK-SVM", embedding=kind, rank_r=r,
                                 accuracy=df.accuracy.mean(), accuracy_std=df.accuracy.std(),
                                 f1_macro=df.f1_macro.mean()))
            pipe = Pipeline(embedding("pca", r) + [("clf", SVC(kernel="rbf", class_weight="balanced"))])
            df, *_ = nested_cv(pipe, {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
                               d["X"], d["y"])
            rows.append(dict(dataset=dname, model="RBF-SVM", embedding="pca", rank_r=r,
                             accuracy=df.accuracy.mean(), accuracy_std=df.accuracy.std(),
                             f1_macro=df.f1_macro.mean()))
            print(f"[rank] {dname} r={r} done")
    return pd.DataFrame(rows)


def circuit_ablation(dsets, r=8):
    rows = []
    for dname, d in dsets.items():
        for L in [1, 2, 3]:
            for ent in ["linear", "ring", "full"]:
                pipe = Pipeline(embedding("pca", r) + [("clf", QuantumKernelSVC(layers=L, entanglement=ent))])
                df, *_ = nested_cv(pipe, QK_GRID, d["X"], d["y"])
                rows.append(dict(dataset=dname, layers=L, entanglement=ent,
                                 accuracy=df.accuracy.mean(), accuracy_std=df.accuracy.std(),
                                 f1_macro=df.f1_macro.mean()))
        print(f"[circuit] {dname} done")
    return pd.DataFrame(rows)


def kernel_geometry(dsets, ranks=(2, 4, 6, 8), gammas=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6)):
    """Concentration, kernel-target alignment and effective dimension vs bandwidth and rank."""
    rows = []
    for dname, d in dsets.items():
        X, y = d["X"], d["y"]
        idx = np.random.default_rng(SEED).choice(len(y), size=min(600, len(y)), replace=False)
        for r in ranks:
            Z = Pipeline(embedding("pca", r)).fit_transform(X[idx])
            for g in gammas:
                K = FidelityQuantumKernel(r, 2, g, "linear")(Z)
                off = K[~np.eye(len(K), dtype=bool)]
                rows.append(dict(dataset=dname, rank_r=r, gamma=g, offdiag_mean=off.mean(),
                                 offdiag_var=off.var(), alignment=kernel_target_alignment(K, y[idx]),
                                 effective_dim=effective_dimension(K)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def figures(dsets, summary, rank_df, geo_df, preds, circ_df):
    # 1. model comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dname in zip(axes, dsets):
        s = summary[summary.dataset == dname].sort_values("accuracy_mean")
        cols = ["#c0392b" if "ours" in m else "#7f8c8d" for m in s.model]
        ax.barh(s.model, s.accuracy_mean, xerr=s.accuracy_std, color=cols)
        ax.set_xlim(max(0, s.accuracy_mean.min() - 0.1), 1.0)
        ax.set_title(f"{dname} - 5-fold nested CV accuracy (r=8)")
    plt.tight_layout(); plt.savefig(FIG / "fig1_model_comparison.png", dpi=200); plt.close()

    # 2. rank ablation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dname in zip(axes, dsets):
        s = rank_df[rank_df.dataset == dname]
        for (m, e), g in s.groupby(["model", "embedding"]):
            ax.errorbar(g.rank_r, g.accuracy, yerr=g.accuracy_std, marker="o", capsize=3, label=f"{m} / {e}")
        ax.set_xlabel("rank r (= number of qubits)"); ax.set_ylabel("accuracy"); ax.set_title(dname)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "fig2_rank_ablation.png", dpi=200); plt.close()

    # 3. kernel geometry
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, dname in enumerate(dsets):
        s = geo_df[geo_df.dataset == dname]
        for ax, met, lab in zip(axes[row], ["offdiag_var", "alignment", "effective_dim"],
                                ["off-diagonal variance (concentration)", "kernel-target alignment",
                                 "effective dimension"]):
            for r, g in s.groupby("rank_r"):
                ax.plot(g.gamma, g[met], marker="o", label=f"r={r}")
            ax.set_xscale("log"); ax.set_xlabel("bandwidth gamma"); ax.set_title(f"{dname}: {lab}", fontsize=9)
            ax.grid(alpha=.3)
        axes[row][0].legend(fontsize=8)
        axes[row][0].set_yscale("log")
    plt.tight_layout(); plt.savefig(FIG / "fig3_kernel_geometry.png", dpi=200); plt.close()

    # 4. kernel heat-maps sorted by class
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (dname, d) in zip(axes, dsets.items()):
        idx = np.random.default_rng(SEED).choice(len(d["y"]), size=min(400, len(d["y"])), replace=False)
        idx = idx[np.argsort(d["y"][idx], kind="stable")]
        Z = Pipeline(embedding("pca", 8)).fit_transform(d["X"][idx])
        K = FidelityQuantumKernel(8, 2, 0.2, "linear")(Z)
        im = ax.imshow(K, cmap="viridis"); ax.set_title(f"{dname}: quantum kernel (r=8, L=2, linear, gamma=0.2)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(FIG / "fig4_kernel_heatmaps.png", dpi=200); plt.close()

    # 5. confusion matrices of QK-SVM
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (dname, d) in zip(axes, dsets.items()):
        yt, yp = preds[(dname, "QK-SVM (ours)")]
        ConfusionMatrixDisplay(confusion_matrix(yt, yp), display_labels=d["classes"]).plot(
            ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"{dname}: QK-SVM (pooled over outer folds)")
    plt.tight_layout(); plt.savefig(FIG / "fig5_confusion_qk.png", dpi=200); plt.close()

    # 6. circuit ablation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, dname in zip(axes, dsets):
        p = circ_df[circ_df.dataset == dname].pivot(index="layers", columns="entanglement", values="accuracy")
        im = ax.imshow(p.values, cmap="RdYlGn"); ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), p.index); ax.set_ylabel("layers L"); ax.set_title(dname)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f"{p.values[i, j]:.4f}", ha="center", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig6_circuit_ablation.png", dpi=200); plt.close()


def main():
    t0 = time.time()
    err = verify_against_pennylane(4, 2)
    print(f"Simulator vs PennyLane max |dK| = {err:.2e}")
    dsets = load_datasets()
    info = {k: dict(n=int(len(v["y"])), d=int(v["X"].shape[1]), classes=v["classes"],
                    class_counts=np.bincount(v["y"]).tolist()) for k, v in dsets.items()}

    per_fold, summary, tests, preds = main_comparison(dsets)
    rank_df = rank_ablation(dsets)
    circ_df = circuit_ablation(dsets)
    geo_df = kernel_geometry(dsets)

    per_fold.to_csv(RES / "main_per_fold.csv", index=False)
    summary.to_csv(RES / "main_summary.csv", index=False)
    tests.to_csv(RES / "significance_tests.csv", index=False)
    rank_df.to_csv(RES / "ablation_rank.csv", index=False)
    circ_df.to_csv(RES / "ablation_circuit.csv", index=False)
    geo_df.to_csv(RES / "kernel_geometry.csv", index=False)
    figures(dsets, summary, rank_df, geo_df, preds, circ_df)

    json.dump(dict(simulator_check_max_abs_diff=err, datasets=info, seed=SEED,
                   drybean_subsample=DRYBEAN_SUBSAMPLE, runtime_minutes=(time.time() - t0) / 60),
              open(RES / "run_info.json", "w"), indent=2)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
