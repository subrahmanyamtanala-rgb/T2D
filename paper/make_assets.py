"""Build the manuscript's figures (PDF) and LaTeX tables from results/revision.

    python paper/make_assets.py [--results results/revision] [--out paper]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"
# Okabe-Ito colour-blind-safe colours; line styles keep the series apart in greyscale print.
SERIES = {"stacking_rfecv": "#0072B2", "lr": "#D55E00", "rf_rfecv": "#009E73", "svm": "#E69F00"}
STYLE = {"stacking_rfecv": "-", "lr": "--", "rf_rfecv": "-.", "svm": ":"}
# Figures are drawn at their printed size (text width 390 pt = 5.4 in) and included unscaled,
# so the font sizes below are the printed sizes (minimum 7 pt).
FULL_W, HALF_W = 5.85, 3.6
LABEL = {
    "stacking_rfecv": "Stacking + RFECV",
    "stacking": "Stacking",
    "lr": "Logistic regression",
    "lr_own_rfecv": "LR + LR-ranked RFECV",
    "rf_rfecv": "Random forest + RFECV",
    "xgb_rfecv": "XGBoost + RFECV",
    "svm": "SVM",
}
MODEL_NAME = {
    "lr": "Logistic regression",
    "svm": "SVM",
    "knn": "KNN",
    "nb": "Naive Bayes",
    "dt": "Decision tree",
    "rf": "Random forest",
    "et": "Extra trees",
    "gb": "Gradient boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "stacking": "Stacking",
    "lr_own_rfecv": "LR + LR-ranked RFECV",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ figures
def fig_cv_forest(res, out):
    s = pd.read_csv(res / "cv_summary.csv")
    s = s[s.model != "lr_own_rfecv"]
    base = s[~s.rfecv].set_index("model").roc_auc_mean.sort_values()
    fig, ax = plt.subplots(figsize=(HALF_W, 3.6))
    for i, m in enumerate(base.index):
        for use, off, filled in ((False, -0.17, False), (True, 0.17, True)):
            r = s[(s.model == m) & (s.rfecv == use)].iloc[0]
            c = SERIES["stacking_rfecv"] if m == "stacking" else INK2
            ax.plot([r.roc_auc_lo, r.roc_auc_hi], [i + off] * 2, color=c, lw=1, alpha=0.7)
            ax.plot(r.roc_auc_mean, i + off, "o", ms=4, mfc=c if filled else "white", mec=c, mew=1)
    ref = s[(s.model == "lr") & (~s.rfecv)].roc_auc_mean.iloc[0]
    ax.axvline(ref, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(base)))
    ax.set_yticklabels([MODEL_NAME[m] for m in base.index])
    ax.set_xlabel("ROC-AUC, 5 × 10-fold CV (mean, corrected 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", mfc="white", mec=INK2, ms=4, label="All 16 features")
    ax.plot([], [], "o", mfc=INK2, mec=INK2, ms=4, label="Common RFECV subset")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    save(fig, out, "fig2_cv_auc")


def fig_selection(res, out):
    f = pd.read_csv(res / "cv_selection_frequency.csv")
    engineered = {"Glucose_BMI", "Glucose_Age", "HOMA_IR_surrogate", "Insulin_Glucose_Ratio", "BMI_Age", "Obese",
                  "IGT_2h", "Pedigree_Age"}
    rf = f[f.selector == "rf"].set_index("feature").frequency
    lr = f[f.selector == "lr"].set_index("feature").frequency
    feats = sorted(set(rf.index) | set(lr.index) | engineered, key=lambda k: (-rf.get(k, 0), -lr.get(k, 0)))
    y = np.arange(len(feats))[::-1]
    fig, ax = plt.subplots(figsize=(HALF_W, 3.8))
    ax.barh(y + 0.2, [rf.get(k, 0) for k in feats], height=0.38, color=SERIES["stacking_rfecv"], label="RF-ranked RFECV")
    ax.barh(y - 0.2, [lr.get(k, 0) for k in feats], height=0.38, color=MUTED, label="LR-ranked RFECV")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    for t, k in zip(ax.get_yticklabels(), feats):
        if k in engineered:
            t.set_fontstyle("italic")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of 50 outer folds in which the feature was kept")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    save(fig, out, "fig3_selection")


def fig_roc_pr(res, out):
    p = pd.read_csv(res / "holdout_predictions.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.7))
    for name in ("rf_rfecv", "lr", "stacking_rfecv"):
        kw = dict(color=SERIES[name], ls=STYLE[name], lw=1.6 if name == "stacking_rfecv" else 1.1, label=LABEL[name])
        fpr, tpr, _ = roc_curve(p.y, p[name])
        a.plot(fpr, tpr, **kw)
        prec, rec, _ = precision_recall_curve(p.y, p[name])
        b.plot(rec, prec, **kw)
    a.plot([0, 1], [0, 1], ":", color=MUTED, lw=0.8)
    b.axhline(p.y.mean(), ls=":", color=MUTED, lw=0.8)
    a.set(xlabel="1 − specificity", ylabel="Sensitivity", title="(a) ROC")
    b.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="(b) Precision–recall")
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    b.legend(loc="lower left")
    save(fig, out, "fig4_roc_pr")


def fig_calibration(res, out):
    oof = pd.read_csv(res / "cv_oof_predictions.csv.gz")
    p = pd.read_csv(res / "holdout_predictions.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.7))
    for name, (m, r) in {"lr": ("lr", False), "rf_rfecv": ("rf", True), "stacking_rfecv": ("stacking", True)}.items():
        g = oof[(oof.model == m) & (oof.rfecv == r)]
        fp, mp = calibration_curve(g.y, g.proba, n_bins=10, strategy="quantile")
        kw = dict(marker="o", ms=3, color=SERIES[name], ls=STYLE[name], lw=1.6 if name == "stacking_rfecv" else 1.1,
                  label=LABEL[name])
        a.plot(mp, fp, **kw)
        fp, mp = calibration_curve(p.y, p[name], n_bins=5, strategy="quantile")
        b.plot(mp, fp, **kw)
    for ax, t in ((a, "(a) Development out-of-fold (n = 614 × 5)"), (b, "(b) Hold-out (n = 154)")):
        ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed proportion", title=t)
    a.legend(loc="upper left")
    save(fig, out, "fig5_calibration")


def fig_dca(res, out):
    d = pd.read_csv(res / "holdout_decision_curve.csv")
    d = d[d.threshold <= 0.6]
    fig, ax = plt.subplots(figsize=(HALF_W, 2.9))
    ax.plot(d.threshold, d.treat_all, color=MUTED, lw=1, label="Treat all")
    ax.axhline(0, color=INK2, lw=0.8, label="Treat none")
    for name in ("rf_rfecv", "lr", "stacking_rfecv"):
        ax.plot(d.threshold, d[name], color=SERIES[name], ls=STYLE[name], lw=1.6 if name == "stacking_rfecv" else 1.1,
                label=LABEL[name])
    ax.set(ylim=(-0.05, 0.4), xlabel="Threshold probability", ylabel="Net benefit")
    ax.legend(loc="upper right")
    save(fig, out, "fig6_dca")


def fig_shap(res, out):
    import shap

    v = pd.read_csv(res / "shap_values_holdout.csv", index_col=0)
    x = pd.read_csv(res / "shap_inputs_holdout.csv", index_col=0)
    expl = shap.Explanation(values=v.values, data=x.values, feature_names=list(v.columns))
    plt.figure()
    shap.plots.beeswarm(expl, show=False, max_display=8, plot_size=(4.3, 2.9), color_bar_label="Feature value")
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel("SHAP value (change in predicted probability)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    for t in ax.get_yticklabels():
        t.set_fontsize(8)
    cb = fig.axes[-1]
    cb.set_ylabel("Feature value", fontsize=8)
    cb.tick_params(labelsize=7.5)
    save(fig, out, "fig7_shap")


# ------------------------------------------------------------------- tables
def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def cirow(*cells):
    """Second table line holding the confidence intervals, in a smaller font."""
    return " & ".join(r"{\scriptsize\color{gray}" + c + "}" if c else "" for c in cells) + r" \\[2pt]"


def lohi(lo, hi, d=3):
    return f"({lo:.{d}f}--{hi:.{d}f})"


def signed(x, d=3):
    """Signed number with a typographic minus (math mode) instead of a hyphen."""
    out = f"{x:+.{d}f}"
    if float(out) == 0:
        return f"{0:.{d}f}"
    return f"${out[0]}${out[1:]}"


def nfeat(n, selected=True):
    """Number of features: a fold mean (one decimal) after RFECV, otherwise the fixed integer count."""
    return f"{n:.1f}" if selected else f"{n:.0f}"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table_cv(res, out):
    s = pd.read_csv(res / "cv_summary.csv")
    rows = []
    for _, r in s.iterrows():
        name = MODEL_NAME[r.model] + (" + RFECV" if r.rfecv and r.model != "lr_own_rfecv" else "")
        is_ref = r.model == "lr" and not r.rfecv
        rows.append(
            f"{name} & {nfeat(r.n_features, r.rfecv)} & {r.roc_auc_mean:.3f} & {{\\color{{gray}}{lohi(r.roc_auc_lo, r.roc_auc_hi)}}} & "
            f"{r.mcc_mean:.3f} & {r.recall_mean:.3f} & {r.specificity_mean:.3f} & {r.brier_mean:.3f} & "
            + (r"\multicolumn{2}{c}{reference}" if is_ref else f"{signed(r.roc_auc_diff_vs_lr)} & ({pfmt(r.roc_auc_p_holm)})")
            + r" \\"
        )
    (out / "tab_cv.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_holdout(res, out):
    t = pd.read_csv(res / "holdout_metrics_ci.csv").set_index("model")
    dl = pd.read_csv(res / "holdout_delong.csv").set_index("comparator")
    order = ["stacking_rfecv", "stacking", "lr", "lr_own_rfecv", "rf_rfecv", "xgb_rfecv", "svm"]
    rows = []
    for m in order:
        r = t.loc[m]
        test = "--" if m == "stacking_rfecv" else f"{signed(dl.loc[m, 'auc_diff'])} ({pfmt(dl.loc[m, 'p_holm'])})"
        rows.append(
            f"{LABEL[m]} & {r.roc_auc:.3f} & {r.recall:.3f} & {r.specificity:.3f} & {r.mcc:.3f} & {test}" + r" \\"
        )
        rows.append(cirow("", lohi(r.roc_auc_lo, r.roc_auc_hi), lohi(r.recall_lo, r.recall_hi),
                          lohi(r.specificity_lo, r.specificity_hi), lohi(r.mcc_lo, r.mcc_hi), ""))
    (out / "tab_holdout.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_thresholds(res, out):
    t = pd.read_csv(res / "holdout_thresholds.csv")
    rows = []
    for i, (_, r) in enumerate(t.iterrows()):
        if i == 0 or r.model != t.model.iloc[i - 1]:
            if i:
                rows.append(r"\midrule")
            rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{LABEL[r.model]}}}}} \\")
        rows.append(
            f"{100 * r.target_sensitivity:.0f}\\% & {r.threshold:.3f} & {r.sensitivity:.3f} & "
            f"{r.specificity:.3f} & {r.ppv:.3f} & {r.npv:.3f} & {100 * r.flagged:.1f}\\%" + r" \\"
        )
        rows.append(cirow("", "", lohi(r.sensitivity_lo, r.sensitivity_hi), lohi(r.specificity_lo, r.specificity_hi),
                          lohi(r.ppv_lo, r.ppv_hi), lohi(r.npv_lo, r.npv_hi), ""))
    (out / "tab_thresholds.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_calibration(res, out):
    oof = pd.read_csv(res / "calibration_oof.csv").set_index("model")
    ho = pd.read_csv(res / "holdout_metrics_ci.csv").set_index("model")
    rows = []
    for m in ("stacking_rfecv", "lr", "rf_rfecv", "svm"):
        o, h = oof.loc[m], ho.loc[m]
        rows.append(
            f"{LABEL[m]} & {o.brier:.3f} & {signed(o.cal_intercept)} & {o.cal_slope:.3f} & "
            f"{h.brier:.3f} {{\\scriptsize\\color{{gray}}{lohi(h.brier_lo, h.brier_hi)}}} & {signed(h.cal_intercept)} & {h.cal_slope:.3f}" + r" \\"
        )
    (out / "tab_calibration.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_missing(res, out):
    s = pd.read_csv(res / "missing_summary.csv")
    names = {"median": "Median", "median_indicator": "Median + indicators", "iterative": "Iterative (MICE-style)",
             "knn": "KNN ($k=10$)"}
    order = list(names)
    rows = []
    for model, label in (("lr", "Logistic regression"), ("stacking_rfecv", "Stacking + RFECV")):
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\")
        for imp in order:
            r = s[(s.model == model) & (s.imputer == imp)].iloc[0]
            diff = "reference" if imp == "median" else f"{signed(r.roc_auc_diff_vs_median)} ({pfmt(r.roc_auc_p_vs_median)})"
            rows.append(
                f"\\quad {names[imp]} & {nfeat(r.n_features, model != 'lr')} & {ci(r.roc_auc_mean, r.roc_auc_lo, r.roc_auc_hi)} & "
                f"{r.mcc_mean:.3f} & {r.recall_mean:.3f} & {r.brier_mean:.3f} & {diff}" + r" \\"
            )
        if model == "lr":
            rows.append(r"\midrule")
    (out / "tab_missing.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


# --------------------------------------------------------- supplementary tables
SUPP_METRICS = [("accuracy", "Acc."), ("balanced_accuracy", "Bal. acc."), ("precision", "Prec."), ("recall", "Sens."),
                ("specificity", "Spec."), ("f1", "F1"), ("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("brier", "Brier")]


def table_supp_cv(res, out):
    s = pd.read_csv(res / "cv_summary.csv")
    rows = []
    for _, r in s.iterrows():
        name = MODEL_NAME[r.model] + (" + RFECV" if r.rfecv and r.model != "lr_own_rfecv" else "")
        rows.append(f"{name} & {nfeat(r.n_features, r.rfecv)} & " + " & ".join(f"{r[m + '_mean']:.3f}" for m, _ in SUPP_METRICS)
                    + f" & {pfmt(r.roc_auc_p_vs_lr) if not (r.model == 'lr' and not r.rfecv) else '--'}"
                    + f" & {pfmt(r.roc_auc_p_holm) if not (r.model == 'lr' and not r.rfecv) else '--'}" + r" \\")
        rows.append(cirow("", "", *[lohi(r[m + "_lo"], r[m + "_hi"]) for m, _ in SUPP_METRICS], "", ""))
    (out / "supp_cv_all.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_supp_holdout(res, out):
    t = pd.read_csv(res / "holdout_metrics_ci.csv").set_index("model")
    rows = []
    for m in ["stacking_rfecv", "stacking", "lr", "lr_own_rfecv", "rf_rfecv", "xgb_rfecv", "svm"]:
        r = t.loc[m]
        rows.append(f"{LABEL[m]} & " + " & ".join(f"{r[k]:.3f}" for k, _ in SUPP_METRICS)
                    + f" & {r.ap:.3f} & {signed(r.cal_intercept)} & {r.cal_slope:.3f}" + r" \\")
        rows.append(cirow("", *[lohi(r[k + "_lo"], r[k + "_hi"]) for k, _ in SUPP_METRICS], "", "", ""))
    (out / "supp_holdout_all.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def table_supp_selection(res, out):
    f = pd.read_csv(res / "cv_selection_frequency.csv")
    rf = f[f.selector == "rf"].set_index("feature").frequency
    lr = f[f.selector == "lr"].set_index("feature").frequency
    feats = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
             "Age", "Glucose_BMI", "Glucose_Age", "HOMA_IR_surrogate", "Insulin_Glucose_Ratio", "BMI_Age", "Obese",
             "IGT_2h", "Pedigree_Age"]
    rows = [f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {'original' if i < 8 else 'derived'} & "
            f"{100 * rf.get(k, 0):.0f} & {100 * lr.get(k, 0):.0f}" + r" \\" for i, k in enumerate(feats)]
    (out / "supp_selection.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


SHORT = {"lr": "LR", "rf_rfecv": "RF + RFECV"}


def table_external(res, ext, out):
    """PIMA vs NHANES side by side for the key configurations."""
    pc = pd.read_csv(res / "cv_summary.csv")
    nc = pd.read_csv(ext / "cv_summary.csv")
    nh = pd.read_csv(ext / "holdout_metrics_ci.csv").set_index("model")
    nd = pd.read_csv(ext / "holdout_delong.csv").set_index("comparator")
    rows = []
    for key, (m, r) in {"stacking_rfecv": ("stacking", True), "stacking": ("stacking", False), "lr": ("lr", False),
                        "lr_own_rfecv": ("lr_own_rfecv", True), "rf_rfecv": ("rf", True), "xgb_rfecv": ("xgb", True),
                        "svm": ("svm", False)}.items():
        a = pc[(pc.model == m) & (pc.rfecv == r)].iloc[0]
        b = nc[(nc.model == m) & (nc.rfecv == r)].iloc[0]
        h = nh.loc[key]
        test = "--" if key == "stacking_rfecv" else f"{signed(nd.loc[key, 'auc_diff'])} ({pfmt(nd.loc[key, 'p_holm'])})"
        ref = "reference" if key == "lr" else f"{signed(b.roc_auc_diff_vs_lr)} ({pfmt(b.roc_auc_p_holm)})"
        # The longest label is split over the value line and the CI line to keep the table within the text width.
        first, second = ("LR +", r"\quad LR-ranked RFECV") if key == "lr_own_rfecv" else (SHORT.get(key, LABEL[key]), "")
        rows.append(f"{first} & {a.roc_auc_mean:.3f} & {b.roc_auc_mean:.3f} & {ref} & {h.roc_auc:.3f} & {test}" + r" \\")
        rows.append(second + " & " + cirow(lohi(a.roc_auc_lo, a.roc_auc_hi), lohi(b.roc_auc_lo, b.roc_auc_hi), "",
                                           lohi(h.roc_auc_lo, h.roc_auc_hi), ""))
    (out / "tab_external.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/revision"))
    ap.add_argument("--out", type=Path, default=Path("paper"))
    ap.add_argument("--external", type=Path, default=Path("results/external_nhanes"))
    args = ap.parse_args()
    figs, tabs = args.out / "figures", args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    for fn in (fig_cv_forest, fig_selection, fig_roc_pr, fig_calibration, fig_dca, fig_shap):
        fn(args.results, figs)
    for fn in (table_cv, table_holdout, table_thresholds, table_calibration):
        fn(args.results, tabs)
    for fn in (table_supp_cv, table_supp_holdout, table_supp_selection):
        fn(args.results, tabs)
    if (args.external / "holdout_delong.csv").exists():
        table_external(args.results, args.external, tabs)
    if (args.results / "missing_summary.csv").exists():
        table_missing(args.results, tabs)
    print("assets written to", args.out)


if __name__ == "__main__":
    main()
