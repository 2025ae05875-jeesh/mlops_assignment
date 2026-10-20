"""Step 2: exploratory data analysis -> reports/figures/*.png + eda_summary.md

Usage:  python scripts/run_eda.py
"""

import matplotlib

matplotlib.use("Agg")  # headless: works in terminal / CI without a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartml import config  # noqa: E402
from heartml.data import load_clean, load_raw  # noqa: E402

sns.set_theme(style="whitegrid", palette="Set2")
FIG = config.FIGURES_DIR
LABELS = {0: "No disease (0)", 1: "Disease (1)"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"[eda] saved {FIG / name}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = load_clean()
    df["diagnosis"] = df[config.TARGET].map(LABELS)

    # 1. Class balance (binary target) + original 0-4 severity
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["diagnosis"].value_counts().reindex(LABELS.values())
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, f"{v} ({v / len(df):.1%})", ha="center")
    axes[0].set(title="Binary target (used for modelling)",
                ylabel="patients", xlabel="")
    sns.countplot(x=raw["num"], ax=axes[1])
    axes[1].set(title="Original UCI 'num' (0 = none, 1-4 = disease)", xlabel="num")
    save(fig, "01_class_balance.png")

    # 2. Missing values in the raw file
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(x=miss.index, y=miss.values, ax=ax)
    ax.set(title="Missing values per raw column ('?' in source)", ylabel="count")
    save(fig, "02_missing_values.png")

    # 3. Numeric distributions split by target
    num = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(data=df, x=col, hue="diagnosis", kde=True, ax=ax,
                     element="step", stat="density", common_norm=False)
        ax.set_title(col)
    save(fig, "03_numeric_histograms_by_target.png")

    # 4. Boxplots (outliers) by target
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, ["age", "trestbps", "chol", "thalach", "oldpeak"]):
        sns.boxplot(data=df, x="diagnosis", y=col, ax=ax)
        ax.set(xlabel="", title=col)
        ax.tick_params(axis="x", rotation=15)
    save(fig, "04_boxplots_by_target.png")

    # 5. Disease rate per category of each categorical feature
    cats = config.BINARY_FEATURES + config.NOMINAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cats):
        rate = (df.assign(**{col: df[col].astype("Int64")})
                .groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index())
        sns.barplot(data=rate, x=col, y="mean", ax=ax)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set(ylim=(0, 1.1), ylabel="disease rate", title=col)
    axes.ravel()[-1].set_visible(False)  # 7 features, 8 panels
    save(fig, "05_categorical_disease_rate.png")

    # 6. Correlation heatmap (Pearson, features + target)
    corr = df[config.FEATURES + [config.TARGET]].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation (features and binary target)")
    save(fig, "06_correlation_heatmap.png")

    # Text summary for the report
    target_corr = corr[config.TARGET].drop(config.TARGET).sort_values(
        key=abs, ascending=False)
    lines = [
        "# EDA summary (auto-generated)",
        f"Rows: {len(df)}  |  Features: {len(config.FEATURES)}",
        f"Class counts: {df[config.TARGET].value_counts().sort_index().to_dict()}",
        f"Missing (raw): {miss[miss > 0].to_dict()}",
        "",
        "## Descriptive statistics (numeric)",
        "```",
        df[num].describe().round(2).to_string(),
        "```",
        "",
        "## Correlation with target (sorted by |r|)",
        "```",
        target_corr.round(3).to_string(),
        "```",
    ]
    (FIG.parent / "eda_summary.md").write_text("\n".join(lines))
    print(f"[eda] saved {FIG.parent / 'eda_summary.md'}")
    print(target_corr.round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
