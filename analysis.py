import json

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.05
TIME_LIMIT_MIN = 120
MIN_VALID_MIN = 1
N_BOOT = 10_000
SEED = 2026
GROUPS = ["without_ai", "with_ai"]
LABELS = {"without_ai": "Без ШІ-асистента", "with_ai": "З ШІ-асистентом"}

raw = pd.read_csv("data/task_time.csv")

cleaning = {"raw_rows": len(raw), "missing": int(raw["time_min"].isna().sum())}
df = raw.dropna(subset=["time_min"])
invalid = (df["time_min"] < MIN_VALID_MIN) | (df["time_min"] > TIME_LIMIT_MIN)
cleaning["invalid_removed"] = df.loc[invalid, ["participant_id", "group", "time_min"]].to_dict("records")
df = df.loc[~invalid]
cleaning["clean_rows"] = len(df)
df.to_csv("data/task_time_clean.csv", index=False)

x = df.loc[df["group"] == "without_ai", "time_min"].to_numpy()
y = df.loc[df["group"] == "with_ai", "time_min"].to_numpy()


def describe(v):
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": int(len(v)), "mean": float(np.mean(v)), "median": float(np.median(v)),
            "sd": float(np.std(v, ddof=1)), "iqr": float(q3 - q1), "min": float(np.min(v)), "max": float(np.max(v))}


descriptive = {"without_ai": describe(x), "with_ai": describe(y)}

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([x, y], tick_labels=[LABELS[g] for g in GROUPS])
ax.set_ylabel("Час виконання задачі, хв")
ax.set_title("Час виконання задачі за групами")
fig.tight_layout()
fig.savefig("results/boxplot.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
bins = np.arange(10, 125, 7.5)
for a, v, g in zip(axes, [x, y], GROUPS):
    a.hist(v, bins=bins, edgecolor="black")
    a.set_title(LABELS[g])
    a.set_xlabel("Час, хв")
axes[0].set_ylabel("Кількість учасників")
fig.tight_layout()
fig.savefig("results/histogram.png", dpi=200)
plt.close(fig)

shapiro = {g: stats.shapiro(v) for g, v in zip(GROUPS, [x, y])}
levene = stats.levene(x, y, center="median")
normal = all(r.pvalue > ALPHA for r in shapiro.values())

diff_mean = float(np.mean(x) - np.mean(y))
if normal:
    equal_var = levene.pvalue > ALPHA
    test = stats.ttest_ind(x, y, equal_var=equal_var)
    test_name = "t-тест Стьюдента" if equal_var else "t-тест Велча"
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    effect = {"name": "d Коена", "value": diff_mean / pooled}
    ci = test.confidence_interval(1 - ALPHA)
    ci_info = {"estimate": "різниця середніх", "value": diff_mean, "low": float(ci.low), "high": float(ci.high)}
    statistic = float(test.statistic)
else:
    test = stats.mannwhitneyu(x, y, alternative="two-sided")
    test_name = "U-тест Манна–Уітні"
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    effect = {"name": "δ Кліффа", "value": float((gt - lt) / (len(x) * len(y)))}
    rng = np.random.default_rng(SEED)
    boot = np.array([np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(N_BOOT)])
    ci_info = {"estimate": "різниця медіан (бутстреп, 10 000 вибірок)",
               "value": float(np.median(x) - np.median(y)),
               "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5))}
    statistic = float(test.statistic)

results = {
    "cleaning": cleaning,
    "descriptive": descriptive,
    "shapiro": {g: {"W": float(r.statistic), "p": float(r.pvalue)} for g, r in shapiro.items()},
    "levene": {"W": float(levene.statistic), "p": float(levene.pvalue)},
    "test": {"name": test_name, "statistic": statistic, "p": float(test.pvalue)},
    "effect": effect,
    "ci95": ci_info,
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
                 "matplotlib": matplotlib.__version__},
}
with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print(json.dumps(results, ensure_ascii=False, indent=2, default=float))
