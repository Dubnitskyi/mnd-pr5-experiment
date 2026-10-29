import numpy as np
import pandas as pd

SEED = 2026
N_PER_GROUP = 40

rng = np.random.default_rng(SEED)

without_ai = rng.lognormal(mean=np.log(46), sigma=0.28, size=N_PER_GROUP)
with_ai = rng.lognormal(mean=np.log(39), sigma=0.30, size=N_PER_GROUP)

df = pd.DataFrame({
    "participant_id": [f"P{i:03d}" for i in range(1, 2 * N_PER_GROUP + 1)],
    "group": ["without_ai"] * N_PER_GROUP + ["with_ai"] * N_PER_GROUP,
    "experience_years": rng.integers(1, 8, size=2 * N_PER_GROUP),
    "time_min": np.round(np.concatenate([without_ai, with_ai]), 1),
})

df.loc[[5, 47, 63], "time_min"] = np.nan
df.loc[18, "time_min"] = 412.0
df.loc[71, "time_min"] = 0.0

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.to_csv("data/task_time.csv", index=False)
print(df.groupby("group")["time_min"].describe())
