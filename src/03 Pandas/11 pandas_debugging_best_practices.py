########## Debugging & Best Practices (Pandas → ML) ##########

# 1. Common Mistakes

# ❌ Looping over rows
for i, row in df.iterrows():
    df.loc[i, "score"] = row["score"] * 1.1
# 💥 Problems: Slow (Python loop), easy to bug, not scalable
# ✅ Vectorized version
df["score"] = df["score"] * 1.1
# Think column-wise, not row-wise.

# ❌ Ignoring NaNs
X = df[features].to_numpy()
model.fit(X, y)   # 💥 crash or silent garbage
# NaNs cause: Model failures, biased results, hidden bugs
# ✅ Always check
df.isna().sum()
# Fix before modeling:
df[features] = df[features].fillna(df[features].mean())

# ❌ Mixing .loc and .iloc
df.iloc[df["score"] > 80, 1]  # ❌ WRONG
# Problem:
# .iloc → integer positions only
# .loc → labels + boolean masks
✅ Correct usage
df.loc[df["score"] > 80, "grade"] = "A"
# Boolean mask → always .loc

❌ Modifying views (SettingWithCopyWarning)
df_high = df[df["score"] > 80]
df_high["grade"] = "A"   # ⚠️ warning
# Problem:
# Might not update original df
# Silent data corruption risk
✅ Safe patterns
df.loc[df["score"] > 80, "grade"] = "A"
# or
df_high = df[df["score"] > 80].copy()
df_high["grade"] = "A"

# 2. Golden Rules

# ✅ Prefer vectorized operations
df["total"] = df["price"] * df["qty"]

# ✅ Inspect data early (ALWAYS)
df.head()
df.info()
df.describe()

# ✅ Clean BEFORE modeling
# Never do this:
model.fit(df.values, y)
# Always do this:
features = ["age", "score"]
df[features] = df[features].astype(float)
df[features] = df[features].fillna(df[features].mean())
X = df[features].to_numpy()

# ✅ Keep columns numeric for ML
# ML wants: float32 / float64
❌ Objects / strings
❌ Mixed types
# Encoding comes before .to_numpy().