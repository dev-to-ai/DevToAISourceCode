########## Eight. Sorting, Ranking, Sampling ##########

# Setup
import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Minimal cleaning for consistency
df["age"] = df["age"].fillna(df["age"].mean())
df["score"] = df["score"].fillna(df["score"].median())
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
df["city"] = df["city"].fillna("Unknown")

# 1. Sorting (Order Data)

# Sort by one column (ascending = default)
df_sorted_age = df.sort_values("age")
print(df_sorted_age[["name", "age"]])
""" PRINT RESULT:
      name        age
6    Grace  22.000000
0    Alice  23.000000
4      Eva  25.000000
9     Jane  26.000000
1      Bob  27.000000
2  Charlie  27.222222
8      Ian  28.000000
5    Frank  29.000000
3    David  31.000000
7    Helen  34.000000
"""

# Sort by one column (descending)
df_sorted_score = df.sort_values("score", ascending=False)
print(df_sorted_score[["name", "score"]])
""" PRINT RESULT:
      name  score
4      Eva   95.0
1      Bob   92.0
7    Helen   90.0
9     Jane   89.0
0    Alice   88.0
3    David   88.0
6    Grace   85.0
2  Charlie   79.0
8      Ian   73.0
5    Frank   67.0
"""

# Sort by multiple columns
print(df.sort_values(
    by=["city", "score"],
    ascending=[True, False]
))
""" PRINT RESULT:
   student_id     name        age gender       city  score  attendance
3        1004    David  31.000000      M    Calgary   88.0    0.900000
6        1007    Grace  22.000000      F    Calgary   85.0    0.950000
7        1008    Helen  34.000000      F    Toronto   90.0    0.800000
0        1001    Alice  23.000000      F    Toronto   88.0    0.920000
2        1003  Charlie  27.222222      M    Toronto   79.0    0.880000
5        1006    Frank  29.000000      M    Toronto   67.0    0.700000
8        1009      Ian  28.000000      M    Unknown   73.0    0.750000
4        1005      Eva  25.000000      F  Vancouver   95.0    0.853333
1        1002      Bob  27.000000      M  Vancouver   92.0    0.850000
9        1010     Jane  26.000000      F  Vancouver   89.0    0.930000
"""
""" EXPLANATION:
group by city (A → Z)
highest score first within each city
"""

# Sort and reset index
df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
print(df_sorted)
""" PRINT RESULT:
   student_id     name        age gender       city  score  attendance
0        1005      Eva  25.000000      F  Vancouver   95.0    0.853333
1        1002      Bob  27.000000      M  Vancouver   92.0    0.850000
2        1008    Helen  34.000000      F    Toronto   90.0    0.800000
3        1010     Jane  26.000000      F  Vancouver   89.0    0.930000
4        1001    Alice  23.000000      F    Toronto   88.0    0.920000
5        1004    David  31.000000      M    Calgary   88.0    0.900000
6        1007    Grace  22.000000      F    Calgary   85.0    0.950000
7        1003  Charlie  27.222222      M    Toronto   79.0    0.880000
8        1009      Ian  28.000000      M    Unknown   73.0    0.750000
9        1006    Frank  29.000000      M    Toronto   67.0    0.700000
"""

# 2. Ranking (Label Importance)

# Simple ranking
df["score_rank"] = df["score"].rank(ascending=False)
print(df[["name", "score", "score_rank"]])
""" PRINT RESULT:
      name  score  score_rank
0    Alice   88.0         5.5
1      Bob   92.0         2.0
2  Charlie   79.0         8.0
3    David   88.0         5.5
4      Eva   95.0         1.0
5    Frank   67.0        10.0
6    Grace   85.0         7.0
7    Helen   90.0         3.0
8      Ian   73.0         9.0
9     Jane   89.0         4.0
"""
""" EXPLANATION:
rank() does NOT reorder rows
It only computes a rank number and keeps the original row order
Alice and David both have score 88, which occupy ranks 5 and 6, so pandas assigns:
(5 + 6) / 2 = 5.5
"""

# ranking and then sorting
df["score_rank"] = df["score"].rank(ascending=False)
df_sorted = df.sort_values("score_rank")
print(df_sorted[["name", "score", "score_rank"]])
""" PRINT RESULT:
      name  score  score_rank
4      Eva   95.0         1.0
1      Bob   92.0         2.0
7    Helen   90.0         3.0
9     Jane   89.0         4.0
3    David   88.0         5.5
0    Alice   88.0         5.5
6    Grace   85.0         7.0
2  Charlie   79.0         8.0
8      Ian   73.0         9.0
5    Frank   67.0        10.0
"""

# Ranking methods (ties matter!)
df["rank_min"] = df["score"].rank(method="min", ascending=False)
print(df["rank_min"])
""" PRINT RESULT:
0     5.0
1     2.0
2     8.0
3     5.0
4     1.0
5    10.0
6     7.0
7     3.0
8     9.0
9     4.0
Name: rank_min, dtype: float64
"""
""" EXPLANATION:
min: Because two people share rank 5, the next rank jumps to 7.
"""

df["rank_dense"] = df["score"].rank(method="dense", ascending=False)
print(df["rank_dense"])
""" PRINT RESULT:
0    5.0
1    2.0
2    7.0
3    5.0
4    1.0
5    9.0
6    6.0
7    3.0
8    8.0
9    4.0
Name: rank_dense, dtype: float64
"""
""" EXPLANATION:
dense: no gaps. 5, 5 → next is 6
"""
""" Summary for Methods
Method	    Behavior
average	    default
min	        lowest rank in tie
dense	    no gaps
first	    order-based
"""

# 3. Sampling (Select Subsets)

# Randomly sample N rows
df_sample_2 = df.sample(2)
print(df_sample_2)
""" PRINT RESULT:
   student_id   name   age gender  ... attendance  score_rank  rank_min  rank_dense
6        1007  Grace  22.0      F  ...       0.95         7.0       7.0         6.0
1        1002    Bob  27.0      M  ...       0.85         2.0       2.0         2.0

[2 rows x 10 columns]
"""

# Sample a fraction of the data
df_sample_80 = df.sample(frac=0.8, random_state=42)
print(df_sample_80)
""" PRINT RESULT:
   student_id     name        age gender  ... attendance  score_rank  rank_min  rank_dense
8        1009      Ian  28.000000      M  ...   0.750000         9.0       9.0         8.0      
1        1002      Bob  27.000000      M  ...   0.850000         2.0       2.0         2.0      
5        1006    Frank  29.000000      M  ...   0.700000        10.0      10.0         9.0      
0        1001    Alice  23.000000      F  ...   0.920000         5.5       5.0         5.0      
7        1008    Helen  34.000000      F  ...   0.800000         3.0       3.0         3.0      
2        1003  Charlie  27.222222      M  ...   0.880000         8.0       8.0         7.0      
9        1010     Jane  26.000000      F  ...   0.930000         4.0       4.0         4.0      
4        1005      Eva  25.000000      F  ...   0.853333         1.0       1.0         1.0      

[8 rows x 10 columns]
"""
""" EXPLANATION:
Randomly pick 80% of the rows from df, and make the randomness repeatable.
sample(): randomly select rows
frac (0.8): fraction (80%) of rows
Rows are chosen without replacement by default (no duplicates)
random_state=42: This is the seed for randomness.
It guarantees:
You get the same 80% every time
Anyone else running the code also gets the same rows
Without it: df.sample(frac=0.8), you’d get different rows each run
"""

# Train / Validation Split (Simple & Practical)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
print(val_df)
""" PRINT RESULT:
   student_id   name   age gender     city  score  attendance  score_rank  rank_min  rank_dense
3        1004  David  31.0      M  Calgary   88.0        0.90         5.5       5.0         5.0 
6        1007  Grace  22.0      F  Calgary   85.0        0.95         7.0       7.0         6.0
"""
""" EXPLANATION:
80% training data
20% validation data
no overlap
"""

# Sorting + Sampling Together (Advanced Pattern)
# Top 80% by score
top_students = (
    df.sort_values("score", ascending=False)
      .head(int(len(df) * 0.8))
)
print(top_students)
""" PRINT RESULT:
   student_id     name        age gender  ... attendance  score_rank  rank_min  rank_dense
4        1005      Eva  25.000000      F  ...   0.853333         1.0       1.0         1.0      
1        1002      Bob  27.000000      M  ...   0.850000         2.0       2.0         2.0      
7        1008    Helen  34.000000      F  ...   0.800000         3.0       3.0         3.0      
9        1010     Jane  26.000000      F  ...   0.930000         4.0       4.0         4.0      
0        1001    Alice  23.000000      F  ...   0.920000         5.5       5.0         5.0      
3        1004    David  31.000000      M  ...   0.900000         5.5       5.0         5.0      
6        1007    Grace  22.000000      F  ...   0.950000         7.0       7.0         6.0      
2        1003  Charlie  27.222222      M  ...   0.880000         8.0       8.0         7.0      

[8 rows x 10 columns]
"""
""" EXPLANATION:
Sort all students by score (highest first), then keep only the top 80%.
len(df):Total number of rows (students), 10
int(len(df) * 0.8): Calculates 80% of the dataset. Converts to integer (required by .head()). 8
.head(8): Takes the first 8 rows. Since data is already sorted by score, these are the top 80% by score
"""