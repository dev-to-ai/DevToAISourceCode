########## Seven. Aggregation & Grouping ##########

"""
groupby() = split → apply → combine

Split data into groups

Apply aggregation to each group

Combine results into a new DataFrame / Series
"""

# Setup (Ensure grade Exists)
import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Ensure missing data is handled
df["age"] = df["age"].fillna(df["age"].mean())
df["score"] = df["score"].fillna(df["score"].median())
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
df["city"] = df["city"].fillna("Unknown")

# Create grade column (vectorized)
df["grade"] = np.where(df["score"] >= 90, "A", "B")
print(df)
""" PRINT RESULT:
   student_id     name        age gender       city  score  attendance grade
0        1001    Alice  23.000000      F    Toronto   88.0    0.920000     B
1        1002      Bob  27.000000      M  Vancouver   92.0    0.850000     A
2        1003  Charlie  27.222222      M    Toronto   79.0    0.880000     B
3        1004    David  31.000000      M    Calgary   88.0    0.900000     B
4        1005      Eva  25.000000      F  Vancouver   95.0    0.853333     A
5        1006    Frank  29.000000      M    Toronto   67.0    0.700000     B
6        1007    Grace  22.000000      F    Calgary   85.0    0.950000     B
7        1008    Helen  34.000000      F    Toronto   90.0    0.800000     A
8        1009      Ian  28.000000      M    Unknown   73.0    0.750000     B
9        1010     Jane  26.000000      F  Vancouver   89.0    0.930000     B
"""

# 1. Average score per grade
avg_score_by_grade = df.groupby("grade")["score"].mean()
print(avg_score_by_grade)
""" PRINT RESULT:
grade
A    92.333333
B    81.285714
Name: score, dtype: float64
"""

# 2. Multiple Aggregations with .agg()
summary = df.groupby("grade").agg({
    "score": ["mean", "max"],
    "age": "mean"
})
print(summary)
""" PRINT RESULT:
           score              age
            mean   max       mean
grade
A      92.333333  95.0  28.666667
B      81.285714  89.0  26.603175
"""

# 3. Flatten Column Names
summary.columns = ["score_mean", "score_max", "age_mean"]
print(summary)
""" PRINT RESULT:
       score_mean  score_max   age_mean
grade
A       92.333333       95.0  28.666667
B       81.285714       89.0  26.603175
"""

# 4. Grouping by Multiple Columns
print(df.groupby(["grade", "city"])["score"].mean())
""" PRINT RESULT:
grade  city
A      Toronto      90.0
       Vancouver    93.5
B      Calgary      86.5
       Toronto      78.0
       Unknown      73.0
       Vancouver    89.0
Name: score, dtype: float64
"""

# 5. Groupby + Filtering (Removes cities - Calgary and Unknown - with fewer than 3 students)
print(df.groupby("city").filter(lambda x: len(x) >= 3))
""" PRINT RESULT:
   student_id     name        age gender       city  score  attendance grade
0        1001    Alice  23.000000      F    Toronto   88.0    0.920000     B
1        1002      Bob  27.000000      M  Vancouver   92.0    0.850000     A
2        1003  Charlie  27.222222      M    Toronto   79.0    0.880000     B
4        1005      Eva  25.000000      F  Vancouver   95.0    0.853333     A
5        1006    Frank  29.000000      M    Toronto   67.0    0.700000     B
7        1008    Helen  34.000000      F    Toronto   90.0    0.800000     A
9        1010     Jane  26.000000      F  Vancouver   89.0    0.930000     B
"""

# 6. Groupby for Feature Engineering
# Mean score per city
city_mean_score = df.groupby("city")["score"].mean()
print(city_mean_score)
""" PRINT RESULT:
city
Calgary      86.5
Toronto      81.0
Unknown      73.0
Vancouver    92.0
Name: score, dtype: float64
"""
# Map back to original DataFrame
df["city_avg_score"] = df["city"].map(city_mean_score)
print(df)
""" PRINT RESULT:
   student_id     name        age gender       city  score  attendance grade  city_avg_score    
0        1001    Alice  23.000000      F    Toronto   88.0    0.920000     B            81.0    
1        1002      Bob  27.000000      M  Vancouver   92.0    0.850000     A            92.0    
2        1003  Charlie  27.222222      M    Toronto   79.0    0.880000     B            81.0    
3        1004    David  31.000000      M    Calgary   88.0    0.900000     B            86.5    
4        1005      Eva  25.000000      F  Vancouver   95.0    0.853333     A            92.0    
5        1006    Frank  29.000000      M    Toronto   67.0    0.700000     B            81.0    
6        1007    Grace  22.000000      F    Calgary   85.0    0.950000     B            86.5    
7        1008    Helen  34.000000      F    Toronto   90.0    0.800000     A            81.0    
8        1009      Ian  28.000000      M    Unknown   73.0    0.750000     B            73.0    
9        1010     Jane  26.000000      F  Vancouver   89.0    0.930000     B            92.0
"""

# 7. Common Aggregation Functions
"""
mean()
sum()
count()
min()
max()
median()
std()
"""