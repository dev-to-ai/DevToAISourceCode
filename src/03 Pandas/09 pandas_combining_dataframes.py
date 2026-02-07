########## Nine. Combining DataFrames ##########

# Combining DataFrames = where real-world data work starts.

import pandas as pd

# 1. Concatenation (pd.concat)
df1 = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "score": [85, 90]
})
df2 = pd.DataFrame({
    "name": ["Charlie", "David"],
    "score": [88, 92]
})

# Vertical concat (axis=0)
print(pd.concat([df1, df2], axis=0))
""" PRINT RESULT:
      name  score
0    Alice     85
1      Bob     90
0  Charlie     88
1    David     92
"""

# Vertical concat (axis=0) and resets index
print(pd.concat([df1, df2], axis=0, ignore_index=True))
""" PRINT RESULT:
      name  score
0    Alice     85
1      Bob     90
2  Charlie     88
3    David     92
"""

# Horizontal concat (axis=1) — add columns
df3 = pd.DataFrame({"age": [25, 30]})
print(pd.concat([df1, df3], axis=1))
""" PRINT RESULT:
    name  score  age
0  Alice     85   25
1    Bob     90   30
"""

# 2. Merge (pd.merge) — SQL-style joins (Combine tables based on keys)
orders = pd.DataFrame({
    "order_id": [1, 2, 3],
    "customer_id": [101, 102, 104],
    "amount": [50, 75, 100]
})
customers = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Charlie"]
})

# INNER JOIN (default)
print(pd.merge(orders, customers, on="customer_id", how="inner"))
""" PRINT RESULT:
   order_id  customer_id  amount   name
0         1          101      50  Alice
1         2          102      75    Bob
"""

# LEFT JOIN (most common in analytics)
print(pd.merge(orders, customers, on="customer_id", how="left"))
""" PRINT RESULT:
   order_id  customer_id  amount   name
0         1          101      50  Alice
1         2          102      75    Bob
2         3          104     100    NaN
"""
# EXPLANATION: Orders are primary, customers optional

# RIGHT JOIN
print(pd.merge(orders, customers, on="customer_id", how="right"))
""" PRINT RESULT:
   order_id  customer_id  amount     name
0       1.0          101    50.0    Alice
1       2.0          102    75.0      Bob
2       NaN          103     NaN  Charlie
"""
# EXPLANATION: Keep all customers

# OUTER JOIN
print(pd.merge(orders, customers, on="customer_id", how="outer"))
""" PRINT RESULT:
   order_id  customer_id  amount     name
0       1.0          101    50.0    Alice
1       2.0          102    75.0      Bob
2       NaN          103     NaN  Charlie
3       3.0          104   100.0      NaN
"""
# EXPLANATION: Keeps everything

# Merge on different column names
print(
    pd.merge(
        orders,
        customers,
        left_on="customer_id",
        right_on="customer_id"
    )
)
""" PRINT RESULT:
   order_id  customer_id  amount   name
0         1          101      50  Alice
1         2          102      75    Bob
"""

# Detect unmatched rows (data quality checks)
print(
    pd.merge(
        orders,
        customers,
        on="customer_id",
        how="left",
        indicator=True
    )
)
""" PRINT RESULT:
   order_id  customer_id  amount   name     _merge
0         1          101      50  Alice       both
1         2          102      75    Bob       both
2         3          104     100    NaN  left_only
"""

# 3. Summary:
"""
Task	                Use
Stack rows	            concat(axis=0)
Add columns by index	concat(axis=1)
SQL-style join	        merge
Match by ID	            merge
Combine files	        concat
"""


