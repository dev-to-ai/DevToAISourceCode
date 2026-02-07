########## Three. Pandas Reading & Writing Data ##########

import pandas as pd

# 1. Create the students.csv CSV File in the Project Root Folder
"""
student_id,name,age,gender,city,score,attendance
1001,Alice,23,F,Toronto,88,0.92
1002,Bob,27,M,Vancouver,92,0.85
1003,Charlie,,M,Toronto,79,0.88
1004,David,31,M,Calgary,,0.90
1005,Eva,25,F,Vancouver,95,
1006,Frank,29,M,Toronto,67,0.70
1007,Grace,22,F,Calgary,85,0.95
1008,Helen,34,F,Toronto,90,0.80
1009,Ian,28,M,,73,0.75
1010,Jane,26,F,Vancouver,89,0.93
"""

"""
Why this file is GOOD

It has:
missing values (age, score, attendance, city)
mixed types (int, float, string)
categorical data (city, gender)
numeric features (perfect for ML later)

This is real-world data, not tutorial data.
"""

# 2. Reading the CSV

df = pd.read_csv("students.csv")
print(df)
print(df.head()) # Get the top 5 rows
print(df.info()) # Get a quick technical summary of the DataFrame
print(df.describe()) # Get a summary / statistics of numeric columns (by default) in the DataFrame.
print(df.dtypes)

# 3. Writing Data Back to CSV
# Create a new file
df.to_csv("students_cleaned.csv", index=False)
""" EXPLANATION
Why index=False?
Index is usually not real data
Avoid polluting files with artificial IDs
"""
# Add a new row
new_row = {
    "student_id": 1011,
    "name": "Kevin",
    "age": 24,
    "gender": "M",
    "city": "Montreal",
    "score": 87,
    "attendance": 0.88
}
df = pd.read_csv("students_cleaned.csv")
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print(df)

# 4. Other Common File Formats
# pip install openpyxl
df = pd.read_excel("students.xlsx")
print(df.head())
df = pd.read_json("students.json")
print(df.tail())