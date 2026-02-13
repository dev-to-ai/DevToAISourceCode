########## One. Python Lists for AI ##########

"""
A Python list is an ordered, mutable collection of items that can hold elements of different data types.

Think of it as a flexible container that can grow, shrink, and change as needed, unlike arrays in many other languages that are fixed in size and type.
"""

# 1. List Properties

# 1.1 Written using square brackets [] and separated by commas
# 1.2 Able to contain different data types (int, string, float, boolean)
# 1.3 Allow duplicate values ("Python", "Python")
# 1.4 Able to contain nested lists [10, 20]
my_list = ["Python", 2026, 3.14, True, "Python", [10, 20]]
print(my_list) 
# PRINT RESULT:  ['Python', 2026, 3.14, True, 'Python', [10, 20]]

# 1.5 Ordered: Index starts at 0
print(my_list[0])
# PRINT RESULT: Python
print(my_list[2])
# PRINT RESULT: 3.14
print(my_list[5][0]) 
# PRINT RESULT: 10

# 1.6 Mutable: Items can be changed
print(f"Original list: {my_list}")
# PRINT RESULT: Original list: ['Python', 2026, 3.14, True, 'Python', [10, 20]]
my_list[4] = "AI"
my_list[5][1] = "ABC"
print(f"Modified list: {my_list}") 
# PRINT RESULT: Modified list: ['Python', 2026, 3.14, True, 'AI', [10, 'ABC']]

# IMPORTANT: Shallow vs Deep Copy
# When working with nested lists in AI, copying requires attention
original_nested = [[1, 2], [3, 4]]
shallow_copy = original_nested.copy()  # Outer list is new, inner lists are shared
shallow_copy[0][0] = 99
print(f"Original after shallow copy mod: {original_nested}")  
# PRINT RESULT: Original after shallow copy mod: [[99, 2], [3, 4]]
# Changed

# For AI work with nested data (like image batches), use deep copy
import copy
deep_copy = copy.deepcopy(original_nested)
deep_copy[0][0] = 100
print(f"Original after deep copy mod: {original_nested}")
# Original after deep copy mod: [[99, 2], [3, 4]]
# Unchanged

# 2. Essential Built-in List Methods

# 2.1 .append(value) - Adds a single item to the end
my_list.append("New Data")
print(my_list) 
# PRINT RESULT: ['Python', 2026, 3.14, True, 'AI', [10, 'ABC'], 'New Data']

# 2.2 .extend(iterable) - Adds all items from another list to the end
# Merging a new batch of data into your main dataset
extra_info = ["Deep Learning", 99.9]
my_list.extend(extra_info)
print(my_list) 
# PRINT RESULT: ['Python', 2026, 3.14, True, 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]
# append adds the object as one item (even if it's a list), while extend "unpacks" the list and adds each element individually.

# Bonus: The * operator for list creation
# Useful for creating baseline arrays in AI
initial_weights = [0.0] * 5  # Creates [0.0, 0.0, 0.0, 0.0, 0.0]
print(f"Initial weights for 5 neurons: {initial_weights}")
# PRINT RESULT: Initial weights for 5 neurons: [0.0, 0.0, 0.0, 0.0, 0.0]
# WARNING: Great for immutable types (int, float, str), but careful with mutable objects!
# bad_matrix = [[]] * 3  # Creates 3 references to the SAME empty list - DON'T DO THIS!

# 2.3 .insert(index, value) - Adds an item at a specific position
# Inserting a 'Class ID' at the beginning of a data row
my_list.insert(0, "ID_001")
print(my_list[0]) 
# PRINT RESULT: ID_001

# 2.4 .remove(value) - Removes the FIRST occurrence of a value
# Removing an unwanted outlier or 'None' value from data
# Insert one first
my_list.insert(5, "Python") # Add a new 'Python' at the sixth position
print(my_list) 
# PRINT RESULT: ['ID_001', 'Python', 2026, 3.14, True, 'Python', 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]
my_list.remove("Python") # Remove the FIRST 'Python'
print(my_list) 
# PRINT RESULT: ['ID_001', 2026, 3.14, True, 'Python', 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]

# 2.5 .pop(index) - Removes and RETURNS the item at the index (default is last)
# Taking the last label out of a list to use it for validation
removed_item = my_list.pop(1)
print(removed_item) 
# 2026
# Use remove if you know the value you want to get rid of; use pop if you know the position (index) of the item.

# 2.6 .index(value) - Returns the FIRST index of a value
# Finding where a specific label is located
position = my_list.index(3.14)
print(position) 
# PRINT RESULT: 1

# 2.7 .count(value) - Counts how many times a value appears
# Checking for class imbalance (e.g., how many 'True' labels exist)
print(my_list.count(True)) 
# PRINT RESULT: 1

# 2.8 Performance Considerations for AI Engineers
# When processing millions of samples, time complexity matters!

"""
Time Complexities Every AI Engineer Should Know:

O(1) - Fast (constant time):
    list.append(item)     # Adding to end
    list[i]               # Access by index
    list.pop()            # Remove from end

O(n) - Slow for large lists (linear time):
    list.insert(0, item)  # Inserting at beginning
    list.remove(item)     # Finding and removing
    item in list          # Membership testing
    list.index(item)      # Finding position
    min(list)/max(list)   # Finding extremes
"""

# Example: Why this matters
large_list = list(range(10_000_000))
# Fast: direct access
first_item = large_list[0]  # O(1) - instant
# Slow: membership test
has_item = 9_999_999 in large_list  # O(n) - checks every element!

# 3. Global Functions for AI Data

# Imagine these are our dataset components
image_ids = [101, 102, 103, 104]
labels = ["Cat", "Dog", "Cat", "Bird"]
confidence_scores = [0.98, 0.85, 0.92, 0.77]

# 3.1 len() - Checks data consistency
# Essential for verifying that you have as many labels as you do images
if len(image_ids) == len(labels) == len(confidence_scores):
    print(f"Dataset is consistent. Total samples: {len(image_ids)}")
# PRINT RESULT: Dataset is consistent. Total samples: 4

# 3.2 zip() - Combines lists into pairs
# Used to bind features to labels so they stay together during shuffling
dataset = list(zip(image_ids, labels, confidence_scores))
print(dataset) 
# PRINT RESULT: [(101, 'Cat', 0.98), (102, 'Dog', 0.85), (103, 'Cat', 0.92), (104, 'Bird', 0.77)]
print(f"First paired sample: {dataset[0]}") 
# PRINT RESULT: First paired sample: (101, 'Cat', 0.98)

# 3.3 enumerate() - Tracks index during iteration
# Useful for logging progress or identifying which specific file failed
print("\nProcessing Batch:")
for index, (img, lbl, score) in enumerate(dataset):
    print(f"Sample #{index}: Classifying {img} as {lbl} and its confidence score is {score}.")
""" PRINT RESULT:
Processing Batch:
Sample #0: Classifying 101 as Cat and its confidence score is 0.98.
Sample #1: Classifying 102 as Dog and its confidence score is 0.85.
Sample #2: Classifying 103 as Cat and its confidence score is 0.92.
Sample #3: Classifying 104 as Bird and its confidence score is 0.77.
"""

# 3.4 min() and max() - Inspects data ranges
# Vital for knowing if your data needs "Scaling" (Normalization)
highest_score = max(confidence_scores)
lowest_score = min(confidence_scores)
print(f"\nModel Confidence Range: {lowest_score} to {highest_score}")
# PRINT RESULT: Model Confidence Range: 0.77 to 0.98

# 3.5 sum() and sorted() - Quick statistics
# sum() can calculate total loss; sorted() can rank your best predictions
avg_confidence = sum(confidence_scores) / len(confidence_scores)
print(f"Average Confidence: {avg_confidence:.2f}") 
# PRINT RESULT: Average Confidence: 0.88
ranked_scores = sorted(confidence_scores, reverse=True)
print(f"Ranked Scores: {ranked_scores}") 
# PRINT RESULT: Ranked Scores: [0.98, 0.92, 0.85, 0.77]
print(f"Top 3 Scores: {ranked_scores[:3]}") 
# PRINT RESULT: Top 3 Scores: [0.98, 0.92, 0.85]

# 4. List Comprehensions (The AI Secret Weapon)

# 4.1 The Transformation: Scaling data in AI, we rarely feed raw numbers into a model. We often scale them between 0 and 1.
# Raw pixel values from an image (0 to 255)
raw_pixels = [0, 51, 102, 153, 204, 255]
# The "Old" way: 4 lines of code
scaled_pixels_old = []
for p in raw_pixels:
    scaled_pixels_old.append(p / 255.0)
# The AI "Secret Weapon" way: 1 line of code
# Syntax: [expression for item in list]
scaled_pixels = [p / 255.0 for p in raw_pixels]
print(f"Scaled Data: {scaled_pixels}")
# PRINT RESULT: Scaled Data: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 4.2 The Filter: Removing Low-Confidence Results
# Often, an AI model outputs many predictions, but you only want to keep the ones it is "sure" about.
predictions = [0.98, 0.12, 0.45, 0.88, 0.30, 0.92]
# Only keep predictions with > 50% confidence
# Syntax: [expression for item in list if condition]
high_confidence = [p for p in predictions if p > 0.50]
print(f"High Confidence Predictions: {high_confidence}") 
# PRINT RESULT: High Confidence Predictions: [0.98, 0.88, 0.92]

# 4.3 The Multi-Tasker: Cleaning Text Data
# If you are working with Natural Language Processing (NLP), you often need to lowercase text and strip whitespace simultaneously.
raw_labels = ["  Cat", "dog  ", " FISH ", "cat "]
# Lowercase everything and remove empty spaces
clean_labels = [label.strip().lower() for label in raw_labels]
print(f"Cleaned Labels: {clean_labels}") 
# PRINT RESULT: Cleaned Labels: ['cat', 'dog', 'fish', 'cat']

# 4.4 Advanced: The If-Else (Categorization)
# You can even use list comprehensions to create binary labels (converting numbers to "Pass/Fail").
scores = [0.8, 0.4, 0.9, 0.3]
# If score >= 0.5 then 1 (Pass), else 0 (Fail)
binary_labels = [1 if s >= 0.5 else 0 for s in scores]
print(f"Binary Labels: {binary_labels}") 
# PRINT RESULT: Binary Labels: [1, 0, 1, 0]

# 5. Advanced: Slicing for Data Splitting

# 5.1 The Basic Split (Train/Test)
# The most common split is the 80/20 rule: 80% for training, 20% for testing.
# A dataset of 10 samples (could be image paths or text strings)
dataset = ["img1", "img2", "img3", "img4", "img5", "img6", "img7", "img8", "img9", "img10"]
# Syntax: list[start:stop] -> 'stop' is not included
train_data = dataset[:8]  # Indices 0 through 7
test_data = dataset[8:]   # Indices 8 through the end
print(f"Training set (80%): {train_data}") 
# PRINT RESULT: Training set (80%): ['img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8']
print(f"Testing set (20%):  {test_data}") 
# PRINT RESULT: Testing set (20%):  ['img9', 'img10']

# 5.2 The "Validation" Split (The 3-Way Split)
# Senior engineers use a validation set to tune the model while it's training, keeping the test set purely for the final grade.
# Splitting 70% Train, 20% Validation, 10% Test
train = dataset[:7]    # First 7 items
val   = dataset[7:9]   # Items at index 7 and 8
test  = dataset[9:]    # Item at index 9
print(f"Train: {train} | Val: {val} | Test: {test}") 
# PRINT RESULT: Train: ['img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7'] | Val: ['img8', 'img9'] | Test: ['img10']

# 5.3 Advanced Slicing: The "Step" Function
# Sometimes you want to downsample a dataset (e.g., taking every 2nd frame of a video) to save processing time.
# Syntax: list[start:stop:step]
# The step is 2. Take every 2nd item from the entire dataset
downsampled_data = dataset[::2]
print(f"Every 2nd sample: {downsampled_data}") 
# Every 2nd sample: ['img1', 'img3', 'img5', 'img7', 'img9']
# The step is -1. Reverse the dataset (occasionally used for data augmentation)
reversed_data = dataset[::-1]
print(f"Reversed data: {reversed_data}") 
# PRINT RESULT: Reversed data: ['img10', 'img9', 'img8', 'img7', 'img6', 'img5', 'img4', 'img3', 'img2', 'img1']

# 5.4 Handling 2D Data (Nested Lists)
# In AI, your data is often a list of lists (e.g., [feature_1, feature_2, label]). Slicing helps you separate features ($X$) from labels ($y$).
# Each row: [Feature_A, Feature_B, Label]
rows = [
    [0.1, 0.2, "Cat"],
    [0.5, 0.6, "Dog"],
    [0.9, 0.1, "Cat"]
]
# Get only the first two rows
subset = rows[:2]
print(f"First two data rows: {subset}") 
# PRINT RESULT: First two data rows: [[0.1, 0.2, 'Cat'], [0.5, 0.6, 'Dog']]

# 6. Putting It All Together: Real AI Pipeline

"""
A complete mini AI pipeline using only lists:
1. Load raw data
2. Clean and preprocess
3. Filter low-quality samples
4. Separate features from labels
5. Prepare for model training
"""

# Raw data from various sources (image files, sensors, etc.)
raw_data = [
    [" image1.jpg ", 0.95, 0.23, "cat"],
    ["image2.jpg  ", 0.45, 0.67, "dog"],
    ["  image3.jpg", 0.88, 0.12, "cat"],
    ["image4.jpg", 0.32, 0.55, "dog"],
    ["  image5.jpg  ", 0.91, 0.34, "cat"],
]

print("Step 1: Raw Data Sample:")
print(raw_data[:2])
""" PRINT RESULT:
Step 1: Raw Data Sample:
[[' image1.jpg ', 0.95, 0.23, 'cat'], ['image2.jpg  ', 0.45, 0.67, 'dog']]
"""

# Step 2: Clean data (strip strings, standardize labels)
cleaned_data = [
    [fname.strip(), feat1, feat2, label.strip().lower()]
    for fname, feat1, feat2, label in raw_data
]

print("\nStep 2: Cleaned Data:")
print(cleaned_data[:2])
""" PRINT RESULT:
Step 2: Cleaned Data:
[['image1.jpg', 0.95, 0.23, 'cat'], ['image2.jpg', 0.45, 0.67, 'dog']]
"""

# Step 3: Filter - keep only high confidence samples (feat1 > 0.5)
high_conf_data = [
    item for item in cleaned_data 
    if item[1] > 0.5  # Confidence threshold
]
# high_conf_data is [['image1.jpg', 0.95, 0.23, 'cat'], ['image3.jpg', 0.88, 0.12, 'cat'], ['image5.jpg', 0.91, 0.34, 'cat']]


print(f"\nStep 3: High confidence samples ({len(high_conf_data)} of {len(raw_data)}):")
for item in high_conf_data:
    print(f"  {item[0]} - {item[3]} (conf: {item[1]:.2f})")
""" PRINT RESULT:
Step 3: High confidence samples (3 of 5):
  image1.jpg - cat (conf: 0.95)
  image3.jpg - cat (conf: 0.88)
  image5.jpg - cat (conf: 0.91)
"""

# Step 4: Separate features (X) and labels (y) for model training
X = [[fname, feat1, feat2] for fname, feat1, feat2, _ in high_conf_data]
# _: Means "ignore this"
"""
X = [
    ["image1.jpg", 0.95, 0.23],  # Only features, no label
    ["image3.jpg", 0.88, 0.12],  # Only features, no label
    ["image5.jpg", 0.91, 0.34]   # Only features, no label
]
"""
y = [label for _, _, _, label in high_conf_data]
# y = ["cat", "cat", "cat"]

print(f"\nStep 4: Ready for training!")
print(f"Features (X): {len(X)} samples, each with 3 features")
print(f"Labels (y): {len(y)} samples: {set(y)}")
# set(y)  # Returns: {"cat"}  (a set with one unique value)
"""
# If y had mixed labels:
y_mixed = ["cat", "dog", "cat", "bird", "dog"]
set(y_mixed)  # Returns: {"cat", "dog", "bird"}
"""
""" PRINT RESULT:
Step 4: Ready for training!
Features (X): 3 samples, each with 3 features
Labels (y): 3 samples: {'cat'}
"""

# Step 5: Train/validation split (80/20)
# This is the 80/20 train-validation split - one of the most fundamental patterns in machine learning
# Calculate where to cut
split_idx = int(len(X) * 0.8)
# len(X) = 3
# 3 * 0.8 = 2.4
# int(2.4) = 2
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
"""
# Training set - first 2 samples (indices 0 and 1)
X_train = X[:2]  # Same as X[0:2]
# Result: [["image1.jpg", 0.95, 0.23], ["image3.jpg", 0.88, 0.12]]

y_train = y[:2]  # Same as y[0:2]
# Result: ["cat", "cat"]

# Validation set - from index 2 to the end
X_val = X[2:]  # Same as X[2:len(X)]
# Result: [["image5.jpg", 0.91, 0.34]]

y_val = y[2:]  # Same as y[2:len(y)]
# Result: ["cat"]
"""

print(f"\nStep 5: Data split complete")
print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
""" PRINT RESULT:
Step 5: Data split complete
  Training: 2 samples
  Validation: 1 samples
"""

# Bonus: Create binary labels for classification
binary_labels = [1 if label == "cat" else 0 for label in y]
print(f"\nBonus: Binary encoding (cat=1, dog=0): {binary_labels}")
""" PRINT RESULT:
Bonus: Binary encoding (cat=1, dog=0): [1, 1, 1]
"""