##### One - List Properties

# 1. Written using square brackets [] and separated by commas
# 2. Able to contain different data types (int, string, float, boolean)
# 3. Allow duplicate values ("Python", "Python")
# 4. Able to contain nested lists [10, 20]
my_list = ["Python", 2026, 3.14, True, "Python", [10, 20]]
# print(my_list)

# 5. Ordered: Index starts at 0
print(my_list[0]) # Python
# print(my_list[2]) # 3.14
print(my_list[5][0]) # 10

# 6. Mutable: Items can be changed
print(f"Original list: {my_list}") 
# Original list: ['Python', 2026, 3.14, True, 'Python', [10, 20]]
my_list[4] = "AI"
my_list[5][1] = "ABC"
print(f"Modified list: {my_list}") 
# Modified list: ['Python', 2026, 3.14, True, 'AI', [10, 'ABC']]

##### Two - Essential Built-in List Methods

# 1. .append(value) - Adds a single item to the end
my_list.append("New Data")
print(my_list) 
# ['Python', 2026, 3.14, True, 'AI', [10, 'ABC'], 'New Data']

# 2. .extend(iterable) - Adds all items from another list to the end
# Use case: Merging a new batch of data into your main dataset
extra_info = ["Deep Learning", 99.9]
my_list.extend(extra_info)
print(my_list) 
# ['Python', 2026, 3.14, True, 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]
# NOTE: 
# append adds the object as one item (even if it's a list), while extend "unpacks" the list and adds each element individually.

# 3. .insert(index, value) - Adds an item at a specific position
# Use case: Inserting a 'Class ID' at the beginning of a data row
my_list.insert(0, "ID_001")
print(my_list[0]) # ID_001

# 4. .remove(value) - Removes the FIRST occurrence of a value
# Use case: Removing an unwanted outlier or 'None' value from data
my_list.insert(5, "Python") # Add a new 'Python' at the sixth position
print(my_list) 
# ['ID_001', 'Python', 2026, 3.14, True, 'Python', 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]
my_list.remove("Python") # Remove the first 'Python'
print(my_list) 
# ['ID_001', 2026, 3.14, True, 'Python', 'AI', [10, 'ABC'], 'New Data', 'Deep Learning', 99.9]

# 5. .pop(index) - Removes and RETURNS the item at the index (default is last)
# Use case: Taking the last label out of a list to use it for validation
removed_item = my_list.pop(1)
print(removed_item) # 2026
# NOTE: 
# Use remove if you know the value you want to get rid of. 
# Use pop if you know the position (index) of the item.

# 6. .index(value) - Returns the first index of a value
# Use case: Finding where a specific label is located
position = my_list.index(3.14)
print(position) # 1

# 7. .count(value) - Counts how many times a value appears
# Use case: Checking for class imbalance (e.g., how many 'True' labels exist)
print(my_list.count(True)) # 1

##### Three - Global Functions for AI Data

# Imagine these are our dataset components
image_ids = [101, 102, 103, 104]
labels = ["Cat", "Dog", "Cat", "Bird"]
confidence_scores = [0.98, 0.85, 0.92, 0.77]

# 1. len() - Check data consistency
# Essential for verifying that you have as many labels as you do images
if len(image_ids) == len(labels) == len(confidence_scores):
    print(f"Dataset is consistent. Total samples: {len(image_ids)}")

# 2. zip() - Combine lists into pairs
# Used to bind features to labels so they stay together during shuffling
dataset = list(zip(image_ids, labels, confidence_scores))
print(dataset) 
# [(101, 'Cat', 0.98), (102, 'Dog', 0.85), (103, 'Cat', 0.92), (104, 'Bird', 0.77)]
print(f"First paired sample: {dataset[0]}") 
# First paired sample: (101, 'Cat', 0.98)

# 3. enumerate() - Track index during iteration
# Useful for logging progress or identifying which specific file failed
print("\nProcessing Batch:")
for index, (img, lbl, score) in enumerate(dataset):
    print(f"Sample #{index}: Classifying {img} as {lbl} and its confidence score is {score}.")
# Processing Batch:
# Sample #0: Classifying 101 as Cat and its confidence score is 0.98.
# Sample #1: Classifying 102 as Dog and its confidence score is 0.85.
# Sample #2: Classifying 103 as Cat and its confidence score is 0.92.
# Sample #3: Classifying 104 as Bird and its confidence score is 0.77.

# 4. min() and max() - Inspect data ranges
# Vital for knowing if your data needs "Scaling" (Normalization)
highest_score = max(confidence_scores)
lowest_score = min(confidence_scores)
print(f"\nModel Confidence Range: {lowest_score} to {highest_score}")
# Model Confidence Range: 0.77 to 0.98

# 5. sum() and sorted() - Quick statistics
# sum() can calculate total loss; sorted() can rank your best predictions
avg_confidence = sum(confidence_scores) / len(confidence_scores)
print(f"Average Confidence: {avg_confidence:.2f}") # Average Confidence: 0.88
ranked_scores = sorted(confidence_scores, reverse=True)
print(f"Ranked Scores: {ranked_scores}") # Ranked Scores: [0.98, 0.92, 0.85, 0.77]
print(f"Top 3 Scores: {ranked_scores[:3]}") # Top 3 Scores: [0.98, 0.92, 0.85]

##### Four - List Comprehensions (The AI Secret Weapon)

# 1. The Transformation: Scaling data in AI, we rarely feed raw numbers into a model. We often scale them between 0 and 1.
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
###Side-by-Side Comparisons: For List Comprehensions, show the 4-line for loop on the left and the 1-line comprehension on the right. Cross out the 4-line version with a red "X."

# 2. The Filter: Removing Low-Confidence Results
# Often, an AI model outputs many predictions, but you only want to keep the ones it is "sure" about.
predictions = [0.98, 0.12, 0.45, 0.88, 0.30, 0.92]
# Only keep predictions with > 50% confidence
# Syntax: [expression for item in list if condition]
high_confidence = [p for p in predictions if p > 0.50]
print(f"High Confidence Predictions: {high_confidence}") 
# High Confidence Predictions: [0.98, 0.88, 0.92]

# 3. The Multi-Tasker: Cleaning Text Data
# If you are working with Natural Language Processing (NLP), you often need to lowercase text and strip whitespace simultaneously.
raw_labels = ["  Cat", "dog  ", " FISH ", "cat "]
# Lowercase everything and remove empty spaces
clean_labels = [label.strip().lower() for label in raw_labels]
print(f"Cleaned Labels: {clean_labels}") # Cleaned Labels: ['cat', 'dog', 'fish', 'cat']

# 4. Advanced: The If-Else (Categorization)
# You can even use list comprehensions to create binary labels (converting numbers to "Pass/Fail").
scores = [0.8, 0.4, 0.9, 0.3]
# If score >= 0.5 then 1 (Pass), else 0 (Fail)
binary_labels = [1 if s >= 0.5 else 0 for s in scores]
print(f"Binary Labels: {binary_labels}") # Binary Labels: [1, 0, 1, 0]

##### Five - Advanced: Slicing for Data Splitting

# 1. The Basic Split (Train/Test)
# The most common split is the 80/20 rule: 80% for training, 20% for testing.
# A dataset of 10 samples (could be image paths or text strings)
dataset = ["img1", "img2", "img3", "img4", "img5", "img6", "img7", "img8", "img9", "img10"]
# Syntax: list[start:stop] -> 'stop' is not included
train_data = dataset[:8]  # Indices 0 through 7
test_data = dataset[8:]   # Indices 8 through the end
print(f"Training set (80%): {train_data}") 
# Training set (80%): ['img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8']
print(f"Testing set (20%):  {test_data}") 
# Testing set (20%):  ['img9', 'img10']

# 2. The "Validation" Split (The 3-Way Split)
# Senior engineers use a validation set to tune the model while it's training, keeping the test set purely for the final grade.
# Splitting 70% Train, 20% Validation, 10% Test
train = dataset[:7]    # First 7 items
val   = dataset[7:9]   # Items at index 7 and 8
test  = dataset[9:]    # Item at index 9
print(f"Train: {train} | Val: {val} | Test: {test}") 
# Train: ['img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7'] | Val: ['img8', 'img9'] | Test: ['img10']

# 3. Advanced Slicing: The "Step" Function
# Sometimes you want to downsample a dataset (e.g., taking every 2nd frame of a video) to save processing time.
# Syntax: list[start:stop:step]
# Take every 2nd item from the entire dataset
downsampled_data = dataset[::2]
print(f"Every 2nd sample: {downsampled_data}") 
# Every 2nd sample: ['img1', 'img3', 'img5', 'img7', 'img9']
# Reverse the dataset (occasionally used for data augmentation)
reversed_data = dataset[::-1]
print(f"Reversed data: {reversed_data}") 
# Reversed data: ['img10', 'img9', 'img8', 'img7', 'img6', 'img5', 'img4', 'img3', 'img2', 'img1']

# 4. Handling 2D Data (Nested Lists)
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
# First two data rows: [[0.1, 0.2, 'Cat'], [0.5, 0.6, 'Dog']]
