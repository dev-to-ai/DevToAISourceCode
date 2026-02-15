########## Two. Python Sets for AI ##########

# 1. Set Properties

# In Python, a set is an unordered collection of unique, immutable elements. 
# Sets are mutable (you can add or remove items), but the elements themselves must be hashable (i.e., immutable types like int, str, tuple, etc.).

# 1.1 Unordered: Sets do not maintain any specific order. You cannot access elements by index (e.g., my_set[0] raises an error).
s = {3, 1, 4}
print(s)
# PRINT RESULT: {1, 3, 4}
# Output might be {1, 3, 4} — order not guaranteed

# 1.2 Unique Elements: Duplicates are automatically removed.
# Cleaning datasets
s = {2, 2, 3, 4, 2, 1, 3, 4, 1}
print(s)
# PRINT RESULT: {1, 2, 3, 4}
# Output might be {1, 2, 3, 4} — order not guaranteed
# Create from a list
data = [1, 2, 2, 3, 4, 4, 5]
unique_data = set(data)
print(unique_data) 
# PRINT RESULT: {1, 2, 3, 4, 5}

# 1.3 Mutable: You can add or remove elements after creation.
s = {1, 2}
s.add(3)
s.remove(1)
s.update([100,101])
print(s)  
# PRINT RESULT: {2, 3, 100, 101}

# 1.4 Elements Must Be Hashable. Only immutable types can be in a set.
# Immutable objects cannot be changed after creation. Any operation that appears to modify them actually creates a new object.
# Immutable Types: int, float, complex, str, bytes, tuple, frozenset, bool, None
# Mutable Types: list, dict, set, bytearray
s = {1, "hello", (2, 3)}
# s = {[1, 2], {3, 4}}  # TypeError: unhashable type: 'list' or 'set'

# 1.5 No Indexing or Slicing
# Since sets are unordered, indexing (s[0]) and slicing (s[1:3]) are not supported.

# 1.6 Empty Set
empty_set = set()
print(empty_set) 
# PRINT RESULT: set()
# {} creates a dictionary, not a set.

# 1.7 Frozenset: Immutable Version
# If you need an immutable set, use frozenset.
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError: 'frozenset' object has no attribute 'add'

# Using frozenset as dictionary keys (regular sets can't be keys)
# Useful for caching or memoization in AI algorithms
cache = {}
feature_set = frozenset(["age", "income", "education"])
cache[feature_set] = "Trained model A"
print(cache[feature_set])
# PRINT RESULT: Trained model A

# In feature engineering, this helps avoid redundant computations
feature_combinations = {
    frozenset(["pixel_intensity", "texture"]): "CNN features",
    frozenset(["word_count", "sentiment"]): "NLP features"
}

# 2. Mathematical Set Operations in AI
# Sets support mathematical operations like: 
# Union (| or union()) 
# Intersection (& or intersection()) 
# Difference (- or difference()) 
# Symmetric difference (^ or symmetric_difference())
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)   
# PRINT RESULT: {1, 2, 3, 4, 5}
print(a & b)   
# PRINT RESULT: {3}
print(a - b)   
# PRINT RESULT: {1, 2}
print(a ^ b)   
# PRINT RESULT: {1, 2, 4, 5}

# 2.1 Sets for Similarity (Jaccard Similarity)
# Used in recommendation systems & document similarity.
doc1 = {"ai", "machine", "learning"}
doc2 = {"ai", "deep", "learning"}
intersection = doc1 & doc2
print(intersection) 
# PRINT RESULT: {'learning', 'ai'}
union = doc1 | doc2
print(union) 
# PRINT RESULT: {'deep', 'learning', 'machine', 'ai'}
jaccard_similarity = len(intersection) / len(union)
print(jaccard_similarity) 
# PRINT RESULT: 0.5

# 2.2 Set Comprehensions
# Set comprehensions are concise and Pythonic
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = {x**2 for x in numbers if x % 2 == 0}
print(even_squares)  
# PRINT RESULT: {64, 4, 36, 16, 100}

# Extracting features from text data
reviews = ["good product", "bad service", "great experience"]
unique_words = {word for review in reviews for word in review.split()}
print(unique_words)  
# PRINT RESULT: {'great', 'bad', 'good', 'service', 'product', 'experience'}

# Feature extraction for ML preprocessing
raw_data = ["cat", "dog", "cat", "bird", "dog", "fish", "cat"]
unique_features = {animal.upper() for animal in raw_data if len(animal) > 3}
print(unique_features)
# PRINT RESULT: {'BIRD', 'FISH'}

# 3. Performance Advantage: Sets vs. Lists
# 3.1 Time Considerations
# In AI, you often work with millions of data points. Checking if a specific value exists in a collection is a frequent operation.
# Lists: Use O(n) time for membership tests. To find an item, Python must scan the list from start to finish.
# Sets: Use O(1) time (on average). Because sets use a hash table under the hood, looking up an item is nearly instantaneous, regardless of how large the set is.
# Note: If you are building a recommendation engine and need to check if a user has already seen 10,000 different items, using a set for those items will be thousands of times faster than using a list.
import time
# Let's create a collection of 10 million items
data_size = 10_000_000
large_list = list(range(data_size))
large_set = set(range(data_size))
# We will search for an item at the very end of the collection - worst-case scenario for a list
target = 9_999_999
# --- Testing the List ---
start_time = time.time()
is_in_list = target in large_list
list_duration = time.time() - start_time
print(f"List Search: {list_duration:.6f} seconds")
# PRINT RESULT: List Search: 0.059848 seconds
# --- Testing the Set ---
start_time = time.time()
is_in_set = target in large_set
set_duration = time.time() - start_time
print(f"Set Search:  {set_duration:.6f} seconds")
# PRINT RESULT: Set Search:  0.000003 seconds
# Calculate the speed factor
factor = list_duration / set_duration
print(f"\nResult: Sets are approximately {factor:,.0f}x faster in this case.")
# PRINT RESULT: Result: Sets are approximately 22,820x faster in this case.
"""
Why the difference is so massive?
The reason for this disparity lies in the underlying data structure.
Lists (Linear Search): To find the target, Python starts at index 0 and checks every single box until it finds a match. If your target is at the end of 10 million items, it does 10 million checks.
Sets (Hash Table): A set uses a hashing function to calculate the exact "address" of the data. Instead of searching, it jumps directly to the location where the data should be.

When to use which in AI?
Use Lists: When the order of your data matters (e.g., time-series data or sequences of words) or if you need to allow duplicate values.
Use Sets: When you need to perform lookups, find unique categories, or compare groups of features, and the order is irrelevant.
"""

# 3.2 Memory Considerations
# Sets have higher memory overhead than lists
# Trade-off: Speed vs Memory
import sys

data = list(range(10000))
list_size = sys.getsizeof(data)
set_size = sys.getsizeof(set(data))

print(f"List memory: {list_size:,} bytes")
print(f"Set memory: {set_size:,} bytes")
print(f"Set overhead: {((set_size/list_size)-1)*100:.1f}%")
# PRINT RESULT: List memory: 87,760 bytes
# PRINT RESULT: Set memory: 524,456 bytes  
# PRINT RESULT: Set overhead: 497.6%
# Sets typically use 4-5x more memory, but the speed benefit often justifies the cost

# 4. Sets in NLP (Natural Language Processing)

# 4.1 Vocabulary Creation
# Example 1:
sentences = [
    "machine learning is powerful",
    "learning ai is fun"
]
vocab = set()
for sentence in sentences:
    vocab.update(sentence.split())
print(vocab) 
# PRINT RESULT: {'powerful', 'ai', 'is', 'learning', 'fun', 'machine'}

# Example 2:
text_data = "AI is the future and the future is AI"
# Tokenize and convert to set to find unique words
vocabulary = set(text_data.lower().split())
print(vocabulary) 
# PRINT RESULT: {'ai', 'future', 'and', 'is', 'the'}

# 4.2 Stop Word Removal
stop_words = {"is", "the", "and", "a"}
tokens = ["ai", "is", "the", "future"]
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens) 
# PRINT RESULT: ['ai', 'future']

# 4.3 Anomaly Detection with Sets
# Using sets to find anomalies in data streams
expected_categories = {"normal", "warning", "error"}
observed_data = ["normal", "warning", "normal", "critical", "error", "unknown"]

anomalies = set(observed_data) - expected_categories
print(f"Anomalous categories detected: {anomalies}")
# PRINT RESULT: Anomalous categories detected: {'critical', 'unknown'}

# Real-world example: Detecting unusual system states in logs
system_logs = ["INFO", "ERROR", "WARNING", "INFO", "DEBUG", "ERROR", "CRITICAL"]
valid_states = {"INFO", "WARNING", "ERROR"}
invalid_states = set(system_logs) - valid_states
if invalid_states:
    print(f"Alert: Invalid log states detected: {invalid_states}")
    # Implement alerting logic here
# PRINT RESULT: Alert: Invalid log states detected: {'DEBUG', 'CRITICAL'}

# 4.4 Set Operations for Data Splitting
# Common in train/validation/test splits for machine learning
import random

# Simulate a dataset of 1000 items
all_items = set(range(1000))

# Randomly sample 700 for training (70%)
train_set = set(random.sample(list(all_items), 700))

# From remaining items, sample 150 for validation (15%)
remaining = all_items - train_set
val_set = set(random.sample(list(remaining), 150))

# Test set is everything else (15%)
test_set = all_items - train_set - val_set

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
# PRINT RESULT: Train: 700, Val: 150, Test: 150

# Ensure no overlap between sets (sets guarantee this automatically)
assert not (train_set & val_set)  # Intersection should be empty
assert not (train_set & test_set)
assert not (val_set & test_set)
print("✓ All splits are mutually exclusive")
# PRINT RESULT: ✓ All splits are mutually exclusive

# 5. Advanced AI Applications with Sets

# 5.1 Feature Selection - Finding Unique Feature Combinations
feature_groups = [
    {"age", "income", "education"},
    {"age", "income", "location"},
    {"age", "education", "occupation"},
    {"income", "education", "occupation"}
]

# Find features that appear in all groups (core features)
core_features = set.intersection(*[set(group) for group in feature_groups])
# 1. [set(group) for group in feature_groups] - Convert each list to set (already sets)
# 2. * operator unpacks the list of sets
# 3. set.intersection() finds common elements across ALL sets
print(f"Core features appearing in all groups: {core_features}")
# PRINT RESULT: Core features appearing in all groups: set()  # No feature appears in all

# Find features that appear in at least one group
all_possible_features = set.union(*[set(group) for group in feature_groups])
print(f"All possible features: {all_possible_features}")
# PRINT RESULT: All possible features: {'location', 'age', 'education', 'occupation', 'income'}

# 5.2 Graph Algorithms - Finding Connected Components
# Represent a graph as adjacency sets
graph = {
    1: {2, 3},
    2: {1, 4},
    3: {1, 5},
    4: {2},
    5: {3},
    6: {7},  # Separate component
    7: {6}   # Separate component
}

def find_connected_components(graph):
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            # BFS/DFS to find all connected nodes
            component = set()
            queue = {node}
            while queue:
                current = queue.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    queue.update(graph.get(current, set()) - visited)
            components.append(component)
    
    return components

components = find_connected_components(graph)
print(f"Connected components: {components}")
# PRINT RESULT: Connected components: [{1, 2, 3, 4, 5}, {6, 7}]

# 6. Summary:
"""
Why Sets Matter in AI & Data Science?
Sets are extremely useful in AI for:
✓ Removing duplicate data
✓ Vocabulary building (NLP)
✓ Feature selection
✓ Similarity comparison
✓ Fast membership testing
✓ Graph & knowledge representation
✓ Anomaly detection
✓ Data splitting for ML pipelines
✓ Memory-efficient caching (with frozenset)

Key Takeaways:
1. Speed vs Memory: Sets are faster for lookups but use more memory
2. Hashability: Only immutable objects can be set elements
3. Set operations (union, intersection, difference) are powerful for data analysis
4. Frozensets enable using sets as dictionary keys for caching
5. Set comprehensions provide concise, readable code for feature extraction
"""