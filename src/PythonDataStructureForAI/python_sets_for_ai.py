##### One - Set Properties

# In Python, a set is an unordered collection of unique, immutable elements. 
# Sets are mutable (you can add or remove items), but the elements themselves must be hashable (i.e., immutable types like int, str, tuple, etc.).

# 1. Unordered: Sets do not maintain any specific order. You cannot access elements by index (e.g., my_set[0] raises an error).
s = {3, 1, 4}
print(s)  
# Output might be {1, 3, 4} — order not guaranteed

# 2. Unique Elements: Duplicates are automatically removed.
# Cleaning datasets
s = {2, 2, 3, 4, 2, 1, 3, 4, 1}
print(s) 
# Output might be {1, 2, 3, 4}
# Create from a list
data = [1, 2, 2, 3, 4, 4, 5]
unique_data = set(data)
print(unique_data) 
# {1, 2, 3, 4, 5}

# 3. Mutable: You can add or remove elements after creation.
s = {1, 2}
s.add(3)
s.remove(1)
s.update([100,101])
print(s)  
# {2, 3, 100, 101}

# 4. Elements Must Be Hashable. Only immutable types can be in a set.
# Immutable objects cannot be changed after creation. Any operation that appears to modify them actually creates a new object.
# Immutable Types: int, float, complex, str, bytes, tuple, frozenset, bool, None
# Mutable Types: list, dict, set, bytearray
s = {1, "hello", (2, 3)}
# s = {[1, 2], {3, 4}}  # TypeError: unhashable type: 'list' or 'set'

# 5. No Indexing or Slicing
# Since sets are unordered, indexing (s[0]) and slicing (s[1:3]) are not supported.

# 6. Empty Set
empty_set = set()
print(empty_set) 
# set()
# {} creates a dictionary, not a set.

# 7. Frozenset: Immutable Version
# If you need an immutable set, use frozenset.
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError: 'frozenset' object has no attribute 'add'

##### Two - Mathematical Set Operations in AI
# Sets support mathematical operations like: 
# Union (| or union()) 
# Intersection (& or intersection()) 
# Difference (- or difference()) 
# Symmetric difference (^ or symmetric_difference())
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)   
# {1, 2, 3, 4, 5}
print(a & b)   
# {3}
print(a - b)   
# {1, 2}
print(a ^ b)   
# {1, 2, 4, 5}

# Sets for Similarity (Jaccard Similarity)
# Used in recommendation systems & document similarity.
doc1 = {"ai", "machine", "learning"}
doc2 = {"ai", "deep", "learning"}
intersection = doc1 & doc2
print(intersection) 
# {'learning', 'ai'}
union = doc1 | doc2
print(union) 
# {'deep', 'learning', 'machine', 'ai'}
jaccard_similarity = len(intersection) / len(union)
print(jaccard_similarity) 
# 0.5

##### Three - Performance Advantage: Sets vs. Lists
# In AI, you often work with millions of data points. Checking if a specific value exists in a collection is a frequent operation.
# Lists: Use O(n) time for membership tests. To find an item, Python must scan the list from start to finish.
# Sets: Use O(1) time (on average). Because sets use a hash table under the hood, looking up an item is nearly instantaneous, regardless of how large the set is.
# Note: If you are building a recommendation engine and need to check if a user has already seen 10,000 different items, using a set for those items will be thousands of times faster than using a list.
import time
# Let's create a collection of 10 million items
data_size = 10_000_000
large_list = list(range(data_size))
large_set = set(range(data_size))
# We will search for an item at the very end of the collection - Worst-case scenario for a List
target = 9_999_999
# --- Testing the List ---
start_time = time.time()
is_in_list = target in large_list
list_duration = time.time() - start_time
print(f"List Search: {list_duration:.6f} seconds")
# --- Testing the Set ---
start_time = time.time()
is_in_set = target in large_set
set_duration = time.time() - start_time
print(f"Set Search:  {set_duration:.6f} seconds")
# Calculate the speed factor
factor = list_duration / set_duration
print(f"\nResult: Sets are approximately {factor:,.0f}x faster in this case.")
# Result: Sets are approximately 21,814x faster in this case.
"""
Why the difference is so massive?
The reason for this disparity lies in the underlying data structure.
Lists (Linear Search): To find the target, Python starts at index 0 and checks every single box until it finds a match. If your target is at the end of 10 million items, it does 10 million checks.
Sets (Hash Table): A set uses a hashing function to calculate the exact "address" of the data. Instead of searching, it jumps directly to the location where the data should be.

When to use which in AI?
Use Lists: When the order of your data matters (e.g., time-series data or sequences of words) or if you need to allow duplicate values.
Use Sets: When you need to perform lookups, find unique categories, or compare groups of features, and the order is irrelevant.
"""

##### Four - Sets in NLP (Natural Language Processing)

# 1. Vocabulary Creation
# Example 1:
sentences = [
    "machine learning is powerful",
    "learning ai is fun"
]
vocab = set()
for sentence in sentences:
    vocab.update(sentence.split())
print(vocab) 
# {'powerful', 'ai', 'is', 'learning', 'fun', 'machine'}

# Example 2:
text_data = "AI is the future and the future is AI"
# Tokenize and convert to set to find unique words
vocabulary = set(text_data.lower().split())
print(vocabulary) 
# {'ai', 'future', 'and', 'is', 'the'}

# 2. Stop Word Removal
stop_words = {"is", "the", "and", "a"}
tokens = ["ai", "is", "the", "future"]
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens) 
# ['ai', 'future']

##### - Why Sets Matter in AI & Data Science

# Sets are extremely useful in AI for:
# Removing duplicate data
# Vocabulary building (NLP)
# Feature selection
# Similarity comparison
# Fast membership testing
# Graph & knowledge representation