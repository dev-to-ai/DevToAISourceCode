########## Two. Python Sets for AI ##########

# 1. Set Properties

# In Python, a set is an unordered collection of unique, immutable elements. 
# Sets are mutable (you can add or remove items), but the elements themselves must be hashable (i.e., immutable types like int, str, tuple, etc.).

"""
Once created, immutable objects cannot be modified. Any operation that appears to modify them actually creates a new object.
Common immutable types: int, float, complex, str, tuple, frozenset, bytes, bool, None

Mutable objects can be modified after creation without changing their identity.
Common mutable types: list, dict, set, bytearray, Custom class instances
"""

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

# 1.3 Mutable: You can add or remove elements after creation. Elements must be immutable.
s = {1, 2}
s.add(3)
s.remove(1)
s.update([100,101])
print(s)  
# PRINT RESULT: {2, 3, 100, 101}

# 1.4 Elements Must Be Hashable. Only immutable types can be in a set.
"""
An object is hashable if it has a hash value that never changes during its lifetime, and can be compared to other objects. 
Hashable objects implement __hash__() and __eq__() methods.
Immutable objects are typically hashable (with some exceptions)
Hashable objects can be used as:
- Dictionary keys
- Set elements
- Members of other hashable collections
"""
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
"""
This is a set comprehension with a nested loop. Let's read it from right to left:
{word for review in reviews for word in review.split()}
#  │    │                    │
#  │    │                    └─ Inner loop: splits each review into words
#  │    └──────────────────────── Outer loop: iterates through each review
#  └───────────────────────────────── What we collect: each word
"""
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

"""
Time Complexities:
┌─────────────────┬───────────────┬──────────────────────────────┐
│ Operation       │ List          │ Set                          │
├─────────────────┼───────────────┼──────────────────────────────┤
│ Membership      │ O(n)          │ O(1) avg                     │
│ Insertion       │ O(1) amort    │ O(1) avg                     │
│ Deletion        │ O(n)          │ O(1) avg                     │
│ Union           │ N/A           │ O(len(s))                    │
│ Intersection    │ N/A           │ O(min(len(s1), len(s2)))     │
└─────────────────┴───────────────┴──────────────────────────────┘
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

# 4.5 Real-World AI Use Cases: Document Deduplication in RAG Systems
# Imports several type hints from Python's built-in typing module.
# These are used to provide static type checking for your code, helping to catch errors before runtime
from typing import Set, List, Dict, Any, FrozenSet, Optional

class DocumentDeduplicator:
    """Remove near-duplicate documents using set operations"""
    # Class docstring explaining the purpose - uses set operations to find and remove near-duplicate documents
    
    def __init__(self, similarity_threshold: float = 0.8):
        # Constructor method with type hint - similarity_threshold must be float, defaults to 0.8 (80% similarity)
        
        self.similarity_threshold = similarity_threshold
        # Store the threshold value as instance variable - documents with Jaccard similarity above this are considered duplicates
        
        self.unique_docs: Set[FrozenSet[str]] = set()
        # Type hint: Set containing FrozenSet of strings
        # Initialize empty set to store unique document shingles as frozensets
        # Using frozenset because:
        #   1. Sets must contain hashable elements (frozenset is hashable, set is not)
        #   2. We need to compare documents using set operations
        #   3. Documents shouldn't be modified after being stored
        
        self.documents: List[str] = []
        # Type hint: List of strings
        # Initialize empty list to store the actual text of unique documents in order added
    
    def _shingle_document(self, doc: str, k: int = 3) -> frozenset[str]:
        # Private method (indicated by _) - converts document to shingles
        # Takes a string document and optional k (shingle size, default 3)
        # Returns a frozenset of strings
        
        """Convert document to set of k-shingles"""
        # Method docstring explaining what shingling does
        
        words = doc.split()
        # Split the document string into a list of words
        # Example: "AI is transforming" -> ["AI", "is", "transforming"]
        
        shingles = {' '.join(words[i:i+k]) for i in range(len(words)-k+1)}
        # Set comprehension that creates shingles (contiguous word sequences)
        # 
        # Let's break it down:
        # range(len(words)-k+1) - generates starting positions for shingles
        #   If words = ["AI", "is", "transforming", "the", "world"] (5 words)
        #   k = 3, then len(words)-k+1 = 5-3+1 = 3 starting positions: 0, 1, 2
        #
        # For each i:
        #   words[i:i+k] - slices k consecutive words starting at i
        #   i=0: words[0:3] = ["AI", "is", "transforming"]
        #   i=1: words[1:4] = ["is", "transforming", "the"]
        #   i=2: words[2:5] = ["transforming", "the", "world"]
        #
        # ' '.join() - joins the words with spaces
        #   Creates: "AI is transforming", "is transforming the", "transforming the world"
        #
        # { ... for i in range(...) } - set comprehension collects all shingles
        #   Result: {"AI is transforming", "is transforming the", "transforming the world"}
        
        return frozenset(shingles)
        # Convert the set of shingles to an immutable frozenset and return
        # frozenset is used because:
        #   1. It's hashable (can be stored in self.unique_docs set)
        #   2. The shingles shouldn't be modified after creation
        #   3. We still get all set operation benefits (intersection, union, etc.)
    
    def add_document(self, doc: str) -> bool:
        # Public method to add a document to the deduplicator
        # Takes a string document, returns boolean (True if added, False if duplicate)
        
        """Returns True if document was added (not duplicate)"""
        # Method docstring
        
        shingles = self._shingle_document(doc)
        # Convert the document to shingles using the private helper method
        # This gives us a set representation of the document's content
        
        # Check similarity with existing documents
        for existing in self.unique_docs:
            # Iterate through each previously stored unique document (as frozenset of shingles)
            
            jaccard = len(shingles & existing) / len(shingles | existing)
            # Calculate Jaccard similarity between new document and existing document
            # 
            # shingles & existing - INTERSECTION: shingles common to both documents
            #   Example: If doc has {"a b", "b c"} and existing has {"b c", "c d"}
            #   Intersection = {"b c"} -> length 1
            #
            # shingles | existing - UNION: all unique shingles from both documents
            #   Union = {"a b", "b c", "c d"} -> length 3
            #
            # Jaccard = 1/3 ≈ 0.33 (33% similar)
            #
            # Jaccard similarity ranges from 0 (completely different) to 1 (identical)
            
            if jaccard > self.similarity_threshold:
                # If similarity exceeds threshold (default 0.8 or 80%)
                
                return False  # Duplicate detected
                # Exit method early, return False to indicate document was NOT added
                # This is a near-duplicate document
        
        self.unique_docs.add(shingles)
        # If we get here, no duplicate was found
        # Add the new document's shingles (as frozenset) to the unique_docs set
        
        self.documents.append(doc)
        # Append the original document text to the documents list
        # This maintains the order in which unique documents were added
        
        return True
        # Return True to indicate document was successfully added as unique
    
    def get_unique_documents(self) -> list[str]:
        # Getter method to retrieve all unique documents
        # Returns a list of strings
        
        return self.documents
        # Return the list of unique document texts in the order they were added

# Usage - demonstrating how to use the class
deduplicator = DocumentDeduplicator()
# Create an instance of DocumentDeduplicator with default threshold (0.8)

docs = [
    "AI is transforming the world of technology",
    "AI is transforming the world of tech",  # Near duplicate
    "Machine learning is a subset of AI"
]
# List of documents to process
# Note: First two documents are very similar (near duplicates)

for doc in docs:
    # Iterate through each document in the list
    
    if deduplicator.add_document(doc):
        # Call add_document method for each document
        # Returns True if document is unique and was added
        
        print(f"Added: {doc[:30]}...")
        # Print first 30 chars of document with "Added:" prefix
        # doc[:30] slices the string to show only first 30 characters
        
    else:
        # If add_document returns False (duplicate detected)
        
        print(f"Duplicate rejected: {doc[:30]}...")
        # Print first 30 chars with "Duplicate rejected:" prefix

""" PRINT RESULT:
Added: AI is transforming the world o...
Added: AI is transforming the world o...
Added: Machine learning is a subset o...
"""
# Note: The output shows ALL documents as "Added" because:
# 1. First document added (unique)
# 2. Second document is NEAR duplicate but not exact - similarity likely < 0.8 threshold
#    Let's calculate approximate similarity:
#    Doc1 shingles (k=3): 
#      - "AI is transforming"
#      - "is transforming the"
#      - "transforming the world"
#      - "the world of"
#      - "world of technology"
#    Doc2 shingles:
#      - "AI is transforming"
#      - "is transforming the"
#      - "transforming the world"
#      - "the world of"
#      - "world of tech"
#    Only last shingle differs: "world of technology" vs "world of tech"
#    Intersection size = 4, Union size = 6, Jaccard = 4/6 ≈ 0.667 < 0.8
#    So second document is considered unique, not duplicate!
# 3. Third document is completely different, definitely added

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
# This code finds connected components in an undirected graph using a DFS/BFS hybrid approach.
# Represent a graph as adjacency sets
graph = {
    1: {2, 3},      # Node 1 connected to nodes 2 and 3
    2: {1, 4},      # Node 2 connected to nodes 1 and 4
    3: {1, 5},      # Node 3 connected to nodes 1 and 5
    4: {2},         # Node 4 connected only to node 2
    5: {3},         # Node 5 connected only to node 3
    6: {7},         # Node 6 connected to node 7 (separate component)
    7: {6}          # Node 7 connected to node 6 (separate component)
}
"""
This creates two separate components:
Component A: {1, 2, 3, 4, 5} - all connected through paths
Component B: {6, 7} - isolated from the first
"""
"""
Visual Representation
text
Component 1:           Component 2:
    1                 6
   / \                |
  2   3               |
  |   |               7
  4   5

Edges:
1-2, 1-3, 2-4, 3-5   6-7
"""

def find_connected_components(graph):
    visited = set() # Tracks all nodes we've seen
    components = [] # Stores each connected component
    
    for node in graph: # Check every node in the graph
        if node not in visited: # Found start of new component
            # Start exploring this component
            component = set() # Will hold all nodes in this component
            queue = {node} # Initialize queue with starting node
            
            while queue: # Explore until we find all connected nodes
                current = queue.pop()  # Take a node from queue
                if current not in visited:
                    visited.add(current)  # Mark as visited
                    component.add(current) # Add to current component
                    # Add all unvisited neighbors to queue
                    queue.update(graph.get(current, set()) - visited)
            
            components.append(component) # Save the completed component
    
    return components

components = find_connected_components(graph)
print(f"Connected components: {components}")
# PRINT RESULT: Connected components: [{1, 2, 3, 4, 5}, {6, 7}]

# 6. Error Handling with Sets
class SafeSetOperations:
    # Define a class that contains static methods for safely handling set operations
    # The class serves as a namespace/container for related utility functions
    
    @staticmethod
    # Decorator that defines a static method (doesn't receive self or cls parameter)
    # Can be called directly on the class without creating an instance
    
    def safe_intersection(set1: Optional[Set], set2: Optional[Set]) -> Set:
        # Method that safely computes intersection between two sets
        # Type hints:
        #   - set1: Optional[Set] - can be either a Set or None
        #   - set2: Optional[Set] - can be either a Set or None  
        #   - Returns: Set (always returns a set, never None)
        
        """Safe intersection that handles None values"""
        # Docstring explaining the purpose
        
        if set1 is None or set2 is None:
            # Check if either input is None
            # This prevents AttributeError when trying to use & operator on None
            
            return set()
            # Return empty set instead of None or raising error
            # This ensures the return type is always a set (consistent)
        
        return set1 & set2
        # If both inputs are actual sets, perform normal intersection
        # & operator is equivalent to set1.intersection(set2)
    
    @staticmethod
    def safe_difference(set1: Optional[Set], set2: Optional[Set]) -> Set:
        # Method for safely computing set difference (elements in set1 not in set2)
        
        """Safe difference that handles None values"""
        # Docstring
        
        if set1 is None:
            # Check if the first set (the one we're subtracting from) is None
            
            return set()
            # If set1 is None, there are no elements to begin with
            # Return empty set as the result
        
        if set2 is None:
            # Check if the second set (the one we're subtracting) is None
            
            return set1.copy()
            # If set2 is None, there's nothing to remove from set1
            # Return a COPY of set1 (not the original) to maintain immutability
            # .copy() creates a shallow copy of the set
            
        return set1 - set2
        # If both are valid sets, perform normal difference operation
        # - operator is equivalent to set1.difference(set2)
    
    @staticmethod
    def safe_add_to_set(base_set: Optional[Set], element: Any) -> Set:
        # Method for safely adding an element to a set
        # Type hints:
        #   - base_set: Optional[Set] - can be Set or None
        #   - element: Any - any Python object
        #   - Returns: Set - the modified set
        
        """Safely add element to set, creating set if None"""
        # Docstring
        
        if base_set is None:
            # Check if the input set is None
            
            base_set = set()
            # Create a new empty set if None was provided
            # This allows the method to always have a set to work with
        
        if not isinstance(element, (int, float, str, tuple, frozenset)):
            # Check if element is NOT an instance of allowed immutable types
            # isinstance() checks if element belongs to any of the types in the tuple
            # The allowed types are all immutable/hashable types:
            #   - int: integer numbers
            #   - float: floating point numbers  
            #   - str: strings
            #   - tuple: immutable sequences
            #   - frozenset: immutable sets
            
            print(f"Warning: {type(element)} may not be hashable")
            # Print warning if element might be unhashable
            # This is just a warning - operation will still proceed
            # If element is actually unhashable, .add() will raise TypeError
        
        base_set.add(element)
        # Add the element to the set
        # If element is unhashable, this will raise TypeError
        
        return base_set
        # Return the modified set

# Usage - demonstrating the safe operations
safe_ops = SafeSetOperations()
# Create instance of SafeSetOperations (though static methods don't need instance)
# Could also call directly as SafeSetOperations.safe_intersection()

result = safe_ops.safe_intersection({1, 2}, None)
# Call safe_intersection with:
#   - set1 = {1, 2} (valid set)
#   - set2 = None (invalid input)
# 
# Inside the method:
#   1. Checks if set1 is None? → No (it's {1, 2})
#   2. Checks if set2 is None? → Yes (it's None)
#   3. Returns set() (empty set) without attempting intersection

print(f"Safe intersection: {result}")
# Print the result
# Since set2 was None, result is empty set: set()

# PRINT RESULT: Safe intersection: set()

# 7. Performance Optimizations with Sets

# 7.1 Batch Operations
import random
import time

# 7.1 Batch Operations
def batch_operations_example():
    """Demonstrate performance of batch vs individual operations"""
    data = range(10_000)
    
    # Inefficient: individual adds
    start = time.time()
    s1 = set()
    for item in data:
        s1.add(item)
    individual_time = time.time() - start
    
    # Efficient: batch update
    start = time.time()
    s2 = set()
    s2.update(data)  # Single operation
    batch_time = time.time() - start
    
    print(f"Individual adds: {individual_time:.4f}s")
    print(f"Batch update: {batch_time:.4f}s")
    print(f"Batch is {individual_time/batch_time:.1f}x faster")

# 7.2 Set vs List for Unique Counting
def count_unique_efficiently():
    """Different ways to count unique elements"""
    data = [random.randint(1, 1000) for _ in range(100_000)]
    
    # Method 1: Using set (most efficient)
    start = time.time()
    unique_count = len(set(data))
    set_time = time.time() - start
    
    # Method 2: Manual counting with dict
    start = time.time()
    count_dict = {}
    for item in data:
        count_dict[item] = count_dict.get(item, 0) + 1
    dict_count = len(count_dict)
    dict_time = time.time() - start
    
    print(f"Set method: {set_time:.4f}s, found {unique_count} unique")
    print(f"Dict method: {dict_time:.4f}s, found {dict_count} unique")

# Call both methods
if __name__ == "__main__":
    print("=" * 50)
    print("7.1 Batch Operations Performance")
    print("=" * 50)
    batch_operations_example()
    """ PRINT RESULT:
    ==================================================
    7.1 Batch Operations Performance
    ==================================================
    Individual adds: 0.0006s
    Batch update: 0.0004s
    Batch is 1.4x faster
    """
    
    print("\n" + "=" * 50)
    print("7.2 Set vs List for Unique Counting")
    print("=" * 50)
    count_unique_efficiently()
    """ PRINT RESULT:
    ==================================================
    7.2 Set vs List for Unique Counting
    ==================================================
    Set method: 0.0015s, found 1000 unique
    Dict method: 0.0081s, found 1000 unique
"""

# 8. Common Pitfalls and How to Avoid Them

# 8.1 Modifying Set During Iteration
# BAD
# s = {1, 2, 3, 4}
# for item in s:  # RuntimeError: Set changed size during iteration
#     if item % 2 == 0:
#         s.remove(item)

# GOOD
s = {1, 2, 3, 4}
s = {item for item in s if item % 2 != 0}  # Set comprehension
print(f"After filtering: {s}")
# PRINT RESULT: After filtering: {1, 3}

# 8.2 Assuming Order
# BAD
s = {3, 1, 4}
first_element = next(iter(s))  # Don't rely on this being consistent!

# GOOD
if 3 in s:  # Always check membership, not position
    print("3 is in the set")
# PRINT RESULT: 3 is in the set

# 8.3 Using Mutable Elements
# BAD
# s = set()
# s.add([1, 2])  # TypeError: unhashable type: 'list'

# GOOD
s = set()
s.add((1, 2))  # Tuple is immutable and hashable
s.add(frozenset([3, 4]))  # Frozenset is immutable
print(f"Set with immutable elements: {s}")
# PRINT RESULT: Set with immutable elements: {frozenset({3, 4}), (1, 2)}

# 9. Advanced Set Operations in AI Pipelines

# 9.1 Feature Interaction Detection
class FeatureInteractionDetector:
    """Detect potential feature interactions using set operations"""
    
    def __init__(self):
        self.feature_sets: Dict[str, Set[str]] = {}
    
    def add_feature_importance(self, model_name: str, important_features: List[str]):
        """Store important features for a model"""
        self.feature_sets[model_name] = set(important_features)
    
    def find_interactions(self, threshold: float = 0.5) -> List[Set[str]]:
        """Find features that frequently appear together"""
        all_features = set.union(*self.feature_sets.values())
        interactions = []
        
        for feature1 in all_features:
            for feature2 in all_features:
                if feature1 >= feature2:  # Avoid duplicates
                    continue
                
                # Count models where both features are important
                count = sum(1 for fs in self.feature_sets.values() 
                           if feature1 in fs and feature2 in fs)
                
                if count / len(self.feature_sets) >= threshold:
                    interactions.append({feature1, feature2})
        
        return interactions

# 9.2 Ensemble Method Voting
class EnsembleVoting:
    """Use sets for ensemble prediction voting"""
    
    def __init__(self):
        self.model_predictions: Dict[str, Set[int]] = {}
    
    def add_predictions(self, model_name: str, predictions: List[int]):
        """Store predictions for a model"""
        self.model_predictions[model_name] = set(predictions)
    
    def unanimous_votes(self) -> Set[int]:
        """Find items all models agree on"""
        return set.intersection(*self.model_predictions.values())
    
    def majority_votes(self, threshold: float = 0.5) -> Set[int]:
        """Find items majority of models agree on"""
        all_predictions = set.union(*self.model_predictions.values())
        majority = set()
        
        for item in all_predictions:
            votes = sum(1 for preds in self.model_predictions.values() 
                       if item in preds)
            if votes / len(self.model_predictions) >= threshold:
                majority.add(item)
        
        return majority

from typing import Dict, Set, List
# Define the classes here or import them
# (Assuming the classes are defined above)

def quick_test():
    """Quick test of both classes"""
    
    # Test FeatureInteractionDetector
    print("\nTesting FeatureInteractionDetector...")
    detector = FeatureInteractionDetector()
    
    detector.add_feature_importance("model1", ["a", "b", "c"])
    detector.add_feature_importance("model2", ["b", "c", "d"])
    detector.add_feature_importance("model3", ["a", "c", "d"])
    
    interactions = detector.find_interactions(threshold=0.66)
    print(f"Interactions (threshold=0.66): {interactions}")
    """
    Testing FeatureInteractionDetector...
    Interactions (threshold=0.66): [{'c', 'b'}, {'c', 'a'}, {'c', 'd'}]
    """
    
    # Test EnsembleVoting
    print("\nTesting EnsembleVoting...")
    voting = EnsembleVoting()
    
    voting.add_predictions("model1", [1, 2, 3, 4])
    voting.add_predictions("model2", [2, 3, 4, 5])
    voting.add_predictions("model3", [3, 4, 5, 6])
    
    unanimous = voting.unanimous_votes()
    majority = voting.majority_votes(threshold=0.66)
    
    print(f"Unanimous votes: {unanimous}")
    print(f"Majority votes (≥66%): {majority}")
    """
    Testing EnsembleVoting...
    Unanimous votes: {3, 4}
    Majority votes (≥66%): {2, 3, 4, 5}
    """

# Run quick test
if __name__ == "__main__":
    quick_test()



# 88888. Summary:
"""
Common Use Cases in AI:
1. Vocabulary building: set(text.split())
2. Feature selection: set(features) - set(noisy_features)
3. Deduplication: set(data)
4. Jaccard similarity: len(a & b) / len(a | b)
5. Outlier detection: set(observed) - set(expected)
6. Data splitting: train | val | test
7. Cache keys: frozenset(feature_names)
8. Graph algorithms: visited nodes, components

Key Takeaways:
1. Speed vs Memory: Sets are faster for lookups but use more memory
2. Hashability: Only immutable objects can be set elements
3. Set operations (union, intersection, difference) are powerful for data analysis
4. Frozensets enable using sets as dictionary keys for caching
5. Set comprehensions provide concise, readable code for feature extraction
"""