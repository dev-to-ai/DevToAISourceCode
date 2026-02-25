########## Four. Python Dictionaries for AI ##########

# A dictionary is an unordered, mutable collection of key-value pairs. It's optimized for fast data retrieval by keys.

# 1. Dictionary Properties

# Creating a dictionary
student = {
    "name": "Alice",
    "age": 20,
    "courses": ["Math", "Physics"]
}

# 1.1 Mutable: You can add, remove, or change items after creation.
student["age"] = 21              # Update value
student["grade"] = "A"           # Add new key-value pair
del student["courses"]           # Remove a key-value pair
print("After mutation:", student)
# PRINT RESULT: After mutation: {'name': 'Alice', 'age': 21, 'grade': 'A'}

# 1.2 Unordered (as of Python 3.7+, insertion order is preserved).
# The order below is the same as insertion order
person = {}
person["first_name"] = "John"
person["last_name"] = "Doe"
person["age"] = 30
print("Insertion order preserved:", person)
# PRINT RESULT: Insertion order preserved: {'first_name': 'John', 'last_name': 'Doe', 'age': 30}

# 1.3 Keys must be immutable (e.g., strings, numbers, tuples), but values can be any type.
valid_dict = {
    "id": 101,                   # string key
    3.14: "pi",                  # float key
    (1, 2): "tuple key"          # tuple key
}
print("Valid keys:", valid_dict)
# PRINT RESULT: Valid keys: {'id': 101, 3.14: 'pi', (1, 2): 'tuple key'}
# invalid_dict = {[1, 2]: "list key"}  # lists are mutable. unhashable type: 'list'

# 1.4 Fast access: Average O(1) time complexity for lookups.
print("Name lookup:", student["name"]) # Fast key-based access
# PRINT RESULT: Name lookup: Alice
# Demonstrating dictionary access vs list access
numbers = list(range(1_000_000))
number_dict = {i: i for i in range(1_000_000)}
# Dictionary lookup (fast)
print(number_dict[999_999])
# PRINT RESULT: 999999
# List lookup (requires index)
print(numbers[999_999])
# PRINT RESULT: 999999

# 2. Essential Dictionary Methods for AI

# Our initial model setup - a Neural Network's hyperparameters and its performance metric
training_state = {
    "learning_rate": 0.01,
    "optimizer": "Adam",
    "loss_function": "CrossEntropy",
    "epochs_completed": 5
}

# 2.1 .keys() - Useful for checking which parameters are being tracked
print(f"Tracked Parameters: {list(training_state.keys())}")
# PRINT RESULT: Tracked Parameters: ['learning_rate', 'optimizer', 'loss_function', 'epochs_completed']

# 2.2 .values() - Useful for checking the settings without the labels
print(f"Current Settings: {list(training_state.values())}")
# PRINT RESULT: Current Settings: [0.01, 'Adam', 'CrossEntropy', 5]

# 2.3 .items() - Perfect for logging or printing status during training
print("--- Model Status Report ---")
for parameter, value in training_state.items():
    print(f"{parameter.replace('_', ' ').title()}: {value}")
"""PRINT RESULT:
--- Model Status Report ---
Learning Rate: 0.01
Optimizer: Adam
Loss Function: CrossEntropy
Epochs Completed: 5
"""

# 2.4 .update() - Merging new results or overriding settings
new_results = {
    "epochs_completed": 10,
    "current_accuracy": 0.94,
    "best_loss": 0.12
}
training_state.update(new_results)
print(f"\nUpdated Accuracy: {training_state['current_accuracy']}")
# PRINT RESULT: Updated Accuracy: 0.94

# 3. Word-to-Index Vocabulary
# Build vocabulary from a list of words
words = ["cat", "dog", "bird", "cat", "dog"]
vocab = {word: idx for idx, word in enumerate(set(words))}
print(vocab)
# PRINT RESULT: {'cat': 0, 'dog': 1, 'bird': 2}
""" EXPLANATION:
Using set() to Get Unique Words: {"cat", "dog", "bird"}
enumerate() for Indexing: enumerate(set(words))
This creates pairs of (index, word):
(0, 'cat')
(1, 'dog')
(2, 'bird')
Dictionary Comprehension: vocab = {word: idx for idx, word in enumerate(set(words))}
This builds a dictionary where:
Keys = words
Values = their corresponding indices
Result: {'cat': 0, 'dog': 1, 'bird': 2}
"""

# Use it to encode a sentence
sentence = ["dog", "cat", "bird"]
encoded = [vocab[word] for word in sentence]
print(encoded)  
# PRINT RESULT: [1, 0, 2]

# 4. Advanced Dictionary Techniques for AI

# 4.1 Nested Dictionaries - Common in AI Configurations
print("\n--- Nested Dictionaries: AI Model Configuration ---")
model_config = {
    "model_name": "Transformer",
    "architecture": {
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "activation": "gelu"
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01
        }
    },
    "data": {
        "train_path": "./data/train.csv",
        "val_path": "./data/val.csv",
        "test_path": "./data/test.csv",
        "num_workers": 4
    }
}

# Accessing nested values
print(f"Model name: {model_config['model_name']}")
# PRINT RESULT: Model name: Transformer
print(f"Number of layers: {model_config['architecture']['num_layers']}")
# PRINT RESULT: Number of layers: 12
print(f"Optimizer: {model_config['training']['optimizer']['name']}")
# PRINT RESULT: Optimizer: AdamW
print(f"Learning rate: {model_config['training']['learning_rate']}")
# PRINT RESULT: Learning rate: 0.001

# 4.2 DefaultDict - Perfect for Counting in NLP
"""
defaultdict is a special dictionary from Python's collections module that provides default values for missing keys. 
Unlike a regular dictionary, it never raises a KeyError - if you access a key that doesn't exist, it creates that key with a default value automatically.
"""
print("\n--- DefaultDict: Word Frequency Counting ---")
from collections import defaultdict

# Count word frequencies in a corpus
corpus = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "the bird flew over the cat"
]
word_counts = defaultdict(int)  # Default value for missing keys is 0

for sentence in corpus:              # Loop through each sentence
    for word in sentence.split():     # Split sentence into words
        word_counts[word] += 1        # Increment count for each word

print("Word frequencies:")
for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  '{word}': {count}")
# PRINT RESULT:
# Word frequencies:
#   'the': 6
#   'cat': 3
#   'on': 1
#   'mat': 1
#   'dog': 1
"""
word_counts.items() returns pairs like [('the', 6), ('cat', 3), ...]
key=lambda x: x[1] means "sort by the second element (the count)"
lambda x: x[1] is a function that takes a pair and returns the count
reverse=True means sort from highest to lowest
[:5] takes only the first 5 items (top 5 most frequent words)
"""
"""
# Original items
[('the', 6), ('cat', 3), ('sat', 1), ('on', 1), ('mat', 1), ('dog', 1), ...]

# After sorting by count (x[1]) in reverse order
[('the', 6), ('cat', 3), ('sat', 1), ('on', 1), ('mat', 1), ...]
# Then [:5] takes first 5: the, cat, sat, on, mat
"""

# 4.3 Counter - Built-in for Counting
"""
Counter is a specialized dictionary from Python's collections module designed specifically for counting hashable objects. 
It's like defaultdict(int) but with many additional features for counting tasks.
"""
print("\n--- Counter: Tracking Model Predictions ---")
from collections import Counter

# Track model predictions vs actual
predictions = ["cat", "dog", "cat", "bird", "cat", "dog", "bird", "cat"]
actual = ["cat", "dog", "bird", "bird", "cat", "cat", "bird", "dog"]

# Count correct predictions per class
correct = Counter() # Empty counter for correct predictions
# Counter counts the frequency of each item in 'actual'
total_per_class = Counter(actual)
# Result: Counter({'cat': 3, 'dog': 2, 'bird': 3})
# Because in actual: cat appears 3 times, dog 2 times, bird 3 times

for pred, act in zip(predictions, actual):
    if pred == act:
        correct[pred] += 1

# Calculate per-class accuracy
print("Per-class accuracy:")
for class_name in total_per_class:
    accuracy = correct[class_name] / total_per_class[class_name]
    print(f"  {class_name}: {correct[class_name]}/{total_per_class[class_name]} = {accuracy:.2%}")
""" PRINT RESULT:
--- Counter: Tracking Model Predictions ---
Per-class accuracy:
  cat: 2/3 = 66.67%
  dog: 1/2 = 50.00%
  bird: 2/3 = 66.67%"
"""

# 4.4 Dictionary Comprehension for Data Preprocessing
print("\n--- Dictionary Comprehension: Data Normalization ---")
data_samples = {
    "sample1": [100, 200, 300, 150],
    "sample2": [150, 250, 350, 200],
    "sample3": [200, 300, 400, 250],
    "sample4": [50, 100, 150, 75]
}

# Find global max for normalization
all_values = []
for values in data_samples.values():
    all_values.extend(values)
global_max = max(all_values)
# This collects ALL values from every sample into a single list, then finds the maximum value across all data (which is 400 in this example).

# Normalize all samples using dictionary comprehension
normalized_data = {
    key: [round(x/global_max, 3) for x in values] 
    for key, values in data_samples.items()
}
"""
This is the key part. It's a nested comprehension that:
Iterates through each key-value pair in data_samples.items()
For each sample, creates a new list where every value x is divided by global_max
Rounds the result to 3 decimal places
Creates a new dictionary with the same keys but normalized values
"""

print("Original max value:", global_max)
# PRINT RESULT: Original max value: 400
print("Normalized samples:")
for sample, values in normalized_data.items():
    print(f"  {sample}: {values}")
""" PRINT RESULT:
Normalized samples:
  sample1: [0.25, 0.5, 0.75, 0.375]
  sample2: [0.375, 0.625, 0.875, 0.5]
  sample3: [0.5, 0.75, 1.0, 0.625]
  sample4: [0.125, 0.25, 0.375, 0.188]
"""

# 4.5 setdefault() - Efficient Grouping
print("\n--- setdefault(): Grouping Data by Category ---")

# Group validation errors by type
validation_errors = [
    ("type_error", "Expected int, got str"),
    ("value_error", "Value out of range"),
    ("type_error", "Expected float, got bool"),
    ("key_error", "Missing required key"),
    ("value_error", "Negative value not allowed")
]

# Group without setdefault (verbose)
error_groups = {}
for error_type, message in validation_errors:
    if error_type not in error_groups:
        error_groups[error_type] = []
    error_groups[error_type].append(message)

# Group with setdefault (elegant)
error_groups_clean = {}
for error_type, message in validation_errors:
    error_groups_clean.setdefault(error_type, []).append(message)

print("Errors grouped by type:")
for error_type, messages in error_groups_clean.items():
    print(f"  {error_type}: {len(messages)} errors")
""" PRINT RESULT:
--- setdefault(): Grouping Data by Category ---
Errors grouped by type:
  type_error: 2 errors
  value_error: 2 errors
  key_error: 1 errors
"""    

# 5. Model Checkpoint Dictionary
print("\n--- Model Checkpoint: Saving Training State ---")
# PRINT RESULT: --- Model Checkpoint: Saving Training State ---
checkpoint = {
    "epoch": 50,
    "model_state_dict": {  # Simplified representation of model weights
        "layer1.weight": [[0.123, -0.456], [0.789, -0.321]],
        "layer1.bias": [0.111, -0.222],
        "layer2.weight": [[-0.333, 0.444], [0.555, -0.666]],
        "layer2.bias": [0.777, -0.888]
    },
    "optimizer_state_dict": {
        "param_groups": [{"lr": 0.001, "momentum": 0.9}],
        "state": {0: {"momentum_buffer": None}}
    },
    "loss": 0.0234,
    "accuracy": 0.9567,
    "hyperparameters": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "architecture": "CNN"
    }
}

print(f"Checkpoint - Epoch: {checkpoint['epoch']}")
# PRINT RESULT: Checkpoint - Epoch: 50
print(f"Checkpoint - Loss: {checkpoint['loss']:.4f}")
# PRINT RESULT: Checkpoint - Loss: 0.0234
print(f"Checkpoint - Accuracy: {checkpoint['accuracy']:.2%}")
# PRINT RESULT: Checkpoint - Accuracy: 95.67%
print(f"Checkpoint - Model layers: {len(checkpoint['model_state_dict'])} tensors")
# PRINT RESULT: Checkpoint - Model layers: 4 tensors

# 6. AI Dictionary Example

# 6.1 Creating AI Dictionary (Concept → Explanation)
print("\n--- AI Dictionary: Concepts and Explanations ---")
# PRINT RESULT: --- AI Dictionary: Concepts and Explanations ---
ai_dictionary = {
    "Artificial Intelligence": "The simulation of human intelligence by machines to perform tasks like reasoning, learning, and decision-making.",
    "Machine Learning": "A subset of AI where systems learn patterns from data instead of being explicitly programmed.",
    "Deep Learning": "A subset of machine learning that uses neural networks with many layers to learn complex patterns.",
    "Neural Network": "A model inspired by the human brain, consisting of layers of interconnected nodes (neurons).",
    "Supervised Learning": "A learning approach using labeled data where the correct output is known.",
    "Unsupervised Learning": "A learning approach using unlabeled data to discover hidden patterns.",
    "Reinforcement Learning": "A learning method where an agent learns by interacting with an environment and receiving rewards or penalties.",
    "Training Data": "Data used to teach a machine learning model.",
    "Test Data": "Data used to evaluate the performance of a trained model.",
    "Feature": "An individual measurable property or characteristic of the data.",
    "Label": "The correct output or answer associated with input data.",
    "Model": "A mathematical representation that learns from data to make predictions.",
    "Algorithm": "A step-by-step procedure used by a model to learn from data.",
    "Overfitting": "When a model learns the training data too well and performs poorly on new data.",
    "Underfitting": "When a model is too simple to capture underlying patterns in data.",
    "Accuracy": "A metric that measures how often a model makes correct predictions."
}

# 6.2 Accessing AI Definitions
term = "Machine Learning"
print(f"\nDefinition of '{term}':")
print(ai_dictionary[term])
""" PRINT RESULT: 
Definition of 'Machine Learning':
A subset of AI where systems learn patterns from data instead of being explicitly programmed.
"""

# 6.3 Safe Lookup (Avoid Errors)
term = "Computer Vision"
definition = ai_dictionary.get(term, "Term not found in AI dictionary.")
print(f"\nLooking up '{term}': {definition}") 
# PRINT RESULT: Looking up 'Computer Vision': Term not found in AI dictionary.

# 6.4 Loop Through the AI Dictionary (showing first 5 only for brevity)
print("\n--- First 5 AI Terms and Definitions ---")
for i, (term, explanation) in enumerate(ai_dictionary.items()):
    if i < 5:
        print(f"\n{term}:")
        print(f"  {explanation[:100]}...")  # Show first 100 chars
""" PRINT RESULT:
--- First 5 AI Terms and Definitions ---

Artificial Intelligence:
  The simulation of human intelligence by machines to perform tasks like reasoning, learning, and deci...

Machine Learning:
  A subset of AI where systems learn patterns from data instead of being explicitly programmed....

Deep Learning:
  A subset of machine learning that uses neural networks with many layers to learn complex patterns....

Neural Network:
  A model inspired by the human brain, consisting of layers of interconnected nodes (neurons)....

Supervised Learning:
  A learning approach using labeled data where the correct output is known....
"""

# 6.5 Simple AI Dictionary Function
def explain_ai_term(term):
    """Return explanation for an AI term."""
    return ai_dictionary.get(term, f"Sorry, '{term}' is not in the AI dictionary.")

print(f"\n{explain_ai_term('Deep Learning')}")
# PRINT RESULT: A subset of machine learning that uses neural networks with many layers to learn complex patterns.

print(explain_ai_term('Random Forest'))
# PRINT RESULT: Sorry, 'Random Forest' is not in the AI dictionary.

# 6.6 Real-World Use Case (Mini AI Tutor)
print("\n--- Mini AI Tutor ---")
print("(This would be interactive - here's a demonstration)")
""" PRINT RESULT: 
--- Mini AI Tutor ---
(This would be interactive - here's a demonstration)
"""

def mini_ai_tutor():
    """Interactive AI tutor function."""
    print("Welcome to the AI Tutor! Ask about any AI term (type 'exit' to quit).")
    # PRINT RESULT: Welcome to the AI Tutor! Ask about any AI term (type 'exit' to quit).
    
    # Demo mode - simulate user input
    demo_queries = ["Training Data", "Overfitting", "Transformer", "exit"]
    
    for query in demo_queries:
        print(f"\nUser: {query}")
        if query.lower() == "exit":
            print("AI Tutor: Goodbye! Keep learning about AI!")
            break
        
        response = explain_ai_term(query)
        print(f"AI Tutor: {response}")

mini_ai_tutor()
""" PRINT RESULT:
Welcome to the AI Tutor! Ask about any AI term (type 'exit' to quit).

User: Training Data
AI Tutor: Data used to teach a machine learning model.

User: Overfitting
AI Tutor: When a model learns the training data too well and performs poorly on new data.

User: Transformer
AI Tutor: Sorry, 'Transformer' is not in the AI dictionary.

User: exit
AI Tutor: Goodbye! Keep learning about AI!
"""

# 7. Performance Comparison: Dictionary vs List Lookup
print("\n--- Performance Comparison ---")
import time

# Create test data
size = 1_000_000
test_dict = {i: f"value_{i}" for i in range(size)}
test_list = [f"value_{i}" for i in range(size)]

# Dictionary lookup
start = time.time()
for i in range(1000):
    _ = test_dict[size // 2]  # Look up middle element
dict_time = time.time() - start

# List lookup (by index)
start = time.time()
for i in range(1000):
    _ = test_list[size // 2]  # Access by index
list_time = time.time() - start

print(f"Dictionary lookup (1000x): {dict_time:.6f} seconds")
print(f"List index access (1000x): {list_time:.6f} seconds")
print(f"Dictionary is {list_time/dict_time:.1f}x faster for random access!")
""" PRINT RESULT:
--- Performance Comparison ---
Dictionary lookup (1000x): 0.000155 seconds
List index access (1000x): 0.000128 seconds
Dictionary is 0.8x faster for random access!
"""

# 8. Real-World: Transformer Model Configuration
print("\n--- Real-World: GPT-style Model Config ---")

gpt_config = {
    "model_type": "causal_lm",
    "vocab_size": 50257,
    "max_position_embeddings": 1024,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "use_cache": True,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}

# Calculate model parameters
hidden_size = gpt_config["hidden_size"]
num_layers = gpt_config["num_hidden_layers"]
intermediate_size = gpt_config["intermediate_size"]
vocab_size = gpt_config["vocab_size"]

# Approximate parameter count
embedding_params = vocab_size * hidden_size
attention_params = num_layers * (4 * hidden_size * hidden_size)  # Q,K,V,O
ffn_params = num_layers * (2 * hidden_size * intermediate_size)  # Up and down projections
total_params = embedding_params + attention_params + ffn_params

print(f"Model: {gpt_config['model_type']}")
print(f"Layers: {gpt_config['num_hidden_layers']}")
print(f"Hidden size: {gpt_config['hidden_size']}")
print(f"Attention heads: {gpt_config['num_attention_heads']}")
print(f"Approximate parameters: {total_params/1e6:.1f}M")
""" PRINT RESULT:
--- Real-World: GPT-style Model Config ---
Model: causal_lm
Layers: 12
Hidden size: 768
Attention heads: 12
Approximate parameters: 123.5M
"""

# 9. Dictionary Views - Dynamic and Efficient
print("\n--- Dictionary Views: Live Windows into Data ---")

training_log = {
    "epoch": 5,
    "loss": 0.234,
    "accuracy": 0.912,
    "val_loss": 0.289,
    "val_accuracy": 0.887
}

# Views are dynamic - they update when the dictionary changes
keys_view = training_log.keys()
values_view = training_log.values()
items_view = training_log.items()

print(f"Initial keys: {list(keys_view)}")

# Add a new metric
training_log["learning_rate"] = 0.001
print(f"Updated keys: {list(keys_view)}")  # Automatically includes new key

# Views support set operations
mandatory_metrics = {"loss", "accuracy", "epoch"}
current_metrics = set(training_log.keys())
missing_metrics = mandatory_metrics - current_metrics
print(f"Missing metrics: {missing_metrics}")
""" PRINT RESULT:
--- Dictionary Views: Live Windows into Data ---
Initial keys: ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
Updated keys: ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'learning_rate']
Missing metrics: set()
"""

# 10. Memory Efficiency: Dict vs List for Lookup Tables
print("\n--- Memory Usage: Dictionary vs List for Lookup ---")

import sys

# Create mapping of 10,000 IDs to values
n_items = 10000
id_to_value_dict = {i: f"user_{i}" for i in range(n_items)}

# List approach (parallel arrays)
ids_list = list(range(n_items))
values_list = [f"user_{i}" for i in range(n_items)]

dict_size = sys.getsizeof(id_to_value_dict) + sum(sys.getsizeof(k) + sys.getsizeof(v) 
                                                   for k, v in id_to_value_dict.items())
list_size = sys.getsizeof(ids_list) + sys.getsizeof(values_list) + \
            sum(sys.getsizeof(i) for i in ids_list) + \
            sum(sys.getsizeof(v) for v in values_list)

print(f"Dictionary total memory: {dict_size/1024:.1f} KB")
print(f"List pair total memory: {list_size/1024:.1f} KB")
print(f"Dictionary uses {dict_size/list_size*100:.1f}% of list memory")
""" PRINT RESULT:
--- Memory Usage: Dictionary vs List for Lookup ---
Dictionary total memory: 1048.7 KB
List pair total memory: 922.0 KB
Dictionary uses 113.7% of list memory
"""

# 11. Summary: 
# When to Use Dictionaries in AI
print("\n--- Summary: Dictionary Use Cases in AI ---")
summary = {
    "Hyperparameters": "Store and organize model configuration",
    "Word Embeddings": "Map words to vectors (word2vec, GloVe)",
    "Feature Maps": "Store extracted features from neural networks",
    "Model Checkpoints": "Save model state during training",
    "Class Labels": "Map class indices to human-readable names",
    "Evaluation Metrics": "Track accuracy, loss, F1-score during training",
    "Data Batches": "Organize input features and labels",
    "Configuration": "Store experiment settings and parameters"
}

for use_case, description in summary.items():
    print(f"  • {use_case}: {description}")

print("""
Key Takeaways:
1. Dictionaries provide O(1) lookup time - crucial for large AI datasets
2. Use .get() for safe access to avoid KeyError exceptions
3. Nested dictionaries are perfect for hierarchical configurations
4. defaultdict and Counter simplify counting operations in NLP
5. Dictionary comprehensions enable elegant data transformations
6. Dictionaries are ideal for storing model states and checkpoints
""")