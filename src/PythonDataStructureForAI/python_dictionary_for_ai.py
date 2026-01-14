##### A dictionary is an unordered, mutable collection of key-value pairs. It's optimized for fast data retrieval by keys.

##### One - Dictionary Properties

# Creating a dictionary
student = {
    "name": "Alice",
    "age": 20,
    "courses": ["Math", "Physics"]
}

# 1. Mutable: You can add, remove, or change items after creation.
student["age"] = 21              # Update value
student["grade"] = "A"           # Add new key-value pair
del student["courses"]           # Remove a key-value pair
print("After mutation:", student)
# After mutation: {'name': 'Alice', 'age': 21, 'grade': 'A'}

# 2. Unordered (as of Python 3.7+, insertion order is preserved).
# The order below is the same as insertion order
person = {}
person["first_name"] = "John"
person["last_name"] = "Doe"
person["age"] = 30
print("Insertion order preserved:", person)
# Insertion order preserved: {'first_name': 'John', 'last_name': 'Doe', 'age': 30}

# 3. Keys must be immutable (e.g., strings, numbers, tuples), but values can be any type.
valid_dict = {
    "id": 101,                   # string key
    3.14: "pi",                  # float key
    (1, 2): "tuple key"          # tuple key
}
print("Valid keys:", valid_dict)
# Valid keys: {'id': 101, 3.14: 'pi', (1, 2): 'tuple key'}
# invalid_dict = {[1, 2]: "list key"}  # lists are mutable. unhashable type: 'list'

# 4. Fast access: Average O(1) time complexity for lookups.
print("Name lookup:", student["name"])   # Fast key-based access
# Name lookup: Alice
# Demonstrating dictionary access vs list access
numbers = list(range(1_000_000))
number_dict = {i: i for i in range(1_000_000)}
# Dictionary lookup (fast)
print(number_dict[999_999])
# List lookup (requires index)
print(numbers[999_999])

##### Two - Essential Dictionary Methods for AI

# Our initial model setup - a Neural Network's hyperparameters and its performance metric
training_state = {
    "learning_rate": 0.01,
    "optimizer": "Adam",
    "loss_function": "CrossEntropy",
    "epochs_completed": 5
}

# 1. .keys() - Useful for checking which parameters are being tracked
print(f"Tracked Parameters: {list(training_state.keys())}")
# Tracked Parameters: ['learning_rate', 'optimizer', 'loss_function', 'epochs_completed']

# 2. .values() - Useful for checking the settings without the labels
print(f"Current Settings: {list(training_state.values())}")
# Current Settings: [0.01, 'Adam', 'CrossEntropy', 5]

# 3. .items() - Perfect for logging or printing status during training
print("--- Model Status Report ---")
for parameter, value in training_state.items():
    print(f"{parameter.replace('_', ' ').title()}: {value}")
"""
--- Model Status Report ---
Learning Rate: 0.01
Optimizer: Adam
Loss Function: CrossEntropy
Epochs Completed: 5
"""

# 4. .update() - Merging new results or overriding settings
new_results = {
    "epochs_completed": 10,
    "current_accuracy": 0.94,
    "best_loss": 0.12
}
training_state.update(new_results)
print(f"\nUpdated Accuracy: {training_state['current_accuracy']}")
# Updated Accuracy: 0.94

##### Three - Word-to-Index Vocabulary
# Build vocabulary from a list of words
words = ["cat", "dog", "bird", "cat", "dog"]
vocab = {word: idx for idx, word in enumerate(set(words))}
print(vocab) # {'dog': 0, 'bird': 1, 'cat': 2}

# Use it to encode a sentence
sentence = ["dog", "cat", "bird"]
encoded = [vocab[word] for word in sentence]
print(encoded)  # [2, 1, 0]

##### Four - AI Dictionary Example

# 1. Creating AI Dictionary (Concept → Explanation)
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

# 2. Accessing AI Definitions
term = "Machine Learning"
print(ai_dictionary[term])
# A subset of AI where systems learn patterns from data instead of being explicitly programmed.

# 3. Safe Lookup (Avoid Errors)
term = "Computer Vision"
definition = ai_dictionary.get(term, "Term not found in AI dictionary.")
print(definition) # Term not found in AI dictionary.

#4. Loop Through the AI Dictionary
for term, explanation in ai_dictionary.items():
    print(f"{term}: {explanation}")

# 5. Simple AI Dictionary Function
def explain_ai_term(term):
    return ai_dictionary.get(term, "Sorry, this AI term is not in the dictionary.")
print(explain_ai_term("Deep Learning"))

# 6. Real-World Use Case (Mini AI Tutor)
while True:
    user_input = input("Enter an AI term (or 'exit'): ")
    if user_input.lower() == "exit":
        break
    print(explain_ai_term(user_input))

