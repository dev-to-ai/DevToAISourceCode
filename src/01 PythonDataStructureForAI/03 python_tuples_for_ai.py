########## Three. Python Tuples for AI ##########

# Tuples are immutable, ordered collections in Python. 
# They're similar to lists but with one key difference: once created, their elements cannot be changed.

# 1. Tuple Properties
"""
1.1 An ordered collection of items
1.2 Immutable (cannot be changed after creation)
1.3 Written using parentheses ()
"""

# Creating a tuple
point = (3, 4)          # A 2D coordinate

# Tuple Immutability. Important for AI Safety
# point[0] = 100 # 'tuple' object does not support item assignment

rgb_color = (255, 128, 0)
print(rgb_color[2]) 
# PRINT RESULT: 0

batch_size = (32,) # Single-element tuple (comma is required)
print(batch_size) 
# PRINT RESULT: (32,)

shape = 224, 224, 3 # Tuple without parentheses
print(shape) 
# PRINT RESULT: (224, 224, 3)

height = shape[0]
width = shape[1]
channels = shape[2]
print(shape[0]) 
# PRINT RESULT: 224
print(height, width, channels) 
# PRINT RESULT: 224 224 3
print(shape[-1]) # -1: Last one
# PRINT RESULT: 3

# 2. Tuple Unpacking
model_config = (
    "CNN",
    (224, 224, 3),   # input shape
    10,              # number of classes
    "relu"
)
model_type, input_shape, num_classes, activation = model_config # This is called tuple unpacking
print(activation)
# PRINT RESULT: relu

# 3. Tuples as Dictionary Keys (Very Important in AI)
feature_map = {
    (0, 0): "background",
    (1, 0): "object",
    (0, 1): "edge"
}
print(feature_map[(1, 0)]) 
# PRINT RESULT: object

# 4. Nested Tuples for Complex AI Structures
layer_structure = (
    ("Conv2D", (3, 3), 32),
    ("MaxPool", (2, 2)),
    ("Dense", 128)
)
for layer in layer_structure:
    print(layer)
""" PRINT RESULT: 
('Conv2D', (3, 3), 32)
('MaxPool', (2, 2))
('Dense', 128)
"""

# 5. Memory Efficiency with Tuples
import sys
ai_config_list = ["CNN", 224, 224, 3, "relu", 0.001]
ai_config_tuple = ("CNN", 224, 224, 3, "relu", 0.001)
print(f"List memory: {sys.getsizeof(ai_config_list)} bytes")
# PRINT RESULT: List memory: 104 bytes
print(f"Tuple memory: {sys.getsizeof(ai_config_tuple)} bytes")
# PRINT RESULT: Tuple memory: 88 bytes
# Tuple uses less memory!

# 6. Hashable Property for Caching
model_signature = hash(("ResNet50", 224, 224, "imagenet_weights"))
print(f"Model cache key: {model_signature}")
# PRINT RESULT: Model cache key: -8432749111323674376

# 7. Advanced AI Model Configuration

# 7.1 Model hyperparameters as tuples
hyperparams = (
    0.001,           # learning rate
    32,              # batch size
    50,              # epochs
    (0.9, 0.999),    # Adam optimizer betas
    "adam"           # optimizer
)

# 7.2 Nested configurations for model ensembles
model_ensemble = (
    ("ResNet50", (224, 224, 3), 1000),
    ("ViT", (224, 224, 3), 1000),
    ("EfficientNet", (224, 224, 3), 1000)
)

# 8. Tuples as Keys for Feature Maps

# 8.1 Feature extraction caching
feature_cache = {}
image_id = "img_001"
feature_type = "edges"
feature_cache[(image_id, feature_type)] = "extracted_features"

# 8.2 Multi-dimensional feature mapping
attention_weights = {
    (0, 1): 0.8,    # Attention from token 0 to token 1
    (0, 2): 0.1,
    (1, 0): 0.3
}

# 9. Advanced Nested Structures

# 9.1 Neural network architecture as tuples
transformer_block = (
    ("MultiHeadAttention", 8, 64),  # 8 heads, 64 dim each
    ("LayerNorm", 1e-6),
    ("FeedForward", 2048, 512),
    ("LayerNorm", 1e-6)
)

# 9.2 Training pipeline stages
pipeline_stages = (
    ("data_loading", "batch", 32),
    ("augmentation", ("random_flip", "random_rotate")),
    ("normalization", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
    ("forward_pass", "inference")
)

# 10. Tuple Methods for AI
sample_data = (1, 2, 3, 2, 4, 2, 5)
print(f"Count of class 2: {sample_data.count(2)}")  
# PRINT RESULT: Count of class 2: 3
print(f"First occurrence of 3 at index: {sample_data.index(3)}")  
# PRINT RESULT: First occurrence of 3 at index: 2

# Convert list to tuple for immutability
train_labels = [0, 1, 0, 1, 2, 1, 0]
safe_labels = tuple(train_labels)  
# Now immutable for distributed training

# 11. Named Tuples for Clear AI Code
"""
Named tuples are extensions of regular tuples that allow both index-based access (like tuples) and attribute-based access (like objects).
They combine the best of both worlds: immutability of tuples and readability of objects.
"""

from collections import namedtuple

# 11.1 Define AI-specific named tuples
ModelConfig = namedtuple('ModelConfig', ['name', 'input_shape', 'num_classes', 'learning_rate'])
"""
First argument: Class name ('ModelConfig')
Second argument: Field names (as list or space-separated string)
Returns: A new named tuple class
"""
Dataset = namedtuple('Dataset', ['train', 'val', 'test', 'class_names'])
Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1_score'])

# 11.2 Use named tuples in practice
config = ModelConfig('EfficientNet', (224, 224, 3), 1000, 0.001)
print(f"Model: {config.name}, Input: {config.input_shape}")
# PRINT RESULT: Model: EfficientNet, Input: (224, 224, 3)

dataset_sizes = Dataset(50000, 5000, 10000, ('cat', 'dog', 'bird'))
print(f"Training samples: {dataset_sizes.train}")
# PRINT RESULT: Training samples: 50000

results = Metrics(0.95, 0.94, 0.93, 0.935)
print(f"F1 Score: {results.f1_score}")
# PRINT RESULT: F1 Score: 0.935

# 12. Tuple Comparison in Model Selection
model_v1 = ("ResNet50", 224, 0.001)
model_v2 = ("ResNet50", 224, 0.0001)

print(f"v1 == v2: {model_v1 == model_v2}")  # False (different LR)
print(f"v1 < v2: {model_v1 < model_v2}")    # False (compares element-wise)
# PRINT RESULT: v1 == v2: False
# PRINT RESULT: v1 < v2: False

# 13. Real-World Dataset Example: Working with Image Dataset Metadata
image_dataset = (
    ("train_001.jpg", (224, 224, 3), (0.5, 0.2, 0.1)),  # filename, shape, label (one-hot)
    ("train_002.jpg", (224, 224, 3), (0.0, 0.7, 0.3)),
    ("train_003.jpg", (224, 224, 3), (0.1, 0.1, 0.8))
)

# Batch processing
for filename, shape, label in image_dataset[:2]:  # First 2 samples
    print(f"Processing {filename}: shape={shape}, dominant class={max(range(len(label)), key=lambda i: label[i])}")
""" PRINT RESULT:
Processing train_001.jpg: shape=(224, 224, 3), dominant class=0
Processing train_002.jpg: shape=(224, 224, 3), dominant class=1
"""

# 14. Speed Comparison: Tuples vs Lists
import timeit

# Setup code
setup = """
tuple_data = tuple(range(1000))
list_data = list(range(1000))
"""

# Test access speed
tuple_time = timeit.timeit("tuple_data[500]", setup=setup, number=1000000)
list_time = timeit.timeit("list_data[500]", setup=setup, number=1000000)

print(f"Tuple access: {tuple_time:.4f}s")
print(f"List access: {list_time:.4f}s")
print(f"Tuple is {(list_time/tuple_time - 1)*100:.1f}% faster")
""" PRINT RESULT:
Tuple access: 0.0175s
List access: 0.0167s
Tuple is -4.6% faster
"""

# 15. Safe Configuration Management
def safe_model_init(config):
    if not isinstance(config, tuple):
        raise TypeError(f"Config must be a tuple, got {type(config)}")
    if len(config) != 4:
        raise ValueError(f"Config must have 4 elements, has {len(config)}")
    
    model_name, input_shape, classes, lr = config
    print(f"Initialized {model_name} safely")
    return config

# Test
try:
    safe_model_init(["CNN", 224, 10, 0.001])  # Wrong type
except TypeError as e:
    print(f"Error caught: {e}")
# PRINT RESULT: Error caught: Config must be a tuple, got <class 'list'>

# 16. Summary:

"""
Quick Reference: Tuples vs Lists in AI
┌─────────────────┬─────────────────────────────┬───────────────────────────┐
│ Use Case        │ Tuple                       │ List                      │
├─────────────────┼─────────────────────────────┼───────────────────────────┤
│ Model configs   │ Perfect (immutable)         │ Risky (can change)        │
│ Batch data      │ No (needs modification)     │ Yes (dynamic)             │
│ Dict keys       │ Yes (hashable)              │ No (unhashable)           │
│ Memory          │ 16% less memory             │ More overhead             │
│ Speed           │ Faster access               │ Slightly slower           │
└─────────────────┴─────────────────────────────┴───────────────────────────┘
"""

"""
Key Takeaways for AI:

1. Immutability Benefits:
   1.1 Thread safety for parallel training
   1.2 Protection against accidental modifications
   1.3 Reliable configuration management

2. Performance Advantages:
   2.1 Memory efficient storage
   2.2 Faster access than lists
   2.3 Hashable for caching and dict keys

3. AI/ML Applications:
   3.1 Model architecture definitions
   3.2 Input/output shapes
   3.3 Hyperparameter configurations
   3.4 Feature mapping and attention mechanisms
   3.5 Dataset metadata storage

4. Best Practices:
   4.1 Use tuples for fixed configurations
   4.2 Leverage named tuples for self-documenting code
   4.3 Employ nested tuples for complex structures
   4.4 Convert lists to tuples for distributed systems
"""