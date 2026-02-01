########## THREE PYHON TUPLES FOR AI ##########

##### Tuples are immutable, ordered collections in Python. They're similar to lists but with one key difference: once created, their elements cannot be changed.

##### One - Tuple Properties

# 1. An ordered collection of items
# 2. Immutable (cannot be changed after creation)
# 3. Written using parentheses ()
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

##### Two - Tuples in AI Model Configuration
model_config = (
    "CNN",
    (224, 224, 3),   # input shape
    10,              # number of classes
    "relu"
)
model_type, input_shape, num_classes, activation = model_config # This is called tuple unpacking.

##### Three - Tuples as Dictionary Keys (Very Important in AI)

feature_map = {
    (0, 0): "background",
    (1, 0): "object",
    (0, 1): "edge"
}
print(feature_map[(1, 0)]) 
# PRINT RESULT: object

#### Four - Nested Tuples for Complex AI Structures
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

"""
Key Takeaways for AI:
Tuples are immutable → safer for AI configs
Ideal for shapes, coordinates, states, parameters
Commonly used with NumPy, TensorFlow, PyTorch
Improve performance and reliability
"""