########## Two. NumPy Arrays Properties (ndarray) ##########

import numpy as np

# 1. A NumPy array (ndarray) is:
# A same data type container
# Stored in contiguous memory
# Designed for multi-dimensional math

# 2. Create a 2D NumPy array (matrix)
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr)
""" PRINT RESULT:
[[1 2 3]
 [4 5 6]]
"""
print(arr *2)
""" PRINT RESULT:
[[ 2  4  6]
 [ 8 10 12]]
"""

# 3. Core array attributes (ndarray)

# .ndim — Number of Dimensions
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a.ndim)  
# PRINT RESULT: 1
print(b.ndim)  
# PRINT RESULT: 2
"""
ndim	Meaning
1	    Vector
2	    Matrix
3+	    Tensors (DL)
"""

# .shape — Size of Each Dimension
print(a.shape)  
# PRINT RESULT: (3,)
print(b.shape)  
# PRINT RESULT: (2, 2)

# .size — Total Number of Elements
print(a.size)  
# PRINT RESULT: 3
print(b.size)  
# PRINT RESULT: 4

# .dtype — Data Type
c = np.array([1, 2, 3])
d = np.array([1.0, 2.0, 3.0])
print(c.dtype)  
# PRINT RESULT: int64
print(d.dtype)  
# PRINT RESULT: float64

arr = np.array([1, 2, 3], dtype=np.float32)
print(arr)
# PRINT RESULT: [1. 2. 3.]
# Deep learning prefers float32 because of memory efficiency
# Common dtypes: int32, float32, float64

# 4. Memory layout properties (crucial for optimization)
arr = np.array([[1, 2, 3], [4, 5, 6]])

# .itemsize - bytes per element
print(f"Itemsize: {arr.itemsize} bytes for {arr.dtype}")  
# PRINT RESULT: Itemsize: 8 bytes for int64

# .nbytes - total bytes consumed
print(f"Total memory: {arr.nbytes} bytes for {arr.size} elements x {arr.itemsize} bytes")  
# PRINT RESULT: Total memory: 48 bytes for 6 elements x 8 bytes

# .strides - bytes to step in each dimension
print(f"Strides: {arr.strides}")  
# PRINT RESULT: Strides: (24, 8)  
# rows: 24 bytes, columns: 8 bytes
# To move to next row, skip 24 bytes (3 cols × 8 bytes)
# To move to next column, skip 8 bytes

# 5. Data type impact on memory and precision
# Memory comparison for AI models
float64_arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
float32_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)

print(f"float64 memory: {float64_arr.nbytes} bytes")  
# PRINT RESULT: float64 memory: 24 bytes
print(f"float32 memory: {float32_arr.nbytes} bytes")
# PRINT RESULT: float32 memory: 12 bytes
# float32 uses half the memory - critical for large datasets

# Precision demonstration
very_small = np.array([1e-10], dtype=np.float32)
print(f"float32 precision: {very_small[0]}")
# PRINT RESULT: float32 precision: 1.000000013351432e-10
very_small = np.array([1e-10], dtype=np.float64)
print(f"float64 precision: {very_small[0]}")
# PRINT RESULT: float64 precision: 1e-10

# 6. Complex numbers (used in audio processing, Fourier transforms)
complex_arr = np.array([1+2j, 3+4j, 5+6j])
# 1+2j, 3+4j, 5+6j → complex numbers
# j → imaginary unit (√−1) in Python
print(f"Complex array: {complex_arr}")
# PRINT RESULT: Complex array: [1.+2.j 3.+4.j 5.+6.j]
print(f"Real parts: {complex_arr.real}")
# PRINT RESULT: Real parts: [1. 3. 5.]
print(f"Imaginary parts: {complex_arr.imag}")
# PRINT RESULT: Imaginary parts: [2. 4. 6.]
print(f"Data type: {complex_arr.dtype}")  
# PRINT RESULT: Data type: complex128

# 7. Array flags and special properties
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Check if array is C-contiguous (row-major) - default in NumPy
# C-contiguous means elements are stored in row-major order (last index changes fastest)
print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
# PRINT RESULT: C-contiguous: True

# Check if array is F-contiguous (column-major)
# F-contiguous means elements are stored in column-major order
print(f"F-contiguous: {arr.flags['F_CONTIGUOUS']}")  # False
# PRINT RESULT: F-contiguous: False

# Base array (if view)
# This checks if the array is a view (shares data with another array)
view = arr[:2, :2]
print(f"Base array exists: {view.base is not None}")
# PRINT RESULT: Base array exists: True

# 8. Special array types
# Structured array (like a spreadsheet row - mixed data types)
structured = np.array([('Alice', 25, 165.5),
                       ('Bob', 30, 180.2)],
                      dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])
print(f"Structured array:\n{structured}")
""" PRINT RESULT:
Structured array:
[('Alice', 25, 165.5) ('Bob', 30, 180.2)]
"""
print(f"All names: {structured['name']}")
# PRINT RESULT: All names: ['Alice' 'Bob']
# Record array (similar but with attribute access)
# Useful for mixed data types in data science

# 9. NumPy array vs Python list comparison
py_list = [1, 2, 3]
np_arr = np.array([1, 2, 3])

# Lists can mix types (but less efficient)
mixed_list = [1, "two", 3.0, [4, 5]]

# NumPy arrays must have same type (more efficient)
try:
    mixed_arr = np.array([1, "two", 3.0])  # Will convert all to string
    print(f"Mixed array becomes: {mixed_arr}, dtype: {mixed_arr.dtype}")
except:
    pass
# PRINT RESULT: Mixed array becomes: ['1' 'two' '3.0'], dtype: <U32

# 10. Quick Reference Card
print("\n" + "=" * 60)
print("📌 QUICK REFERENCE CARD")
print("=" * 60)

reference = """
Attribute    Description                    Example Usage
─────────────────────────────────────────────────────────────────
.ndim        Number of dimensions           arr.ndim → 2
.shape       Size per dimension             arr.shape → (3, 4)
.size        Total elements                 arr.size → 12
.dtype       Data type                      arr.dtype → int64
.itemsize    Bytes per element               arr.itemsize → 8
.nbytes      Total memory usage              arr.nbytes → 96
.strides     Bytes to step in each dim       arr.strides → (32, 8)

Pro Tips:
• Use .shape for debugging array dimensions
• Monitor .nbytes for large datasets
• Choose float32 over float64 when possible in deep learning
• Check .dtype before operations to avoid unexpected results
"""

print(reference)
