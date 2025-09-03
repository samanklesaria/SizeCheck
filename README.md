# ShapeCheck: Runtime Shape Validation for Size-Annotated PyTorch Code

A Python decorator that automatically adds runtime shape checking to PyTorch functions based on size-annotated variable names using AST transformation.

## Overview

When writing PyTorch code, it's common to use naming conventions that indicate tensor shapes. For example, if a tensor `weights` has shape `n × k`, you might name the variable `weights_nk`. This library automatically validates that tensors match their annotated shapes at runtime by analyzing and modifying your function's Abstract Syntax Tree (AST).

## Key Features

- **AST-based transformation**: Automatically injects shape checks into function arguments and variable assignments
- **Intuitive naming convention**: Use underscores to indicate tensor shapes
- **Comprehensive checking**: Validates both function parameters and intermediate assignments
- **Clear error messages**: Provides detailed information when shapes don't match

## Quick Start

```python
import torch
from shapecheck import shapecheck

@shapecheck
def matrix_multiply(A_NK, B_KM):
    """Matrix multiplication with automatic shape checking."""
    result_NM = torch.matmul(A_NK, B_KM)
    return result_NM

# This works fine
A = torch.randn(3, 4)  # N=3, K=4
B = torch.randn(4, 5)  # K=4, M=5
result = matrix_multiply(A, B)  # Shape: (3, 5)

# This raises an AssertionError
A = torch.randn(3, 4)
B = torch.randn(5, 6)  # Wrong! K dimensions don't match
result = matrix_multiply(A, B)  # AssertionError!
```

## Shape Annotation Format

Variable names should follow the pattern: `name_dimensions`

Each character in the dimensions suffix represents one dimension:
- `tensor_NK`: 2D tensor with dimensions N × K
- `data_BCHW`: 4D tensor with dimensions B × C × H × W

Only single capital letters are supported: `NK`, `BCHW`, `IJ`

## What Gets Checked

The decorator automatically adds shape validation for:

1. **Function arguments** with underscores in their names
2. **Variable assignments** to names containing underscores
3. **Augmented assignments** (+=, -=, *=, etc.)
4. **Annotated assignments** (PEP 526 style)
