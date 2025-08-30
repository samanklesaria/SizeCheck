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
def matrix_multiply(A_nk, B_km):
    """Matrix multiplication with automatic shape checking."""
    result_nm = torch.matmul(A_nk, B_km)
    return result_nm

# This works fine
A = torch.randn(3, 4)  # n=3, k=4
B = torch.randn(4, 5)  # k=4, m=5
result = matrix_multiply(A, B)  # Shape: (3, 5)

# This raises an AssertionError
A = torch.randn(3, 4)
B = torch.randn(5, 6)  # Wrong! k dimensions don't match
result = matrix_multiply(A, B)  # AssertionError!
```

## Shape Annotation Format

Variable names should follow the pattern: `name_dimensions`

Each character in the dimensions suffix represents one dimension:
- `tensor_nk`: 2D tensor with dimensions n × k
- `data_bchw`: 4D tensor with dimensions b × c × h × w

Only single letters are supported: `nk`, `bchw`, `ij`

## What Gets Checked

The decorator automatically adds shape validation for:

1. **Function arguments** with underscores in their names
2. **Variable assignments** to names containing underscores
3. **Augmented assignments** (+=, -=, *=, etc.)
4. **Annotated assignments** (PEP 526 style)
