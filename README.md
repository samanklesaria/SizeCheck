# Runtime Shape Validation for Size-Annotated Python Code

The `sizecheck` package provides a decorator that automatically adds runtime shape checking to Python functions based on size-annotated variable names using AST transformation. [![][docs-dev-img]][docs-dev-url]

For previous versions of the documentation, substitute `latest` in the documentation URL above for the desired version.

## Overview

When writing PyTorch or NumPy code, it's common to use naming conventions that indicate tensor shapes, as in this [Medium post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd). For example, if a tensor `weights` has shape `N × K`, you might name the variable `weights_NK`. This library automatically validates that tensors match their annotated shapes at runtime by analyzing and modifying your function's Abstract Syntax Tree (AST).

## Key Features

- **AST-based transformation**: Automatically injects shape checks into function arguments and variable assignments
- **Intuitive naming convention**: Use underscores to indicate tensor shapes
- **Framework agnostic**: Works with PyTorch, NumPy, Jax, and any other libraries that use `.shape` to indicate tensor shapes.
- **Comprehensive checking**: Validates both function parameters and intermediate assignments
- **Clear error messages**: Provides detailed information when shapes don't match

## Quick Start

```python
import torch
from sizecheck import sizecheck

@sizecheck
def matrix_multiply(a_NK, b_KM):
    """Matrix multiplication with automatic shape checking."""
    result_NM = torch.matmul(a_NK, b_KM)
    return result_NM

# This works fine
a_NK = torch.randn(3, 4)  # N=3, K=4
b_KM = torch.randn(4, 5)  # K=4, M=5
result = matrix_multiply(a_NK, b_KM)  # Shape: (3, 5)

# This raises an AssertionError
a_NK = torch.randn(3, 4)
b_KM = torch.randn(5, 6)  # Wrong! K dimensions don't match
result = matrix_multiply(a_NK, b_KM)  # AssertionError!
```

## Shape Annotation Format

Each character in the dimensions suffix represents one dimension:
- `tensor_NK`: 2D tensor with dimensions N × K
- `data_BCHW`: 4D tensor with dimensions B × C × H × W

Dimension annotations can contain:

- Variable dimensions (uppercase letters): `N`, `K`, `M` - stored in variables of the same name
- Constant dimensions (single digits): `3`, `4`, `2` - checked for exact size

## What Gets Checked

The decorator automatically adds shape validation for:

- **Function arguments** with underscores in their names
- **Variable assignments** to names containing underscores

## Dimension Scope

The dimensions are scoped to the function they are defined in. For example, if you define a function `foo` with a parameter `x_NK`, the dimension `N` is only valid within the scope of `foo`. If you define another function `bar` with a parameter `y_NL`, this dimension `N` can differ from the one in `foo`, but it is only valid within the scope of `bar`.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://samanklesaria.github.io/sizecheck/latest

## Julia version

If you're looking for the Julia version of this library, check out [SizeCheck.jl](https://github.com/samanklesaria/SizeCheck.jl).
