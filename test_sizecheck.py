import torch
import pytest
from sizecheck import sizecheck

@sizecheck
def matrix_operations(A_NK, B_KM, C_NM):
    """Matrix operations with comprehensive shape checking."""
    # These assignments will be automatically checked
    intermediate_NM = torch.matmul(A_NK, B_KM)
    intermediate_NM += C_NM
    return intermediate_NM

@sizecheck
def transformer_attention(queries_BSH, keys_BSH, values_BSH):
    """Multi-head attention with automatic shape validation."""
    scores_BSS = torch.matmul(queries_BSH, keys_BSH.transpose(-2, -1))
    scaled_scores_BSS = scores_BSS * 0.1
    attention_weights_BSS = torch.softmax(scaled_scores_BSS, dim=-1)
    attended_values_BSH = torch.matmul(attention_weights_BSS, values_BSH)
    return attended_values_BSH

@sizecheck
def destructuring_function():
    """Test function with destructuring assignments."""
    # Create some tensors to destructure
    tensor1_NK = torch.randn(3, 4)
    tensor2_KM = torch.randn(4, 5)

    # Destructuring assignment - both should be checked
    result1_NM, result2_KN = torch.matmul(tensor1_NK, tensor2_KM), tensor1_NK.T

    # List destructuring
    [final1_NM, final2_KN] = [result1_NM, result2_KN]

    return final1_NM, final2_KN

def test_matrix_operations():
    """Test matrix operations."""
    print("1. Matrix operations...")
    A_NK = torch.randn(3, 4)
    B_KM = torch.randn(4, 5)
    C_NM = torch.randn(3, 5)
    matrix_operations(A_NK, B_KM, C_NM)

def test_transformer_attention():
    batch, seq_len, hidden = 2, 8, 64
    queries_BSH = torch.randn(batch, seq_len, hidden)
    keys_BSH = torch.randn(batch, seq_len, hidden)
    values_BSH = torch.randn(batch, seq_len, hidden)
    transformer_attention(queries_BSH, keys_BSH, values_BSH)

def test_destructuring_assignments():
    """Test destructuring assignments with shape checking."""
    destructuring_function()

class LinearLayer:
    """Test class with shape-checked method."""

    @sizecheck
    def forward(self, input_BH, weight_HO):
        """Forward pass with shape checking."""
        output_BO = torch.matmul(input_BH, weight_HO)
        return output_BO

def test_class_method():
    """Test class method with @sizecheck decorator."""
    layer = LinearLayer()
    input_BH = torch.randn(2, 3)
    weight_HO = torch.randn(3, 4)
    output = layer.forward(input_BH, weight_HO)
    assert output.shape == (2, 4)


@pytest.mark.xfail(reason="Expected dimension mismatch", raises=AssertionError)
def test_dimension_mismatch():
    """Test that shape checking catches dimension mismatches."""
    A_NK = torch.randn(3, 4)
    B_KM = torch.randn(5, 6)  # Wrong dimensions - K should be 4, not 5
    C_NM = torch.randn(3, 6)
    matrix_operations(A_NK, B_KM, C_NM)
