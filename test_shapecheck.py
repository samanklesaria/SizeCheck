import torch
import pytest
from shapecheck import shapecheck

@shapecheck
def matrix_operations(A_NK, B_KM, C_NM):
    """Matrix operations with comprehensive shape checking."""
    # These assignments will be automatically checked
    intermediate_NM = torch.matmul(A_NK, B_KM)
    intermediate_NM += C_NM
    return intermediate_NM

@shapecheck
def transformer_attention(queries_BSH, keys_BSH, values_BSH):
    """Multi-head attention with automatic shape validation."""
    scores_BSS = torch.matmul(queries_BSH, keys_BSH.transpose(-2, -1))
    scaled_scores_BSS = scores_BSS * 0.1
    attention_weights_BSS = torch.softmax(scaled_scores_BSS, dim=-1)
    attended_values_BSH = torch.matmul(attention_weights_BSS, values_BSH)
    return attended_values_BSH

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

@pytest.mark.xfail(reason="Expected dimension mismatch", raises=AssertionError)
def test_dimension_mismatch():
    """Test that shape checking catches dimension mismatches."""
    A_NK = torch.randn(3, 4)
    B_KM = torch.randn(5, 6)  # Wrong dimensions - K should be 4, not 5
    C_NM = torch.randn(3, 6)
    matrix_operations(A_NK, B_KM, C_NM)
