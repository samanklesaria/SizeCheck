import torch
import pytest
from shapecheck import shapecheck

@shapecheck
def matrix_operations(A_nk, B_km, C_nm):
    """Matrix operations with comprehensive shape checking."""
    # These assignments will be automatically checked
    intermediate_nm = torch.matmul(A_nk, B_km)
    intermediate_nm += C_nm
    return intermediate_nm

@shapecheck
def neural_network_layer(input_bh, weights_ho, bias_o):
    """Neural network layer with automatic shape validation."""
    linear_output_bo = torch.matmul(input_bh, weights_ho)
    activated_bo = linear_output_bo + bias_o
    output_bo = torch.relu(activated_bo)
    return output_bo


@shapecheck
def convolutional_block(input_bchw, kernel_oihw, bias_o):
    """Convolutional operations with shape checking."""
    conv_output_bohw = torch.conv2d(input_bchw, kernel_oihw, bias_o)
    mean_o = conv_output_bohw.mean(dim=[0, 2, 3])
    normalized_bohw = conv_output_bohw - mean_o.view(1, -1, 1, 1)
    activated_bohw = torch.relu(normalized_bohw)
    return activated_bohw


@shapecheck
def transformer_attention(queries_bsh, keys_bsh, values_bsh):
    """Multi-head attention with automatic shape validation."""
    scores_bss = torch.matmul(queries_bsh, keys_bsh.transpose(-2, -1))
    scaled_scores_bss = scores_bss * 0.1
    attention_weights_bss = torch.softmax(scaled_scores_bss, dim=-1)
    attended_values_bsh = torch.matmul(attention_weights_bss, values_bsh)
    return attended_values_bsh

def test_matrix_operations():
    """Test matrix operations."""
    print("1. Matrix operations...")
    A_nk = torch.randn(3, 4)
    B_km = torch.randn(4, 5)
    C_nm = torch.randn(3, 5)
    matrix_operations(A_nk, B_km, C_nm)

def test_convolutional_block():
    input_bchw = torch.randn(1, 3, 32, 32)
    kernel_oihw = torch.randn(16, 3, 3, 3)
    bias_o = torch.randn(16)
    convolutional_block(input_bchw, kernel_oihw, bias_o)


def test_transformer_attention():
    batch, seq_len, hidden = 2, 8, 64
    queries_bsh = torch.randn(batch, seq_len, hidden)
    keys_bsh = torch.randn(batch, seq_len, hidden)
    values_bsh = torch.randn(batch, seq_len, hidden)
    transformer_attention(queries_bsh, keys_bsh, values_bsh)

@pytest.mark.xfail(reason="Expected dimension mismatch", raises=AssertionError)
def test_dimension_mismatch():
    """Test that shape checking catches dimension mismatches."""
    input_bchw = torch.randn(1, 3, 32, 32)  # batch=1, channels=3, height=32, width=32
    kernel_oihw = torch.randn(16, 3, 3, 3)  # out=16, in=3, height=3, width=3
    bias_o = torch.randn(16)  # 16 output channels
    convolutional_block(input_bchw, kernel_oihw, bias_o)
