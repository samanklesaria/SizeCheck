import torch
import pytest
from sizecheck import sizecheck

@sizecheck
def matrix_operations(A_NK, B_KM, C_NM):
    """Matrix operations with comprehensive shape checking."""
    intermediate_NM = torch.matmul(A_NK, B_KM)
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
def test_destructuring_function():
    """Test function with destructuring assignments."""
    # Create some tensors to destructure
    tensor1_NK = torch.randn(3, 4)
    tensor2_KM = torch.randn(4, 5)

    # Destructuring assignment - both should be checked
    result1_NM, result2_KN = torch.matmul(tensor1_NK, tensor2_KM), tensor1_NK.T

    # List destructuring
    final1_NM, final2_KN = result1_NM, result2_KN

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

def test_dimension_variables_accessible():
    """Test that shape dimensions are accessible as raw variable names."""
    @sizecheck
    def test_function(x_NM, y_MK):
        # Dimension variables should be accessible directly
        assert N == x_NM.shape[0]
        assert M == x_NM.shape[1]
        assert M == y_MK.shape[0]  # Same M dimension
        assert K == y_MK.shape[1]

        # Use dimension variables in calculations
        result_NK = torch.zeros(N, K)
        return result_NK

    # Test with actual tensors
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    result = test_function(x, y)

    # Result should have correct shape based on dimension variables
    assert result.shape == (3, 5)

def test_numeric_literal_dimensions():
    """Test that numeric literals in shape dimensions work correctly."""
    @sizecheck
    def test_function(input_N3, weight_3M):
        # 3 should be checked as a literal, N and M as variables
        assert N == input_N3.shape[0]
        assert M == weight_3M.shape[1]
        # The middle dimensions should be exactly 3
        result_NM = torch.matmul(input_N3, weight_3M)
        return result_NM

    # Test with correct dimensions
    input_tensor = torch.randn(4, 3)  # N=4, literal=3
    weight_tensor = torch.randn(3, 5)  # literal=3, M=5
    result = test_function(input_tensor, weight_tensor)
    assert result.shape == (4, 5)

@pytest.mark.xfail(reason="Expected literal dimension mismatch", raises=AssertionError)
def test_numeric_literal_mismatch():
    """Test that numeric literal dimension checking catches mismatches."""
    @sizecheck
    def test_function(input_N3):
        return input_N3

    # This should fail because second dimension is 4, not 3
    input_tensor = torch.randn(5, 4)
    test_function(input_tensor)

def test_mixed_literals_and_variables():
    """Test that mixed numeric literals and variables work together."""
    @sizecheck
    def test_function(rgb_image_N3HW, conv_filter_36KK):
        # N, H, W are variables; 3, 6, K are checked/assigned appropriately
        assert N == rgb_image_N3HW.shape[0]
        assert H == rgb_image_N3HW.shape[2]
        assert W == rgb_image_N3HW.shape[3]
        assert K == conv_filter_36KK.shape[2]
        assert K == conv_filter_36KK.shape[3]  # K should be same for both dims

        # Simulate convolution output
        result_N6HW = torch.zeros(N, 6, H, W)
        return result_N6HW

    # Test with matching dimensions
    rgb = torch.randn(2, 3, 64, 48)  # N=2, 3 channels, H=64, W=48
    filt = torch.randn(3, 6, 5, 5)   # 3 in, 6 out, K=5 (5x5 kernel)
    result = test_function(rgb, filt)
    assert result.shape == (2, 6, 64, 48)

@pytest.mark.xfail(reason="Expected mixed literal/variable mismatch", raises=AssertionError)
def test_mixed_literals_mismatch():
    """Test error handling with mixed literals and variables."""
    @sizecheck
    def test_function(rgb_N3HW, filter_36KK):
        return torch.zeros(N, 6, H, W)

    # This should fail - first tensor has 4 channels, not 3
    rgb = torch.randn(2, 4, 32, 32)
    filt = torch.randn(3, 6, 3, 3)
    test_function(rgb, filt)

@pytest.mark.xfail(reason="Expected dimension mismatch", raises=AssertionError)
def test_dimension_mismatch():
    """Test that shape checking catches dimension mismatches."""
    A_NK = torch.randn(3, 4)
    B_KM = torch.randn(5, 6)  # Wrong dimensions - K should be 4, not 5
    C_NM = torch.randn(3, 6)
    matrix_operations(A_NK, B_KM, C_NM)
