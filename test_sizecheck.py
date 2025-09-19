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

def test_explicit_assignment_before_use():
    """Test that explicit assignment before use shows 'explicitly provided' in error messages."""
    @sizecheck
    def test_func():
        N = 5  # explicit assignment
        x_N = torch.randn(3)  # should fail with "explicitly provided"
        return x_N

    with pytest.raises(AssertionError, match=r"expected 5 \(from explicitly provided\)"):
        test_func()

def test_explicit_assignment_after_use_raises_error():
    """Test that explicit assignment after use raises ValueError."""
    @sizecheck
    def test_func(x_N):
        N = 10  # should raise error at runtime
        return x_N

    with pytest.raises(ValueError, match="Cannot assign to dimension variable 'N' after it has been used"):
        test_func(torch.randn(5))

def test_explicit_assignment_after_use_in_body():
    """Test that explicit assignment after shape annotation in body raises error."""
    @sizecheck
    def test_func():
        x_N = torch.randn(5)  # N gets recorded
        N = 10  # should raise error at runtime
        return x_N

    with pytest.raises(ValueError, match="Cannot assign to dimension variable 'N' after it has been used"):
        test_func()

def test_multiple_explicit_assignments():
    """Test multiple explicit assignments before use."""
    @sizecheck
    def test_func():
        N = 5
        K = 3
        x_NK = torch.randn(4, 3)  # N mismatch
        return x_NK

    with pytest.raises(AssertionError, match=r"expected 5 \(from explicitly provided\)"):
        test_func()

def test_annotated_assignment_explicit():
    """Test explicit assignment with annotated assignments."""
    @sizecheck
    def test_func():
        N = 5
        x_N: torch.Tensor = torch.randn(3)
        return x_N

    with pytest.raises(AssertionError, match=r"expected 5 \(from explicitly provided\)"):
        test_func()

def test_annotated_assignment_after_use():
    """Test annotated assignment after use raises error."""
    @sizecheck
    def test_func():
        x_N = torch.randn(5)
        N: int = 10  # should raise error
        return x_N

    with pytest.raises(ValueError, match="Cannot assign to dimension variable 'N' after it has been used"):
        test_func()

def test_mixed_explicit_and_shape_vars():
    """Test mixing explicit assignments with normal shape variables."""
    @sizecheck
    def test_func(x_MN):
        K = 7  # explicit assignment to unused dimension
        y_NK = torch.randn(5, 7)  # N from x_MN, K explicitly provided
        return x_MN, y_NK

    x = torch.randn(3, 5)  # M=3, N=5
    y_expected_shape = (5, 7)  # N=5, K=7
    result = test_func(x)
    assert result[1].shape == y_expected_shape

def test_multi_letter_vars_not_affected():
    """Test that multi-letter variables are not treated as dimension vars."""
    @sizecheck
    def test_func():
        NN = 10  # should not be treated as dimension variable
        x_N = torch.randn(5)  # N should work normally
        return x_N

    result = test_func()
    assert result.shape == (5,)

def test_lowercase_vars_not_affected():
    """Test that lowercase variables are not treated as dimension vars."""
    @sizecheck
    def test_func():
        n = 10  # should not be treated as dimension variable
        x_N = torch.randn(5)  # N should work normally
        return x_N

    result = test_func()
    assert result.shape == (5,)

def test_reassignment_of_explicit_var():
    """Test reassigning an explicitly provided dimension variable."""
    @sizecheck
    def test_func():
        N = 5
        N = 7  # reassigning explicit var should work
        x_N = torch.randn(7)  # should use latest value
        return x_N

    result = test_func()
    assert result.shape == (7,)

def test_explicit_assignment_before_use_exact_message():
    """Verify that explicit assignment before use shows 'explicitly provided' in error message."""
    @sizecheck
    def test_func():
        N = 10  # explicit assignment to dimension variable
        x_N = torch.randn(5)  # mismatch: expected 10, got 5
        return x_N

    with pytest.raises(AssertionError) as exc_info:
        test_func()

    error_msg = str(exc_info.value)
    assert "expected 10 (from explicitly provided)" in error_msg
    assert "got 5" in error_msg

def test_mixed_explicit_and_inferred():
    """Test mixing explicit assignments with normal dimension inference."""
    @sizecheck
    def test_func():
        N = 3  # explicit
        x_NK = torch.randn(3, 4)  # N explicit, K inferred to 4
        y_KM = torch.randn(4, 5)  # K=4 from x_NK, M inferred to 5
        return x_NK, y_KM

    result = test_func()
    assert result[0].shape == (3, 4)
    assert result[1].shape == (4, 5)

    # Test mismatch in mixed scenario
    @sizecheck
    def test_func2():
        N = 3  # explicit
        x_NK = torch.randn(3, 4)  # N explicit, K inferred
        y_KM = torch.randn(2, 5)  # K mismatch with inferred value
        return x_NK, y_KM

    with pytest.raises(AssertionError) as exc_info:
        test_func2()

    error_msg = str(exc_info.value)
    assert "expected 4 (from x_NK)" in error_msg  # Should reference the inferred source

class Model:
    """Test class for property assignment shape checking."""
    def __init__(self):
        pass

    @sizecheck
    def property_assignment_correct(self, A_BL, B_LH):
        """Test that property assignments with shape suffixes work correctly."""
        # Matrix multiplication A_BL @ B_LH results in shape BH
        self.inputs_BH = A_BL @ B_LH
        return self.inputs_BH

    @sizecheck
    def property_assignment_mismatch(self, A_BL, B_LH):
        """Test that property assignments catch shape mismatches."""
        # This should raise an error because A_BL @ B_LH has shape BH, not BL
        self.result_BL = A_BL @ B_LH
        return self.result_BL

    @sizecheck
    def multiple_property_assignments(self, A_BL, B_LH, C_BH):
        """Test multiple property assignments in one function."""
        self.intermediate_BH = A_BL @ B_LH
        self.final_BH = self.intermediate_BH + C_BH
        return self.final_BH

def test_property_assignment_works():
    """Test that correct property assignment shapes work."""
    model = Model()
    A_BL = torch.randn(3, 4)  # B=3, L=4
    B_LH = torch.randn(4, 5)  # L=4, H=5

    result = model.property_assignment_correct(A_BL, B_LH)
    assert result.shape == (3, 5)  # B=3, H=5

@pytest.mark.xfail(reason="Expected property assignment shape mismatch", raises=AssertionError)
def test_property_assignment_shape_mismatch():
    """Test that property assignments catch shape mismatches."""
    model = Model()
    A_BL = torch.randn(3, 4)  # B=3, L=4
    B_LH = torch.randn(4, 5)  # L=4, H=5

    # This should fail because A_BL @ B_LH has shape BH, not BL
    model.property_assignment_mismatch(A_BL, B_LH)

def test_multiple_property_assignments_work():
    """Test that multiple property assignments work correctly."""
    model = Model()
    A_BL = torch.randn(2, 3)  # B=2, L=3
    B_LH = torch.randn(3, 4)  # L=3, H=4
    C_BH = torch.randn(2, 4)  # B=2, H=4

    result = model.multiple_property_assignments(A_BL, B_LH, C_BH)
    assert result.shape == (2, 4)  # B=2, H=4

def test_error_line_numbers():
    """Test that errors thrown by generated code have reasonable line numbers."""
    import traceback
    import inspect

    @sizecheck
    def test_function_with_error():
        x_NK = torch.randn(3, 4)  # N=3, K=4
        y_KM = torch.randn(2, 5)  # ERROR: K should be 4, not 2
        return x_NK, y_KM

    try:
        test_function_with_error()
        assert False, "Expected AssertionError was not raised"
    except AssertionError as e:
        # Get the traceback to check line numbers
        tb = traceback.extract_tb(e.__traceback__)

        # Find the frame that corresponds to our test function
        error_frame = None
        for frame in tb:
            if frame.name == 'test_function_with_error':
                error_frame = frame
                break

        assert error_frame is not None, f"Could not find error frame in traceback: {[f.name for f in tb]}"

        # Get the function's starting line for reference
        func_start_line = inspect.getsourcelines(test_function_with_error)[1]
        actual_line = error_frame.lineno

        # The error should be reasonably close to the function definition
        # Currently the generated code may not have perfect line numbers,
        # but it should be within a reasonable range of the function
        line_diff = abs(actual_line - func_start_line)

        assert line_diff <= 5, (
            f"Error line {actual_line} is too far from function start line {func_start_line}. "
            f"Difference: {line_diff}. This suggests line numbers are completely wrong."
        )

        # Verify the error message mentions the correct variable and dimension
        error_msg = str(e)
        assert "y_KM" in error_msg, f"Error message should mention 'y_KM': {error_msg}"
        assert "dimension K" in error_msg, f"Error message should mention 'dimension K': {error_msg}"
        assert "expected 4" in error_msg, f"Error message should mention 'expected 4': {error_msg}"
        assert "got 2" in error_msg, f"Error message should mention 'got 2': {error_msg}"
