"""
This library provides `sizecheck`, a macro that automatically adds runtime shape checking to Python functions based on size-annotated variable names.

When writing Python code, it's common to use naming conventions that indicate
tensor shapes, as in this [Medium
post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).
For example, if a tensor `weights` has shape `N × K`, you might name the
variable `weights_NK`. This macro adds validation checks that tensors match
their annotated shapes at runtime.
"""
import ast
import inspect
import textwrap
from typing import Dict, List, Any
from collections.abc import Iterable
from itertools import chain

def _quasiquote(template: str, **kwargs) -> ast.expr:
    """Generic quasi-quotation function that parses string templates and substitutes values."""
    tree = ast.parse(template, mode='eval')

    class TemplateTransformer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in kwargs:
                return kwargs[node.id]
            return node

    transformer = TemplateTransformer()
    return transformer.visit(tree.body)

def _extract_shape_dims(name: str) -> List[str]:
    """Extract shape dimensions from annotated variable name."""
    parts = name.split('_')
    if len(parts) < 2:
        return []
    shape_part = parts[-1]
    if all(c.isupper() or c.isdigit() for c in shape_part) and shape_part.isalnum():
        return list(shape_part)
    else:
        return []

def _create_check_shape_call(target, dims: List[str], first_vars: Dict[str, str], original_node: ast.AST) -> List[ast.stmt]:
    """Create statements for shape checking and dimension variable assignment.

    Args:
        target: Either a string (variable name) or ast.Attribute (property)
        dims: List of dimension names
        first_vars: Dictionary tracking first occurrence of dimension variables
    """
    statements = []

    if isinstance(target, str):
        var_name = target
        target_load = ast.Name(id=var_name, ctx=ast.Load())
        display_name = var_name
    else:
        var_name = target.attr
        target_load = ast.Attribute(
            value=target.value,
            attr=target.attr,
            ctx=ast.Load()
        )
        display_name = target.attr

    # Check dimension count using quasi-quotation
    dim_count_check = ast.Expr(
        value=_quasiquote(
            "_check_dimension_count(target, dim_count, display)",
            target=target_load,
            dim_count=ast.Constant(value=len(dims)),
            display=ast.Constant(value=display_name)
        )
    )
    statements.append(dim_count_check)

    for i, dim in enumerate(dims):
        shape_access = _quasiquote(
            "target.shape[index]",
            target=target_load,
            index=ast.Constant(value=i)
        )

        if dim.isdigit():
            check_expr = ast.Expr(
                value=_quasiquote(
                    "_check_literal_dimension(shape, literal, dim_name, var_name)",
                    shape=shape_access,
                    literal=ast.Constant(value=int(dim)),
                    dim_name=ast.Constant(value=dim),
                    var_name=ast.Constant(value=display_name)
                )
            )
            statements.append(check_expr)
        elif dim in first_vars and first_vars[dim] != var_name:
            first_var_name = first_vars[dim] if first_vars[dim] is not None else "explicitly provided"
            check_expr = ast.Expr(
                value=_quasiquote(
                    "_check_dimension(shape, expected, dim_name, var_name, first_var)",
                    shape=shape_access,
                    expected=ast.Name(id=dim, ctx=ast.Load()),
                    dim_name=ast.Constant(value=dim),
                    var_name=ast.Constant(value=display_name),
                    first_var=ast.Constant(value=first_var_name)
                )
            )
            statements.append(check_expr)
        else:
            assign = ast.Assign(
                targets=[ast.Name(id=dim, ctx=ast.Store())],
                value=shape_access
            )
            statements.append(assign)

    # Copy location information from the original node to all generated statements
    for stmt in statements:
        ast.copy_location(stmt, original_node)
        for child in ast.walk(stmt):
            if not hasattr(child, 'lineno'):
                ast.copy_location(child, original_node)

    return statements

class _SizeCheckTransformer(ast.NodeTransformer):
    """AST transformer that injects shape checking code."""

    def __init__(self):
        self.first_vars : dict[str, str] = {}  # Maps dimension names to first variable using them

    def extract_names_from_target(self, target) -> Iterable[str]:
        """Recursively extract all names from assignment targets."""
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)):
            return chain.from_iterable([self.extract_names_from_target(elt) for elt in target.elts])
        elif isinstance(target, ast.Attribute):
            return [target.attr]
        else:
            return []

    def visit_Assign(self, node: ast.Assign) -> List[ast.stmt]:
        return self.handle_assignment(node.targets, node)

    def handle_assignment(self, targets: List[ast.expr], node: ast.Assign) -> List[ast.stmt]:
        """Transform assignment nodes to add shape checks."""
        node = self.generic_visit(node)  # type: ignore
        new_nodes: List[ast.stmt] = [node]

        for target in targets:
            names = self.extract_names_from_target(target)
            for name in names:
                # Check if this is an explicit assignment to a dimension variable
                if len(name) == 1 and name.isupper():
                    if name in self.first_vars and self.first_vars[name] is not None:
                        # Error: assigning to dimension variable after it's been used
                        error_stmt = ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(id='ValueError', ctx=ast.Load()),
                                args=[ast.Constant(value=f"Cannot assign to dimension variable '{name}' after it has been used in shape annotations")],
                                keywords=[]
                            ),
                            cause=None)
                        ast.copy_location(error_stmt, node)
                        new_nodes.append(error_stmt)
                    elif name not in self.first_vars:
                        # Mark as explicitly provided
                        self.first_vars[name] = None

                # Handle shape-annotated variables and properties
                dims = _extract_shape_dims(name)
                if dims:
                    for dim in dims:
                        if dim not in self.first_vars and not dim.isdigit():
                            self.first_vars[dim] = name

                    if isinstance(target, ast.Attribute):
                        # For property assignments, pass the ast.Attribute directly
                        check_nodes = _create_check_shape_call(target, dims, self.first_vars, node)
                    else:
                        # For regular variables, pass the variable name
                        check_nodes = _create_check_shape_call(name, dims, self.first_vars, node)

                    new_nodes.extend(check_nodes)

        return new_nodes

    def visit_AnnAssign(self, node: ast.AnnAssign) -> List[ast.stmt]:
        return self.handle_assignment([node.target], node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definition to add argument checks."""
        arg_checks: List[ast.stmt] = []

        # Check arguments
        for arg in chain(node.args.args, node.args.kwonlyargs):
            dims = _extract_shape_dims(arg.arg)
            if dims:
                for dim in dims:
                    if dim not in self.first_vars and not dim.isdigit():
                        self.first_vars[dim] = arg.arg
                arg_checks.extend(_create_check_shape_call(arg.arg, dims, self.first_vars, node))

        # Transform the function body
        node = self.generic_visit(node)

        # Insert argument checks at the beginning
        node.body = arg_checks + node.body

        return node


def _check_dimension_count(tensor: Any, expected_ndims: int, var_name: str) -> None:
    """Check if tensor has correct number of dimensions."""
    if len(tensor.shape) != expected_ndims:
        raise ValueError(
            f"Shape dimension mismatch for {var_name}: expected {expected_ndims} dimensions, got {len(tensor.shape)}"
        )

def _check_dimension(actual_dim: int, expected_dim: int, dim_name: str, var_name: str, first_var: str) -> None:
    """Runtime dimension checking function."""
    if actual_dim != expected_dim:
        raise AssertionError(
            f"Shape mismatch for {var_name} dimension {dim_name}: expected {expected_dim} (from {first_var}), got {actual_dim}"
        )

def _check_literal_dimension(actual_dim: int, expected_literal: int, dim_name: str, var_name: str) -> None:
    """Runtime checking function for literal dimensions."""
    if actual_dim != expected_literal:
        raise AssertionError(
            f"Shape mismatch for {var_name} dimension {dim_name}: expected {expected_literal}, got {actual_dim}"
        )

def sizecheck(func):
    """
    Automatically adds runtime shape checking to functions based on
    size-annotated variable names. Variables with underscores followed by dimension
    letters (e.g., `x_NK`) are validated to ensure consistent shapes.

    Dimension annotations can contain:

    - Variable dimensions (uppercase letters): `N`, `K`, `M` - stored in variables of the same name
    - Constant dimensions (single digits): `3`, `4`, `2` - checked for exact size

    The macro automatically adds shape validation for:

    - **Function arguments** with underscores in their names
    - **Variable assignments** to names containing underscores, including destructuring assignments and property assignments.

    **Explicit dimension assignment**: You can explicitly assign values to dimension variables
    (single uppercase letters) before using them in shape annotations. This allows you to
    specify exact dimension sizes that tensors must match:

    ```python
    @sizecheck
    def example():
        N = 10  # Explicit assignment
        x_N = torch.randn(10)  # Must have size 10 in first dimension
    ```

    Once a dimension variable has been used in a shape annotation, it cannot be reassigned.

    The dimensions are scoped to the function they are defined in.
    For example, if you define a function `foo` with a parameter `x_NK`, the dimension `N` is only valid within the scope of `foo`.
    If you define another function `bar` with a parameter `y_NL`, this dimension `N` can differ from the one in `foo`,
    but it is only valid within the scope of `bar`.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with shape checking

    Example:
        ```python
        @sizecheck
        def matrix_multiply(A_NK, B_KM):
            result_NM = torch.matmul(A_NK, B_KM)
            return result_NM

        # This works fine
        a_NK = torch.randn(3, 4)  # N=3, K=4
        b_KM = torch.randn(4, 5)  # K=4, M=5
        result = matrix_multiply(a_NK, b_KM)  # size: (3, 5)

        # This raises an error
        a_NK = torch.randn(3, 4)
        b_KM = torch.randn(5, 6)  # Wrong! K dimensions don't match
        result = matrix_multiply(a_NK, b_KM)  # Error!
        ```
    """
    source = textwrap.dedent(inspect.getsource(func))
    original_filename = inspect.getfile(func)
    original_lineno = inspect.getsourcelines(func)[1]
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.FunctionDef):
        tree.body[0].decorator_list.pop(0)
    transformer = _SizeCheckTransformer()
    new_tree = transformer.visit(tree)

    # Adjust line numbers and fix missing locations
    ast.increment_lineno(new_tree, original_lineno - 1)
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, filename=original_filename, mode='exec')
    namespace = func.__globals__
    namespace['_check_dimension_count'] = _check_dimension_count
    namespace['_check_dimension'] = _check_dimension
    namespace['_check_literal_dimension'] = _check_literal_dimension
    exec(code, namespace)
    return namespace[func.__name__]

__all__ = ["sizecheck"]
