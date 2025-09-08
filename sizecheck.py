"""
This library provides `sizecheck`, a macro that automatically adds runtime shape checking to Julia functions based on size-annotated variable names.

When writing Julia code, it's common to use naming conventions that indicate
tensor shapes, as in this [Medium
post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).
For example, if a tensor `weights` has shape `N × K`, you might name the
variable `weights_NK`. This macro adds validation checks that tensors match
their annotated shapes at runtime.
"""
import ast
import inspect
import textwrap
from typing import Dict, List, Any, Union

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

def _create_check_shape_call(var_name: str, dims: List[str], first_vars: Dict[str, str], lineno: int = 1) -> List[ast.stmt]:
    """Create statements for shape checking and dimension variable assignment."""
    statements = []

    # Create if statement to check if tensor has shape attribute
    if_test = ast.Call(
        func=ast.Name(id='hasattr', ctx=ast.Load()),
        args=[
            ast.Name(id=var_name, ctx=ast.Load()),
            ast.Constant(value='shape')
        ],
        keywords=[]
    )

    if_body = []

    # Check dimension count
    dim_count_check = ast.Expr(
        value=ast.Call(
            func=ast.Name(id='_check_dimension_count', ctx=ast.Load()),
            args=[
                ast.Name(id=var_name, ctx=ast.Load()),
                ast.Constant(value=len(dims)),
                ast.Constant(value=var_name)
            ],
            keywords=[]
        )
    )
    if_body.append(dim_count_check)

    for i, dim in enumerate(dims):
        # Create shape access: var_name.shape[i]
        shape_access = ast.Subscript(
            value=ast.Attribute(
                value=ast.Name(id=var_name, ctx=ast.Load()),
                attr='shape',
                ctx=ast.Load()
            ),
            slice=ast.Constant(value=i),
            ctx=ast.Load()
        )

        if dim.isdigit():
            # Numeric literal - check against constant
            check_expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='check_literal_dimension', ctx=ast.Load()),
                    args=[
                        shape_access,
                        ast.Constant(value=int(dim)),
                        ast.Constant(value=dim),
                        ast.Constant(value=var_name)
                    ],
                    keywords=[]
                )
            )
            if_body.append(check_expr)
        elif dim in first_vars and first_vars[dim] != var_name:
            # Check dimension matches existing variable
            check_expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='check_dimension', ctx=ast.Load()),
                    args=[
                        shape_access,
                        ast.Name(id=dim, ctx=ast.Load()),
                        ast.Constant(value=dim),
                        ast.Constant(value=var_name),
                        ast.Constant(value=first_vars[dim])
                    ],
                    keywords=[]
                )
            )
            if_body.append(check_expr)
        else:
            # Assign dimension to variable (first time we see this dimension)
            assign = ast.Assign(
                targets=[ast.Name(id=dim, ctx=ast.Store())],
                value=shape_access
            )
            if_body.append(assign)

    # Create the if statement
    if_stmt = ast.If(
        test=if_test,
        body=if_body,
        orelse=[]
    )
    if_stmt.lineno = lineno
    if_stmt.col_offset = 0
    statements.append(if_stmt)

    return statements

class _SizeCheckTransformer(ast.NodeTransformer):
    """AST transformer that injects shape checking code."""

    def __init__(self):
        self.first_vars = {}  # Maps dimension names to first variable using them

    def extract_names_from_target(self, target):
        """Recursively extract all names from assignment targets."""
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)):
            names = []
            for elt in target.elts:
                names.extend(self.extract_names_from_target(elt))
            return names
        else:
            return []

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, List[ast.stmt]]:
        """Transform assignment nodes to add shape checks."""
        node = self.generic_visit(node)  # type: ignore
        new_nodes: List[ast.stmt] = [node]
        for target in node.targets:
            names = self.extract_names_from_target(target)
            for name in names:
                dims = _extract_shape_dims(name)
                if dims:
                    for dim in dims:
                        if dim not in self.first_vars and not dim.isdigit():
                            self.first_vars[dim] = name
                    check_nodes = _create_check_shape_call(name, dims, self.first_vars, node.lineno)
                    new_nodes.extend(check_nodes)
        if len(new_nodes) > 1:
            return new_nodes
        else:
            return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Union[ast.AnnAssign, List[ast.stmt]]:
        """Handle annotated assignments."""
        node = self.generic_visit(node)  # type: ignore
        names = self.extract_names_from_target(node.target)
        check_nodes: List[ast.stmt] = []
        for name in names:
            dims = _extract_shape_dims(name)
            if dims:
                for dim in dims:
                    if dim not in self.first_vars and not dim.isdigit():
                        self.first_vars[dim] = name
                check_nodes.extend(_create_check_shape_call(name, dims, self.first_vars, node.lineno))
        if check_nodes:
            return [node] + check_nodes
        else:
            return node

    def visit_AugAssign(self, node: ast.AugAssign) -> Union[ast.AugAssign, List[ast.stmt]]:
        """Handle augmented assignments (+=, -=, etc.)."""
        node = self.generic_visit(node)  # type: ignore
        if isinstance(node.target, ast.Name):
            dims = _extract_shape_dims(node.target.id)
            if dims:
                for dim in dims:
                    if dim not in self.first_vars and not dim.isdigit():
                        self.first_vars[dim] = node.target.id
                check_nodes = _create_check_shape_call(node.target.id, dims, self.first_vars, node.lineno)
                return [node] + check_nodes
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definition to add argument checks."""
        arg_checks: List[ast.stmt] = []

        # Check regular arguments
        for arg in node.args.args:
            if '_' in arg.arg:
                dims = _extract_shape_dims(arg.arg)
                if dims:
                    for dim in dims:
                        if dim not in self.first_vars and not dim.isdigit():
                            self.first_vars[dim] = arg.arg
                    arg_checks.extend(_create_check_shape_call(arg.arg, dims, self.first_vars, node.lineno))

        # Check keyword-only arguments
        for arg in node.args.kwonlyargs:
            if '_' in arg.arg:
                dims = _extract_shape_dims(arg.arg)
                if dims:
                    for dim in dims:
                        if dim not in self.first_vars and not dim.isdigit():
                            self.first_vars[dim] = arg.arg
                    arg_checks.extend(_create_check_shape_call(arg.arg, dims, self.first_vars, node.lineno))

        # Transform the function body
        new_body = []
        for stmt in node.body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_body.extend(transformed)
            else:
                new_body.append(transformed)

        # Insert argument checks at the beginning
        node.body = arg_checks + new_body

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

    1. **Function arguments** with underscores in their names
    2. **Variable assignments** to names containing underscores, including destructuring assignments
    3. **Augmented assignments** (+=, -=, *=, etc.)

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
    # Adjust line numbers in the parsed tree
    ast.increment_lineno(tree, original_lineno - 1)
    transformer = _SizeCheckTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    # print(ast.dump(new_tree, indent=4))
    code = compile(new_tree, filename=original_filename, mode='exec')
    namespace = func.__globals__.copy()
    namespace['_check_dimension_count'] = _check_dimension_count
    namespace['check_dimension'] = _check_dimension
    namespace['check_literal_dimension'] = _check_literal_dimension
    exec(code, namespace)
    return namespace[func.__name__]

__all__ = ["sizecheck"]
