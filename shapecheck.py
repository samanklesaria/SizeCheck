import ast
import inspect
from typing import Dict, List, Any

def extract_shape_dims(name: str) -> List[str]:
    """Extract shape dimensions from annotated variable name."""
    parts = name.split('_')
    if len(parts) < 2:
        return []
    shape_part = parts[-1]
    if shape_part.isalpha():
        return list(shape_part)
    else:
        return []

def create_check_shape_call(var_name: str, dims: List[str]) -> ast.stmt:
    """Create a call to the shape checking function."""
    dims_list = ast.List(elts=[ast.Constant(value=dim) for dim in dims], ctx=ast.Load())
    dims_list.lineno = 1
    dims_list.col_offset = 0

    expr = ast.Expr(
        value=ast.Call(
            func=ast.Name(id='check_shape', ctx=ast.Load()),
            args=[
                ast.Name(id=var_name, ctx=ast.Load()),
                dims_list,
                ast.Name(id="_dims_", ctx=ast.Load()),
                ast.Constant(value=var_name)
            ],
            keywords=[]
        )
    )
    # Set line numbers to avoid compilation errors
    expr.lineno = 1
    expr.col_offset = 0
    expr.value.lineno = 1
    expr.value.col_offset = 0
    return expr

class ShapeCheckTransformer(ast.NodeTransformer):
    """AST transformer that injects shape checking code."""

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Transform assignment nodes to add shape checks."""
        node = self.generic_visit(node)
        new_nodes = [node]
        for target in node.targets:
            if isinstance(target, ast.Name):
                dims = extract_shape_dims(target.id)
                if dims:
                    check_node = create_check_shape_call(target.id, dims)
                    new_nodes.append(check_node)
        return new_nodes if len(new_nodes) > 1 else node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Handle annotated assignments."""
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            dims = extract_shape_dims(node.target.id)
            if dims:
                check_node = create_check_shape_call(node.target.id, dims)
                return [node, check_node]
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        """Handle augmented assignments (+=, -=, etc.)."""
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            dims = extract_shape_dims(node.target.id)
            if dims:
                check_node = create_check_shape_call(node.target.id, dims)
                return [node, check_node]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definition to add argument checks."""
        # Collect shape-annotated arguments
        dims_assign = ast.Assign(targets=[ast.Name(id='_dims_', ctx=ast.Store())], value=ast.Dict())
        dims_assign.lineno = 1
        dims_assign.col_offset = 0
        arg_checks = [dims_assign]

        # Check regular arguments
        for arg in node.args.args:
            if '_' in arg.arg:
                dims = extract_shape_dims(arg.arg)
                if dims:
                    check_node = create_check_shape_call(arg.arg, dims)
                    arg_checks.append(check_node)

        # Check keyword-only arguments
        for arg in node.args.kwonlyargs:
            if '_' in arg.arg:
                dims = extract_shape_dims(arg.arg)
                if dims:
                    check_node = create_check_shape_call(arg.arg, dims)
                    arg_checks.append(check_node)

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


def check_shape(tensor: Any, dim_names: List[str], _dims_: Dict[str, int], var_name: str) -> None:
    """Runtime shape checking function."""
    if not hasattr(tensor, 'shape'):
        return  # Not a tensor, skip

    if len(dim_names) != len(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {var_name}: expected shape {dim_names}, got {tensor.shape}"
        )

    expected_dims = [_dims_.get(dim_name, None) for dim_name in dim_names]
    for expected_dim, actual_dim, dim_name in zip(expected_dims, tensor.shape, dim_names):
        if expected_dim is not None:
            if actual_dim != expected_dim:
                raise AssertionError(
                    f"Shape mismatch for {var_name} dimension {dim_name}: expected {expected_dim}, got {actual_dim}"
                )
        else:
            _dims_[dim_name] = actual_dim

def shapecheck(func):
    """
    Shape checking decorator using AST transformation.

    This decorator modifies the function's AST to inject shape checking
    for both function arguments and variable assignments within the function.

    Variable naming convention:
    - Variables with underscores in their names are checked
    - The suffix after the last underscore indicates the shape
    - Single letters are treated as separate dimensions: 'nk' -> ['n', 'k']

    Args:
        func: Function to decorate

    Returns:
        Decorated function with shape checking

    Examples:
        @shapecheck
        def matrix_multiply(A_nk, B_km):
            result_nm = torch.matmul(A_nk, B_km)
            return result_nm

    """
    tree = ast.parse(inspect.getsource(func))
    tree.body[0].decorator_list.pop(0)
    transformer = ShapeCheckTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    # print(ast.dump(new_tree, indent=4))
    code = compile(new_tree, filename=f"<shapecheck:{func.__name__}>", mode='exec')
    namespace = func.__globals__.copy()
    namespace['check_shape'] = check_shape
    exec(code, namespace)
    return namespace[func.__name__]
