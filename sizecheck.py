import ast
import inspect
import textwrap
from typing import Dict, List, Any, Union, Tuple

def extract_shape_dims(name: str) -> List[str]:
    """Extract shape dimensions from annotated variable name."""
    parts = name.split('_')
    if len(parts) < 2:
        return []
    shape_part = parts[-1]
    if shape_part.isalpha() and shape_part.isupper():
        return list(shape_part)
    else:
        return []

def create_check_shape_call(var_name: str, dims: List[str], lineno: int = 1) -> ast.stmt:
    """Create a call to the shape checking function."""
    dims_list = ast.List(elts=[ast.Constant(value=dim) for dim in dims], ctx=ast.Load())
    dims_list.lineno = lineno
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
    # Set line numbers to match original source
    expr.lineno = lineno
    expr.col_offset = 0
    expr.value.lineno = lineno
    expr.value.col_offset = 0
    return expr

class SizeCheckTransformer(ast.NodeTransformer):
    """AST transformer that injects shape checking code."""

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
        node = self.generic_visit(node)
        new_nodes: List[ast.stmt] = [node]
        for target in node.targets:
            names = self.extract_names_from_target(target)
            for name in names:
                dims = extract_shape_dims(name)
                if dims:
                    check_node = create_check_shape_call(name, dims, node.lineno)
                    new_nodes.append(check_node)
        return new_nodes if len(new_nodes) > 1 else node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Union[ast.AnnAssign, List[ast.stmt]]:
        """Handle annotated assignments."""
        node = self.generic_visit(node)
        names = self.extract_names_from_target(node.target)
        check_nodes: List[ast.stmt] = []
        for name in names:
            dims = extract_shape_dims(name)
            if dims:
                check_node = create_check_shape_call(name, dims, node.lineno)
                check_nodes.append(check_node)
        return [node] + check_nodes if check_nodes else node

    def visit_AugAssign(self, node: ast.AugAssign) -> Union[ast.AugAssign, List[ast.stmt]]:
        """Handle augmented assignments (+=, -=, etc.)."""
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            dims = extract_shape_dims(node.target.id)
            if dims:
                check_node = create_check_shape_call(node.target.id, dims, node.lineno)
                result: List[ast.stmt] = [node, check_node]
                return result
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definition to add argument checks."""
        # Collect shape-annotated arguments
        dims_assign = ast.Assign(targets=[ast.Name(id='_dims_', ctx=ast.Store())], value=ast.Dict())
        dims_assign.lineno = node.lineno
        dims_assign.col_offset = 0
        arg_checks: List[ast.stmt] = [dims_assign]

        # Check regular arguments
        for arg in node.args.args:
            if '_' in arg.arg:
                dims = extract_shape_dims(arg.arg)
                if dims:
                    check_node = create_check_shape_call(arg.arg, dims, node.lineno)
                    arg_checks.append(check_node)

        # Check keyword-only arguments
        for arg in node.args.kwonlyargs:
            if '_' in arg.arg:
                dims = extract_shape_dims(arg.arg)
                if dims:
                    check_node = create_check_shape_call(arg.arg, dims, node.lineno)
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


def check_shape(tensor: Any, dim_names: List[str], _dims_: Dict[str, Tuple[int, str]], var_name: str) -> None:
    """Runtime shape checking function."""
    if not hasattr(tensor, 'shape'):
        return  # Not a tensor, skip

    if len(dim_names) != len(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {var_name}: expected shape {dim_names}, got {tensor.shape}"
        )

    expected_dims = [_dims_.get(dim_name, None) for dim_name in dim_names]
    for expected_dim_info, actual_dim, dim_name in zip(expected_dims, tensor.shape, dim_names):
        if expected_dim_info is not None:
            expected_dim, original_var = expected_dim_info
            if actual_dim != expected_dim:
                raise AssertionError(
                    f"Shape mismatch for {var_name} dimension {dim_name}: expected {expected_dim} (from {original_var}), got {actual_dim}"
                )
        else:
            _dims_[dim_name] = (actual_dim, var_name)

def sizecheck(func):
    """
    Shape checking decorator using AST transformation.

    This decorator modifies the function's AST to inject shape checking
    for both function arguments and variable assignments within the function.

    Variable naming convention:
    - Variables with underscores in their names are checked
    - The suffix after the last underscore indicates the shape
    - Single capital letters are treated as separate dimensions: 'NK' -> ['N', 'K']

    Args:
        func: Function to decorate

    Returns:
        Decorated function with shape checking

    Examples:
        @sizecheck
        def matrix_multiply(A_NK, B_KM):
            result_NM = torch.matmul(A_NK, B_KM)
            return result_NM

    """
    source = textwrap.dedent(inspect.getsource(func))
    original_filename = inspect.getfile(func)
    original_lineno = inspect.getsourcelines(func)[1]
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.FunctionDef):
        tree.body[0].decorator_list.pop(0)
    # Adjust line numbers in the parsed tree
    ast.increment_lineno(tree, original_lineno - 1)
    transformer = SizeCheckTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    # print(ast.dump(new_tree, indent=4))
    code = compile(new_tree, filename=original_filename, mode='exec')
    namespace = func.__globals__.copy()
    namespace['check_shape'] = check_shape
    exec(code, namespace)
    return namespace[func.__name__]
