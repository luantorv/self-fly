import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TERMS = ("best", "winner", "ranking", "superior", "gana", "ganador")


_DOCSTRING_CONTAINERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_CONTAINERS) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            ids.add(id(first.value))
    return ids


def _non_docstring_string_literals(tree: ast.AST) -> list[str]:
    docstring_ids = _docstring_node_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_ids
    ]


def test_analysis_module_never_uses_ranking_language_outside_docstrings():
    """Mechanical guardrail for the user's explicit requirement: block 9's
    comparisons must describe differences between conditions, never
    declare a "winner". Checks actual string VALUES (dict keys, output
    text) via the AST, not comments or docstrings discussing the policy
    itself -- this module's own docstrings say "no ranking" using exactly
    these words, which is not a violation."""
    violations = []
    for path in sorted((PROJECT_ROOT / "self_fly" / "analysis").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for literal in _non_docstring_string_literals(tree):
            lowered = literal.lower()
            for term in FORBIDDEN_TERMS:
                if term in lowered:
                    violations.append((str(path.relative_to(PROJECT_ROOT)), literal, term))

    assert violations == [], f"ranking language found in non-docstring string literals: {violations}"
