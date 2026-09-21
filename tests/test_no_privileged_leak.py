import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IDENTIFIERS = {
    "StimulusLabel",
    "GroundTruthStimulus",
    "stage",
    "category_id",
    "ground_truth_label",
    "label",
}

FORBIDDEN_MODULES = {
    "self_fly.environment.stimulus_types",
    "self_fly.experiment.types",
}

TARGET_FILES = [
    *sorted((PROJECT_ROOT / "self_fly" / "agent").glob("*.py")),
    PROJECT_ROOT / "self_fly" / "visualization" / "render.py",
]


def _collect_identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _collect_imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_target_files_exist():
    assert len(TARGET_FILES) >= 2
    for path in TARGET_FILES:
        assert path.exists(), path


def test_agent_and_render_never_reference_privileged_stimulus_info():
    """agent/ decides actions from raw features only, and render.py draws
    every stimulus (including the self-referential one) the same way -- so
    neither may reference the ground-truth label, stage, or category that
    only experiment/ and environment/ are allowed to see."""
    violations = []
    for path in TARGET_FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        bad_names = _collect_identifiers(tree) & FORBIDDEN_IDENTIFIERS
        bad_modules = _collect_imported_modules(tree) & FORBIDDEN_MODULES
        if bad_names or bad_modules:
            violations.append((str(path.relative_to(PROJECT_ROOT)), bad_names, bad_modules))

    assert violations == [], f"privileged stimulus info leaked: {violations}"
