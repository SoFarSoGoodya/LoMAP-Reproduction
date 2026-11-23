"""Build the original D4RL PointMaze XMLs from the pinned LoMAP checkout."""

import ast
import io
import runpy
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / ".upstream/lomap"
REVISION = "8ccb052481484352affaada26a0895102aef3e52"
MAZES = {"umaze": "U_MAZE", "medium": "MEDIUM_MAZE", "large": "LARGE_MAZE"}


def main(mazes=("umaze", "medium")) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"LoMAP upstream is at {revision}, expected {REVISION}")
    source = UPSTREAM / "d4rl/pointmaze/maze_model.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    # Load only the source's XML builder, maps and tile constants. Legacy Gym
    # and the D4RL registration side effects are irrelevant to XML generation.
    functions = {"parse_maze", "point_maze"}
    constants = {"WALL", "EMPTY", "GOAL", *MAZES.values()}
    definitions = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            definitions.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            definitions.append(node)

    builder = runpy.run_path(str(source.with_name("dynamic_mjc.py")))["MJCModel"]
    scope = {"np": np, "MJCModel": builder}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), scope)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    for name in mazes:
        output = io.StringIO()
        scope["point_maze"](scope[MAZES[name]]).root.write(output)
        (assets / f"maze2d-{name}-v1.xml").write_text(output.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
