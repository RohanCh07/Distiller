"""Write docs/cli.md from the CLI's own --help output.  --check: fail if it is stale."""

import sys
from pathlib import Path

from typer.main import get_command

from distiller.cli import app

OUT = Path(__file__).parents[1] / "docs" / "cli.md"


def walk(cmd, path: list[str]):
    yield path, cmd
    if hasattr(cmd, "commands"):  # a group
        for name in sorted(cmd.commands):
            yield from walk(cmd.commands[name], path + [name])


def _is_arg(p) -> bool:
    return p.param_type_name == "argument"


def _param_row(p) -> str:
    if _is_arg(p):
        name = f"`{p.name.upper()}`"
        kind = "argument" + ("" if p.required else ", optional")
    else:
        name = " / ".join(f"`{o}`" for o in p.opts + p.secondary_opts)
        kind = "flag" if getattr(p, "is_flag", False) else "option"
    default = (
        p.default if p.default not in (None, (), [], False) and not callable(p.default) else ""
    )
    help_ = (getattr(p, "help", None) or "").replace("|", "\\|")
    return f"| {name} | {kind} | {help_} | {f'`{default}`' if default != '' else ''} |"


def render() -> str:
    root = get_command(app)
    parts = [
        "# CLI reference",
        "",
        "Generated from the code by `python scripts/gen_cli_docs.py` (CI checks it is current).",
        "",
    ]
    for path, cmd in walk(root, ["distiller"]):
        if hasattr(cmd, "commands"):
            continue
        params = [p for p in cmd.params if p.name not in ("help",)]
        usage = " ".join(path) + "".join(f" {p.name.upper()}" if _is_arg(p) else "" for p in params)
        parts += [
            f"## {' '.join(path[1:])}",
            "",
            (cmd.help or "").strip(),
            "",
            f"`{usage} [OPTIONS]`",
            "",
        ]
        if params:
            parts += ["| Name | Kind | Description | Default |", "|---|---|---|---|"]
            parts += [_param_row(p) for p in params]
            parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    content = render()
    if "--check" in sys.argv:
        if OUT.read_text(encoding="utf-8") != content:
            sys.exit("docs/cli.md is stale: run python scripts/gen_cli_docs.py")
        print("docs/cli.md is up to date")
    else:
        OUT.write_text(content, encoding="utf-8")
        print(f"wrote {OUT}")
