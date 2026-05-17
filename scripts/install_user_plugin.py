#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = hermes_home() / "plugins" / "lossless_context"
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text("from lossless_context.plugin import register\n", encoding="utf-8")
    print(f"Installed Hermes user plugin shim: {target}")
    skill_src = root / "skills" / "lossless-context"
    skill_dst = (hermes_home() / "skills" / "lossless-context").resolve()
    skills_root = (hermes_home() / "skills").resolve()
    if skill_src.exists():
        if not (skill_dst == skills_root / "lossless-context" and skills_root in skill_dst.parents):
            raise RuntimeError(f"Refusing unsafe skill destination: {skill_dst}")
        if skill_dst.exists():
            shutil.rmtree(skill_dst)
        shutil.copytree(skill_src, skill_dst)
        print(f"Installed Hermes skill docs: {skill_dst}")
    print("Next: add plugins.enabled: [lossless_context] to ~/.hermes/config.yaml and restart/new session.")


if __name__ == "__main__":
    main()
