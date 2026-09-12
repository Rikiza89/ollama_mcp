from __future__ import annotations

from pathlib import Path

import pytest

from ollama_mcp import config
from ollama_mcp.sandbox import SandboxError, rel, resolve


@pytest.fixture()
def cfg(tmp_path: Path) -> config.Config:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    return config.load(tmp_path)


def test_resolves_relative_path(cfg: config.Config) -> None:
    assert resolve(cfg, "pkg/mod.py", must_exist=True).name == "mod.py"


def test_rejects_traversal(cfg: config.Config) -> None:
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve(cfg, "../outside.py")


def test_rejects_absolute_path_outside_workspace(cfg: config.Config, tmp_path: Path) -> None:
    other = tmp_path.parent / "elsewhere.py"
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve(cfg, str(other))


def test_rejects_denied_directory(cfg: config.Config) -> None:
    with pytest.raises(SandboxError, match="denied by sandbox"):
        resolve(cfg, ".git/config")


def test_rejects_denied_filename(cfg: config.Config) -> None:
    with pytest.raises(SandboxError, match="denied by sandbox"):
        resolve(cfg, ".env")


def test_rejects_denied_extension(cfg: config.Config) -> None:
    with pytest.raises(SandboxError, match="denied by sandbox"):
        resolve(cfg, "certs/server.pem")


def test_missing_file_reported(cfg: config.Config) -> None:
    with pytest.raises(SandboxError, match="no such file"):
        resolve(cfg, "pkg/nope.py", must_exist=True)


def test_rel_is_posix(cfg: config.Config) -> None:
    assert rel(cfg, resolve(cfg, "pkg/mod.py")) == "pkg/mod.py"
