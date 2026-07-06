"""T15a acceptance tests: pyproject.toml finalization + pip-install smoke.

Covers:
- ``pip install -e .`` succeeds in a fresh venv and resolves ``tinyagent`` to
  the flat-layout module at the repo root (no ``src/`` shadowing).
- ``tinyagent.__version__`` is exactly ``"0.1.0"``.
- Stub: ``test_each_example_runs_under_mocked_llm`` is owned by T15b; this
  test is intentionally a no-op marker until that task lands the example
  scripts.

Per plan §13 T15a + T15b and §3 (flat-module layout, no src/).
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tomllib
import venv
from pathlib import Path
from typing import Any, cast

import pytest

# ---------------------------------------------------------------------
# Resolve the repo root (this test file lives at <repo>/tests/test_examples_run.py)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _venv_python(venv_dir: Path) -> str:
    """Return the absolute path of the python interpreter inside *venv_dir*."""
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _is_truthy(name: str) -> bool:
    """Return True if env var *name* is set to a truthy value."""
    val = os.getenv(name)
    return val is not None and val.lower() not in ("", "0", "false", "no")


def _load_pyproject() -> dict[str, Any]:
    """Load and return the parsed ``pyproject.toml`` at the repo root."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------
# Test: pip install -e . smoke (T15a)
# ---------------------------------------------------------------------
def test_pip_install_smoke(tmp_path: Path) -> None:
    """``pip install -e .`` resolves ``tinyagent`` to the repo-root module.

    Builds a fresh venv at ``tmp_path / "venv"``, installs the project in
    editable mode, then imports ``tinyagent`` and asserts:

    1. ``tinyagent.__file__`` lives at the repo root (flat layout, no
       ``src/`` shadowing).
    2. ``tinyagent.__version__`` is exactly ``"0.1.0"``.

    Skips if ``python3 -m venv`` is unavailable on the host or if the
    ``TINYAGENT_SKIP_PIP_INSTALL`` env var is set (CI opt-out).
    """
    if _is_truthy("TINYAGENT_SKIP_PIP_INSTALL"):
        pytest.skip("TINYAGENT_SKIP_PIP_INSTALL is set")

    # Probe venv availability without committing to it yet.
    if not hasattr(venv, "EnvBuilder"):
        pytest.skip("venv module not available on this Python build")

    venv_dir = tmp_path / "venv"
    try:
        builder = venv.EnvBuilder(
            system_site_packages=False,
            clear=True,
            symlinks=(sys.platform != "win32"),
            with_pip=True,
        )
        builder.create(str(venv_dir))
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"venv creation failed on this host: {exc}")

    py_exe = _venv_python(venv_dir)
    if not Path(py_exe).exists():
        pytest.skip(f"venv interpreter not present at {py_exe}")

    # The venv builder only seeds pip; setuptools is required for the PEP 517
    # build backend declared in pyproject.toml. Install it first.
    bootstrap = subprocess.run(  # noqa: S603
        [py_exe, "-m", "pip", "install", "--quiet", "setuptools", "wheel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if bootstrap.returncode != 0:
        pytest.fail(
            "bootstrapping setuptools+wheel in the venv failed:\n"
            f"STDOUT:\n{bootstrap.stdout}\n"
            f"STDERR:\n{bootstrap.stderr}"
        )

    # pip install -e . from the repo root.
    install = subprocess.run(  # noqa: S603
        [py_exe, "-m", "pip", "install", "--quiet", "--no-build-isolation", "-e", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if install.returncode != 0:
        pytest.fail(
            "pip install -e . failed:\n"
            f"STDOUT:\n{install.stdout}\n"
            f"STDERR:\n{install.stderr}"
        )

    # Verify import resolves to the flat-layout module.
    verify = subprocess.run(  # noqa: S603
        [py_exe, "-c", (
            "import sys, tinyagent; "
            "print(tinyagent.__file__); "
            "print(tinyagent.__version__)"
        )],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert verify.returncode == 0, (
        "import tinyagent in fresh venv failed:\n"
        f"STDOUT:\n{verify.stdout}\n"
        f"STDERR:\n{verify.stderr}"
    )

    lines = [ln for ln in verify.stdout.splitlines() if ln]
    assert len(lines) >= 2, f"unexpected verifier output: {verify.stdout!r}"  # noqa: PLR2004
    resolved_file = lines[0]
    resolved_version = lines[1]

    # 1. Module file must be the flat-layout tinyagent.py at the repo root.
    expected = (REPO_ROOT / "tinyagent.py").resolve()
    assert Path(resolved_file).resolve() == expected, (
        f"tinyagent.__file__ resolved to {resolved_file!r}; expected {expected!r}. "
        "This usually means a `src/` layout is shadowing the flat module."
    )

    # 2. Version pin.
    assert resolved_version == "0.1.0", (
        f"tinyagent.__version__ is {resolved_version!r}; expected '0.1.0'"
    )


# ---------------------------------------------------------------------
# Test: flat-layout sanity (in-process — no venv, no pip)
# ---------------------------------------------------------------------
def test_tinyagent_module_is_flat_layout() -> None:
    """``import tinyagent`` in the dev environment resolves to the repo-root file.

    The venv smoke test above exercises a fresh ``pip install -e .``; this
    in-process variant locks the same invariant for the dev workflow that
    the rest of the unit-test suite uses (pytest with ``pythonpath = ["."]``).

    Either way: ``tinyagent.__file__`` must be ``<repo_root>/tinyagent.py``.
    """
    # Force a fresh import in case earlier tests cached an alternate path.
    sys.modules.pop("tinyagent", None)
    tinyagent = importlib.import_module("tinyagent")
    expected = (REPO_ROOT / "tinyagent.py").resolve()
    actual = Path(cast("str", tinyagent.__file__)).resolve()
    assert actual == expected, (
        f"tinyagent.__file__ resolved to {actual!r}; expected {expected!r}. "
        "The flat-layout invariant from plan §3 is broken."
    )


def test_tinyagent_version_is_pinned() -> None:
    """``tinyagent.__version__`` is exactly the locked ``0.1.0``."""
    tinyagent = importlib.import_module("tinyagent")
    assert tinyagent.__version__ == "0.1.0"


def test_no_src_layout_shadowing() -> None:
    """The repo MUST NOT contain a ``src/`` directory or ``src/__init__.py``.

    Per plan §0 C6 (flat layout, round-3 m5 + m7). A ``src/`` tree would
    shadow the repo-root ``tinyagent.py`` after ``pip install -e .`` and
    silently break the venv smoke test.
    """
    src_dir = REPO_ROOT / "src"
    assert not src_dir.exists(), (
        f"{src_dir} exists; the package MUST use the flat layout "
        "(py-modules = ['tinyagent']) without any src/ tree. See plan §0 C6."
    )
    # Also reject any stray __init__.py inside what would be src/ if created.
    for stray in (REPO_ROOT / "src" / "__init__.py",):
        assert not stray.exists(), f"stray package marker at {stray}"


# ---------------------------------------------------------------------
# Test: pyproject.toml metadata pins (T15a — locks §3 acceptance criteria)
# ---------------------------------------------------------------------
def test_pyproject_has_flat_setuptools_config() -> None:
    """pyproject.toml declares the flat-module install.

    The ``[tool.setuptools]`` table MUST contain ``py-modules = ["tinyagent"]``
    and MUST NOT carry a ``package-dir`` mapping or a ``packages.find`` section
    (both would imply a non-flat layout).
    """
    data = _load_pyproject()
    setuptools_cfg = data.get("tool", {}).get("setuptools", {})
    assert "py-modules" in setuptools_cfg, (
        "pyproject.toml [tool.setuptools] is missing 'py-modules = [\"tinyagent\"]'"
    )
    assert setuptools_cfg["py-modules"] == ["tinyagent"], (
        f"py-modules is {setuptools_cfg['py-modules']!r}; expected ['tinyagent']"
    )
    assert "package-dir" not in setuptools_cfg, (
        "[tool.setuptools] package-dir is set; flat layout forbids it. See §0 C6."
    )
    assert "packages" not in setuptools_cfg, (
        "[tool.setuptools] packages is set; flat layout forbids packages.find. See §0 C6."
    )


def test_pyproject_dependencies_pinned() -> None:
    """Runtime dependencies are exactly the pinned set from plan §3."""
    data = _load_pyproject()
    deps = data.get("project", {}).get("dependencies", [])
    expected = {
        "any-llm-sdk>=1.16,<1.20",
        "mcp==1.28.1",
        "opentelemetry-api>=1.27.0",
        "pydantic>=2.5",
        "simpleeval>=1.0",
        "httpx>=0.27",
        "typing-extensions>=4.0",
    }
    assert expected.issubset(set(deps)), (
        f"missing pins: {expected - set(deps)}; "
        f"current deps: {sorted(deps)}"
    )


def test_pyproject_python_and_license() -> None:
    """``requires-python`` is >= 3.11 and the license text is Apache-2.0."""
    data = _load_pyproject()
    project = data.get("project", {})
    assert project.get("requires-python") == ">=3.11", (
        f"requires-python is {project.get('requires-python')!r}; "
        "expected '>=3.11'"
    )
    assert project.get("version") == "0.1.0", (
        f"version is {project.get('version')!r}; expected '0.1.0'"
    )
    license_field = project.get("license", {})
    assert license_field.get("text") == "Apache-2.0", (
        f"license.text is {license_field.get('text')!r}; expected 'Apache-2.0'"
    )


# ---------------------------------------------------------------------
# Stub: owned by T15b (example scripts). Added now as a placeholder so the
# pytest collection has a known test name to wire T15b into.
# ---------------------------------------------------------------------
def test_each_example_runs_under_mocked_llm() -> None:
    """Stub for T15b: smoke-run each ``examples/*.py`` against a mocked LLM.

    Owned by T15b — implementation lands in that task. Marked
    ``pytest.xfail`` here so the test name is reserved and the file shows a
    clear "pending" status in CI rather than silently passing as a no-op.
    """
    pytest.xfail(
        "T15b-owned: example scripts (calculator_mcp_stdio, http_demo, "
        "tracing_otlp) not yet shipped"
    )


# ---------------------------------------------------------------------
# Teardown helper exposed for other tests (e.g. parallel install smoke).
# ---------------------------------------------------------------------
def _rm_venv(venv_dir: Path) -> None:
    """Best-effort cleanup of a test venv directory."""
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
