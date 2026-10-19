"""The provider helpers that need no extra must not load one."""
from __future__ import annotations

import subprocess
import sys

import pytest

PURE_MODULES = (
    "protocore.providers.chain",
    "protocore.providers.context_window",
    "protocore.providers.spend",
    "protocore.providers.text_tool_calls",
    # Only the HTTP transport needs httpx, and it is imported when asked for.
    "protocore.mcp",
)


@pytest.mark.parametrize("module", PURE_MODULES)
def test_a_pure_module_imports_without_httpx(module: str) -> None:
    # A fresh interpreter: this one has long since imported httpx for the
    # adapter's own tests.
    code = f"import sys, {module}; sys.exit(1 if 'httpx' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or f"importing {module} loaded httpx"
