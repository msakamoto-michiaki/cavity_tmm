# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, math, json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np

def is_verbose(verbose: bool | None = None) -> bool:
    """Return whether tests should print detailed diagnostics."""
    if verbose is not None:
        return bool(verbose)
    v = os.environ.get("TEST_VERBOSE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")

def summarize_complex(arr, max_items: int = 3) -> str:
    a = np.asarray(arr, dtype=complex).ravel()
    if a.size == 0:
        return "[]"
    head = a[:max_items]
    tail = a[-max_items:] if a.size > max_items else np.array([], dtype=complex)
    def fmt(z: complex) -> str:
        return f"{z.real:+.6g}{z.imag:+.6g}j"
    if a.size <= max_items:
        return "[" + ", ".join(fmt(z) for z in head) + "]"
    return "[" + ", ".join(fmt(z) for z in head) + ", ..., " + ", ".join(fmt(z) for z in tail) + "]"

def summarize_real(arr, max_items: int = 5) -> str:
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return "[]"
    head = a[:max_items]
    tail = a[-max_items:] if a.size > max_items else np.array([], dtype=float)
    def fmt(x: float) -> str:
        return f"{x:.6g}"
    if a.size <= max_items:
        return "[" + ", ".join(fmt(x) for x in head) + "]"
    return "[" + ", ".join(fmt(x) for x in head) + ", ..., " + ", ".join(fmt(x) for x in tail) + "]"

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_import_paths() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pm = root / "PyMoosh-stable"
    if pm.exists() and str(pm) not in sys.path:
        sys.path.insert(0, str(pm))

def require_pymoosh():
    ensure_import_paths()
    try:
        import PyMoosh  # type: ignore
        from PyMoosh.classes import Structure  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyMoosh-stable が見つからない/ import できません。"
            " ver17 ルートに PyMoosh-stable/ があることを確認してください。"
        ) from e
    return PyMoosh, Structure

def max_abs(x) -> float:
    x = np.asarray(x)
    return float(np.nanmax(np.abs(x)))

def max_rel(x, ref, eps: float = 1e-30) -> float:
    x = np.asarray(x)
    ref = np.asarray(ref)
    den = np.maximum(1.0, np.abs(ref))
    return float(np.nanmax(np.abs(x-ref) / (den + eps)))

def assert_close_complex(name: str, x, ref, atol: float, rtol: float) -> list[str]:
    xa = np.asarray(x, dtype=complex)
    ra = np.asarray(ref, dtype=complex)
    ma = max_abs(xa - ra)
    mr = max_rel(xa, ra)
    fails = []
    if not (ma <= atol or mr <= rtol):
        fails.append(f"{name}: max|Δ|={ma:.3e} (atol={atol}), max rel={mr:.3e} (rtol={rtol})")
    return fails

def check_close_complex(
    name: str,
    x,
    ref,
    atol: float,
    rtol: float,
    verbose: bool | None = None,
    context: str | None = None,
) -> list[str]:
    """Compare complex arrays and (optionally) print detailed diagnostics.

    PASS criterion:
      max|Δ| <= atol OR max_rel <= rtol
    where max_rel = max(|x-ref|/max(1,|ref|)).
    """
    xa = np.asarray(x, dtype=complex)
    ra = np.asarray(ref, dtype=complex)
    ma = max_abs(xa - ra)
    mr = max_rel(xa, ra)
    ok = (ma <= atol) or (mr <= rtol)

    if is_verbose(verbose):
        print(f"[CHECK] {name}")
        if context:
            print(f"  context: {context}")
        print(f"  criterion: PASS if max|Δ|<= {atol:.3e} OR max_rel<= {rtol:.3e}")
        print(f"  result   : max|Δ|={ma:.3e}, max_rel={mr:.3e} => {'PASS' if ok else 'FAIL'}")
        print(f"  x   sample: {summarize_complex(xa)}")
        print(f"  ref sample: {summarize_complex(ra)}")
        print()

    fails: list[str] = []
    if not ok:
        fails.append(f"{name}: max|Δ|={ma:.3e} (atol={atol}), max rel={mr:.3e} (rtol={rtol})")
    return fails

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def run_subprocess(cmd: list[str], cwd: Path | None = None, timeout_s: int = 300) -> tuple[int,str,str]:
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err + "\n[TIMEOUT]"
    return p.returncode, out, err

def temp_dir(prefix: str = "test_out_") -> Path:
    root = repo_root() / "test" / "_out"
    root.mkdir(parents=True, exist_ok=True)
    # unique subdir
    import time
    p = root / f"{prefix}{int(time.time()*1000)}"
    p.mkdir(parents=True, exist_ok=True)
    return p
