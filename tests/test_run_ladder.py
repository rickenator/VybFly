"""Engine auto-selection in scripts/cpu/run_ladder.sh (no Vyb needed: VYB is stubbed with echo)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpu" / "run_ladder.sh"


def _run(tmp_path: Path, n_edges: int, **env: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    with open(bin_dir / "csr.out.indices.i32", "wb") as f:
        f.truncate(4 * n_edges)  # sparse file; only the size matters
    e = dict(os.environ, VYB="echo", VYBFLY_LADDER_DIR=str(tmp_path), VYBFLY_ARCH="x86_64")
    e.pop("VYBFLY_ENGINE", None)
    e.pop("VYBFLY_DESC_MIN_EDGES", None)
    e.update(env)
    return subprocess.run(["bash", str(SCRIPT), "--module-path", "std"], env=e,
                          capture_output=True, text=True)


@pytest.mark.parametrize("n_edges,engine", [
    (2_700_513, "ladder_prop"),    # 1x
    (5_400_000, "ladder_prop"),    # 2x
    (13_900_000, "ladder_desc"),   # 5x
    (27_842_102, "ladder_desc"),   # 10x
])
def test_auto_picks_by_edge_count(tmp_path, n_edges, engine):
    r = _run(tmp_path, n_edges)
    assert r.returncode == 0, r.stderr
    out = r.stdout.split()
    assert out[:2] == ["--module-path", "std"]
    assert out[-1].endswith(f"src/vyb/{engine}.vyb")
    assert Path(out[-1]).exists()


def test_override_and_threshold(tmp_path):
    r = _run(tmp_path, 2_700_513, VYBFLY_ENGINE="ladder_seg")
    assert r.returncode == 0 and r.stdout.split()[-1].endswith("ladder_seg.vyb")
    r = _run(tmp_path, 2_700_513, VYBFLY_DESC_MIN_EDGES="1000")
    assert r.stdout.split()[-1].endswith("ladder_desc.vyb")
    r = _run(tmp_path, 10, VYBFLY_ENGINE="nope")
    assert r.returncode == 2


def test_missing_csr(tmp_path):
    e = dict(os.environ, VYB="echo", VYBFLY_LADDER_DIR=str(tmp_path / "none"))
    e.pop("VYBFLY_ENGINE", None)
    r = subprocess.run(["bash", str(SCRIPT)], env=e, capture_output=True, text=True)
    assert r.returncode == 2


@pytest.mark.parametrize("name", ["gpu", "ladder_gpu"])
def test_gpu_engine(tmp_path, name):
    env_script = (tmp_path / "envecho.sh")
    env_script.write_text('#!/usr/bin/env bash\necho "PTX=$VYBFLY_GPU_PTX" "$@"\n')
    env_script.chmod(0o755)
    r = _run(tmp_path, 10, VYBFLY_ENGINE=name, VYB=str(env_script))
    assert r.returncode == 0, r.stderr
    out = r.stdout.split()
    assert out[-1].endswith("src/vyb/ladder_gpu.vyb") and Path(out[-1]).exists()
    ptx = Path(out[0].removeprefix("PTX="))
    assert ptx.is_absolute() and ptx.name == "ladder_gpu_kernel.ptx" and ptx.exists()
    r = _run(tmp_path, 10, VYBFLY_ENGINE=name, VYB=str(env_script), VYBFLY_GPU_PTX="/x/k.ptx")
    assert r.stdout.split()[0] == "PTX=/x/k.ptx"


@pytest.mark.parametrize("arch", ["aarch64", "arm64"])
def test_arm_always_prop(tmp_path, arch):
    r = _run(tmp_path, 27_842_102, VYBFLY_ARCH=arch)
    assert r.returncode == 0 and r.stdout.split()[-1].endswith("ladder_prop.vyb")
    r = _run(tmp_path, 27_842_102, VYBFLY_ARCH=arch, VYBFLY_DESC_MIN_EDGES="1000")
    assert r.stdout.split()[-1].endswith("ladder_desc.vyb")
