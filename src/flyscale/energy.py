"""Energy instrumentation and the two-track energy model (PROJECT-VYBFLY.md §17, §18, M4).

§17 requires **two separate** energy measurements that must never be conflated:

(A) **biological-equivalent energy** -- ``P_bio(N) = P0 * N / N0``, anchored to published
    metabolic measurements of real nervous tissue (see :data:`ANCHORS`). Every constant
    here carries its citation plus explicit provenance flags, because the mammalian
    linear-per-neuron relationship is *not* a Drosophila measurement (§17).
(B) **actual hardware energy** -- joules integrated from real power sampling on this
    machine: GPU power via ``nvidia-smi``, CPU package / DRAM power via Linux RAPL
    (``/sys/class/powercap/intel-rapl*/energy_uj``) when readable, wall clock always.
    A device that cannot be read is reported as *unavailable with the reason* -- it is
    never estimated or substituted.

This module invents no measurements. :func:`probe_devices` records what this box can
and cannot measure, and :meth:`PowerSampler.report` returns the raw sample series plus
per-device and per-phase integrals so a claim of "X joules" can always be traced back
to timestamps and watt readings.

Typical use::

    s = PowerSampler(interval_s=0.2)
    s.start()
    s.mark("idle")
    time.sleep(10)
    s.mark("workload")
    do_work()
    s.stop()
    report = s.report(baseline_phase="idle")
"""
from __future__ import annotations

import glob
import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "FLYWIRE_V783_NEURONS",
    "ANCHORS",
    "BiologicalEnergyModel",
    "MetabolicAnchor",
    "PowerSampler",
    "integrate",
    "probe_devices",
    "write_json",
]

#: Neuron count of the canonical dataset in ``data/processed/canonical_v783``
#: (``meta.json`` -> ``counts.n_neurons``; FlyWire FAFB v783).
FLYWIRE_V783_NEURONS = 139_255

GPU_QUERY_FIELDS = ("index", "name", "power.draw", "power.limit", "utilization.gpu",
                    "memory.used", "temperature.gpu", "clocks.sm")

#: NVML constants (nvidia-ml.h)
NVML_SUCCESS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0


class NvmlBackend:
    """In-process GPU power reader via NVML (``libnvidia-ml``) through ctypes.

    Preferred over spawning ``nvidia-smi`` per sample: on this box (RTX 3090, driver
    580.173.02) repeated ``nvidia-smi --query-gpu=power.draw`` calls made from a background
    sampling thread returned a *stale* idle value for the whole run even at ~97% GPU
    utilisation, while NVML calls from the same thread tracked the load correctly
    (22 W idle -> 155 W loaded). See results/energy/README.md.
    """

    def __init__(self, lib_names: Sequence[str] = ("libnvidia-ml.so.1", "libnvidia-ml.so")):
        self.lib = None
        self.handles: dict[int, Any] = {}
        self.device_names: dict[int, str] = {}
        self.reason: str | None = None
        self.lib_name: str | None = None
        for name in lib_names:
            try:
                self.lib = ctypes.CDLL(name)
                self.lib_name = name
                break
            except OSError as exc:
                self.reason = f"{name}: {exc}"
        if self.lib is None:
            return
        init = self._call("nvmlInit_v2") or self._call("nvmlInit")
        if init is None:
            self.reason = "libnvidia-ml has no nvmlInit/nvmlInit_v2 symbol"
            self.lib = None
            return
        rc = init()
        if rc != NVML_SUCCESS:
            self.reason = f"nvmlInit_v2 returned {rc}"
            self.lib = None
            return
        count = ctypes.c_uint()
        count_fn = self._call("nvmlDeviceGetCount_v2") or self._call("nvmlDeviceGetCount")
        if count_fn is None or count_fn(ctypes.byref(count)) != NVML_SUCCESS:
            self.reason = "nvmlDeviceGetCount failed"
            self.lib = None
            return
        handle_fn = (self._call("nvmlDeviceGetHandleByIndex_v2")
                     or self._call("nvmlDeviceGetHandleByIndex"))
        for idx in range(int(count.value)):
            handle = ctypes.c_void_p()
            if handle_fn is None or handle_fn(idx, ctypes.byref(handle)) != NVML_SUCCESS:
                continue
            self.handles[idx] = handle
            buf = ctypes.create_string_buffer(96)
            if self.lib.nvmlDeviceGetName(handle, buf, 96) == NVML_SUCCESS:
                self.device_names[idx] = buf.value.decode(errors="replace")

    def _call(self, name: str):
        return getattr(self.lib, name, None)

    @property
    def available(self) -> bool:
        return bool(self.handles)

    def read(self, idx: int) -> dict[str, float | None]:
        h = self.handles[idx]
        mw = ctypes.c_uint()
        out: dict[str, float | None] = {"power_w": None, "power_limit_w": None,
                                        "util_pct": None, "sm_mhz": None, "mem_mhz": None,
                                        "temp_c": None}
        if self.lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == NVML_SUCCESS:
            out["power_w"] = mw.value / 1000.0
        lim = ctypes.c_uint()
        if self.lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(lim)) == NVML_SUCCESS:
            out["power_limit_w"] = lim.value / 1000.0
        util = _NvmlUtilization()
        if self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == NVML_SUCCESS:
            out["util_pct"] = float(util.gpu)
        for key, clock in (("sm_mhz", NVML_CLOCK_SM), ("mem_mhz", NVML_CLOCK_MEM)):
            c = ctypes.c_uint()
            if self.lib.nvmlDeviceGetClockInfo(h, clock, ctypes.byref(c)) == NVML_SUCCESS:
                out[key] = float(c.value)
        t = ctypes.c_uint()
        if self.lib.nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU, ctypes.byref(t)) == \
                NVML_SUCCESS:
            out["temp_c"] = float(t.value)
        return out


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd: Sequence[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, creating parents. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, p)
    return p


def integrate(times: Sequence[float], values: Sequence[float | None],
              t0: float | None = None, t1: float | None = None) -> tuple[float, int]:
    """Trapezoidal ∫ P dt (joules for watts) over consecutive valid samples.

    A sample that is ``None`` (device missing that tick) breaks the series; no
    interpolation is invented across the gap. ``t0``/``t1`` clip the window, splitting
    the straddling interval by linear interpolation. Returns ``(joules, n_intervals)``.
    """
    lo = -float("inf") if t0 is None else float(t0)
    hi = float("inf") if t1 is None else float(t1)
    joules = 0.0
    n = 0
    for i in range(len(times) - 1):
        va, vb = values[i], values[i + 1]
        if va is None or vb is None:
            continue
        ta, tb = times[i], times[i + 1]
        if tb <= lo or ta >= hi:
            continue
        a, b = max(ta, lo), min(tb, hi)
        if b <= a or tb <= ta:
            continue
        fa, fb = (a - ta) / (tb - ta), (b - ta) / (tb - ta)
        ya = va + (vb - va) * fa
        yb = va + (vb - va) * fb
        joules += 0.5 * (ya + yb) * (b - a)
        n += 1
    return joules, n


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    s = sorted(vals)
    mid = len(s) // 2
    median = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    return {"n": len(vals), "min": min(vals), "max": max(vals),
            "mean": sum(vals) / len(vals), "median": median}


# --------------------------------------------------------------------------------------
# device probing -- what can this machine actually measure?
# --------------------------------------------------------------------------------------

def _probe_nvidia_smi(probe_all: bool = True) -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "method": "nvidia-smi --query-gpu=power.draw",
                "reason": "nvidia-smi not found on PATH", "devices": []}
    try:
        proc = _run([exe, "--query-gpu=" + ",".join(GPU_QUERY_FIELDS[:4]) + ("," + GPU_QUERY_FIELDS[6] if probe_all else ""),
                     "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "method": "nvidia-smi --query-gpu=power.draw",
                "reason": f"nvidia-smi failed: {exc!r}", "devices": []}
    if proc.returncode != 0:
        return {"available": False, "method": "nvidia-smi --query-gpu=power.draw",
                "reason": f"nvidia-smi exit {proc.returncode}: {proc.stderr.strip()[:200]}",
                "devices": []}
    devices, reason = [], None
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[2] in ("", "[N/A]", "N/A"):
            reason = reason or f"power.draw unavailable on gpu {parts[0] if parts else '?'}"
            continue
        devices.append({"device": f"gpu:{parts[0]}", "name": parts[1]})
    return {"available": bool(devices), "method": "nvidia-smi --query-gpu=power.draw",
            "reason": reason, "devices": devices,
            "caveat": ("spawning nvidia-smi from the sampler thread returned stale idle "
                       "readings on this box (driver-dependent); NVML is preferred")}


def probe_gpu() -> dict[str, Any]:
    """Whether GPU power can be sampled, with which backend, and for which GPUs."""
    nvml = NvmlBackend()
    nvml_info: dict[str, Any] = {
        "available": nvml.available,
        "method": f"NVML nvmlDeviceGetPowerUsage via ctypes ({nvml.lib_name})",
        "reason": nvml.reason if not nvml.available else None,
        "devices": [{"device": f"gpu:{i}", "name": nvml.device_names.get(i, "")}
                    for i in sorted(nvml.handles)],
    }
    smi = _probe_nvidia_smi()
    used = "nvml" if nvml_info["available"] else ("nvidia-smi" if smi["available"] else None)
    return {
        "available": used is not None,
        "used_backend": used,
        "method": nvml_info["method"] if used == "nvml" else smi["method"],
        "devices": nvml_info["devices"] if used == "nvml" else smi["devices"],
        "reason": None if used else f"nvml: {nvml_info['reason']}; nvidia-smi: {smi['reason']}",
        "nvml_available": nvml_info["available"],
        "nvml": nvml_info,
        "nvidia_smi": smi,
        "backend_selection_note": (
            "NVML reads the same sensor as nvidia-smi --query-gpu=power.draw but in-process; "
            "on this box nvidia-smi polling from the sampler thread went stale under load, so "
            "NVML is used by default and nvidia-smi is kept as an independent cross-check."),
    }


def probe_rapl(root: str | Path = "/sys/class/powercap") -> dict[str, Any]:
    """Whether RAPL CPU package / DRAM energy counters are readable by this user."""
    domains, reasons = [], []
    for name_path in sorted(glob.glob(os.path.join(str(root), "intel-rapl*/name"))):
        d = Path(name_path).parent
        name = _read_text(name_path)
        energy = d / "energy_uj"
        readable, reason = True, None
        try:
            int(energy.read_text().strip())
        except OSError as exc:
            readable = False
            try:
                st = energy.stat()
                mode = oct(st.st_mode & 0o7777)
                reason = (f"{energy} unreadable for uid {os.getuid()}: {type(exc).__name__}: "
                          f"{exc.strerror}; mode {mode} owner uid {st.st_uid}")
            except OSError:
                reason = f"{energy} unreadable: {exc!r}"
            reasons.append(reason)
        domains.append({"device": f"cpu:{name or d.name}", "sysfs": str(energy),
                        "readable": readable, "reason": reason,
                        "max_energy_range_uj": _read_text(d / "max_energy_range_uj")})
    available = any(x["readable"] for x in domains)
    if not domains:
        reasons.append(f"no intel-rapl* energy_uj domains under {root}")
    return {"available": available,
            "method": "Linux powercap/RAPL: delta of energy_uj counters over wall time",
            "reason": None if available else "; ".join(reasons) or "no RAPL domains found",
            "domains": domains}


def probe_wall(root_hwmon: str = "/sys/class/hwmon") -> dict[str, Any]:
    """Whether *system* power draw (wall / PSU) can be read on this box."""
    hwmon = sorted(glob.glob(os.path.join(root_hwmon, "hwmon*/power*_input")))
    supply = sorted(glob.glob("/sys/class/power_supply/*/power_now"))
    tools = {t: shutil.which(t) for t in ("powerstat", "turbostat")}
    available = bool(hwmon or supply)
    if available:
        reason = None
    else:
        reason = (
            "no PSU/wall sensor exposed: no /sys/class/hwmon/hwmon*/power*_input read-only "
            "power sensor and no /sys/class/power_supply/*/power_now on this desktop board; "
            "no external meter/PDU is attached to this harness. "
            f"powerstat {'present' if tools['powerstat'] else 'absent'}, "
            f"turbostat {'present but needs root (CAP_SYS_RAWIO)' if tools['turbostat'] else 'absent'}"
            " and `sudo -n true` fails (password required), so neither can be used."
        )
    return {"available": available, "method": "hwmon power*_input / power_supply power_now",
            "reason": reason, "hwmon_power_inputs": hwmon, "power_supply_power_now": supply,
            "tools": tools}


def probe_devices(rapl_root: str | Path = "/sys/class/powercap") -> dict[str, Any]:
    """Full device-availability record for this machine (the harness's own audit trail)."""
    gpu = probe_gpu()
    rapl = probe_rapl(rapl_root)
    wall = probe_wall()
    return {
        "probed_utc": _utcnow(),
        "host": os.uname().nodename,
        "gpu": gpu,
        "cpu_rapl": rapl,
        "wall": wall,
        "available_devices": ([d["device"] for d in gpu["devices"]] if gpu["available"] else [])
                             + ([d["device"] for d in rapl["domains"] if d["readable"]] if rapl["available"] else [])
                             + (["wall:ac"] if wall["available"] else []),
        "unavailable_devices": {k: v["reason"] for k, v in
                                (("gpu", gpu), ("cpu_rapl", rapl), ("wall", wall))
                                if not v["available"]},
    }


# --------------------------------------------------------------------------------------
# hardware sampler (track B)
# --------------------------------------------------------------------------------------

class PowerSampler:
    """Background power sampler: GPU (NVML in-process, nvidia-smi fallback) + CPU/DRAM (RAPL)
    + wall clock.

    Fixed-interval sampling thread; every tick appends one row to :attr:`samples` with a
    monotonic timestamp, the current phase label and one column per readable device.
    Unreadable devices simply have no columns -- the gap is preserved, never filled.
    """

    def __init__(self, interval_s: float = 0.2, gpu: bool = True, rapl: bool = True,
                 rapl_root: str | Path = "/sys/class/powercap", notes: Iterable[str] = ()):
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = float(interval_s)
        self.rapl_root = str(rapl_root)
        self.notes: list[str] = list(notes)
        self.probe = probe_devices(self.rapl_root) if (gpu or rapl) else {}
        self.samples: list[dict[str, Any]] = []
        self.marks: list[dict[str, Any]] = []
        self.gpu_devices: list[dict[str, Any]] = (
            self.probe.get("gpu", {}).get("devices", []) if gpu else [])
        self.rapl_domains: list[dict[str, Any]] = [
            d for d in (self.probe.get("cpu_rapl", {}).get("domains", []) if rapl else [])
            if d.get("readable")]
        self._rapl_prev: dict[str, tuple[float, int]] = {}
        self._phase = "unmarked"
        self._t0 = 0.0
        self._t_end: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._nvidia_smi = shutil.which("nvidia-smi")
        self.gpu_backend = self.probe.get("gpu", {}).get("used_backend") if self.probe else None
        self._nvml: NvmlBackend | None = None
        if gpu and self.gpu_backend == "nvml":
            self._nvml = NvmlBackend()
            if not self._nvml.available:
                self._nvml = None
                self.gpu_backend = "nvidia-smi" if self._nvidia_smi else None
        if gpu and self.gpu_backend == "nvidia-smi":
            self.notes.append(
                "GPU power is read by spawning nvidia-smi per sample; on this box that path "
                "has been observed to return stale idle values when polled from the sampler "
                "thread, so NVML is preferred when available.")
        self.device_columns = [d["device"] for d in self.gpu_devices] + \
                              [d["device"] for d in self.rapl_domains]

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> "PowerSampler":
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._t0 = time.perf_counter()
        self._stop.clear()
        self.samples = []
        self.marks = [{"label": self._phase, "t_rel_s": 0.0, "wall_utc": _utcnow()}]
        self._thread = threading.Thread(target=self._loop, name="power-sampler", daemon=True)
        self._thread.start()
        return self

    def mark(self, label: str) -> None:
        """Open a new phase: later samples are labeled ``label``."""
        self._phase = label
        t = time.perf_counter() - self._t0
        self.marks.append({"label": label, "t_rel_s": round(t, 6), "wall_utc": _utcnow()})

    def stop(self) -> "PowerSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 4 * self.interval_s))
            self._thread = None
        self._t_end = time.perf_counter() - self._t0
        self.marks.append({"label": "__end__", "t_rel_s": round(self._t_end, 6),
                           "wall_utc": _utcnow()})
        return self

    # -- one tick ----------------------------------------------------------------------
    def _sample_row(self) -> dict[str, Any]:
        t = time.perf_counter() - self._t0
        row: dict[str, Any] = {"t_rel_s": round(t, 6), "wall_utc": _utcnow(),
                               "phase": self._phase}
        for gpu_row in self._gpu_read():
            row[gpu_row["device"]] = gpu_row["power_w"]
            row[gpu_row["device"] + ":util_pct"] = gpu_row["util_pct"]
        for dev, watts in self._rapl_read(t).items():
            row[dev] = watts
        return row

    def _gpu_read(self) -> list[dict[str, Any]]:
        if not self.gpu_devices:
            return []
        if self._nvml is not None:
            out = []
            for idx, dev in enumerate(self.gpu_devices):
                r = self._nvml.read(idx)
                out.append({"device": dev["device"], "name": dev["name"], **r})
            return out
        if self._nvidia_smi is None:
            return []
        try:
            proc = _run([self._nvidia_smi, "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
                         "--format=csv,noheader,nounits"])
        except (OSError, subprocess.SubprocessError):
            return []
        if proc.returncode != 0:
            return []
        out = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(GPU_QUERY_FIELDS):
                continue

            def _f(x: str) -> float | None:
                try:
                    return float(x)
                except ValueError:
                    return None

            out.append({"device": f"gpu:{parts[0]}", "name": parts[1],
                        "power_w": _f(parts[2]), "power_limit_w": _f(parts[3]),
                        "util_pct": _f(parts[4]), "memory_used_mib": _f(parts[5]),
                        "temp_c": _f(parts[6]), "sm_mhz": _f(parts[7])})
        return out

    def _rapl_read(self, t: float) -> dict[str, float | None]:
        """RAPL power = delta(energy_uj) / delta(t) / 1e6, with counter-wrap handling."""
        out: dict[str, float | None] = {}
        for dom in self.rapl_domains:
            dev, path = dom["device"], Path(dom["sysfs"])
            txt = _read_text(path)
            if txt is None:
                out[dev] = None
                continue
            try:
                e = int(txt)
            except ValueError:
                out[dev] = None
                continue
            prev = self._rapl_prev.get(dev)
            self._rapl_prev[dev] = (t, e)
            if prev is None:
                out[dev] = None
                continue
            t_prev, e_prev = prev
            dt = t - t_prev
            if dt <= 0:
                out[dev] = None
                continue
            de = e - e_prev
            if de < 0:  # counter wrapped (or was reset); reconstruct against the range
                rng = int(dom["max_energy_range_uj"] or 0)
                if rng and -de < rng:
                    de = de + rng
                else:
                    out[dev] = None
                    continue
            out[dev] = de / dt / 1e6
        return out

    def _loop(self) -> None:
        next_t = self._t0
        while not self._stop.is_set():
            self.samples.append(self._sample_row())
            next_t += self.interval_s
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                self._stop.wait(sleep_for)
            else:
                next_t = time.perf_counter()  # fell behind: resync instead of bursting

    # -- reporting ---------------------------------------------------------------------
    def _phase_windows(self) -> list[tuple[str, float, float]]:
        label, start = self.marks[0]["label"], self.marks[0]["t_rel_s"]
        out: list[tuple[str, float, float]] = []
        end_all = self._t_end if self._t_end is not None else (
            self.samples[-1]["t_rel_s"] if self.samples else 0.0)
        for m in self.marks[1:]:
            out.append((label, start, m["t_rel_s"]))
            label, start = m["label"], m["t_rel_s"]
        out.append((label, start, end_all))
        return [(l, a, b) for l, a, b in out if b > a and l != "__end__"]

    def report(self, baseline_phase: str | None = None,
               series_decimals: int = 3) -> dict[str, Any]:
        """Full hardware-energy report: series, per-device stats, per-phase integrals.

        ``baseline_phase`` names a phase (typically an idle baseline) whose median power
        per device is subtracted to give ``joules_above_baseline`` -- the energy actually
        attributable to the workload rather than to the machine sitting there powered on.
        """
        times = [r["t_rel_s"] for r in self.samples]
        device_values: dict[str, list[float | None]] = {
            dev: [r.get(dev) for r in self.samples] for dev in self.device_columns}
        intervals = [b - a for a, b in zip(times, times[1:])]
        iv_stats = _stats(intervals)
        duration = (self._t_end if self._t_end is not None
                    else (times[-1] if times else 0.0))

        devices: dict[str, Any] = {}
        for dev, vals in device_values.items():
            joules, n_int = integrate(times, vals)
            st = _stats(vals)
            devices[dev] = {
                "unit": "W", "n_samples": st["n"],
                "min_w": st["min"], "max_w": st["max"], "mean_w": st["mean"],
                "median_w": st["median"], "joules_total": joules, "integral_intervals": n_int,
                "mean_w_reported": (None if st["mean"] is None else round(st["mean"], 3)),
            }

        baseline_w: dict[str, float | None] = {}
        if baseline_phase is not None:
            for dev, vals in device_values.items():
                sel = [v for r, v in zip(self.samples, vals)
                       if r["phase"] == baseline_phase and v is not None]
                baseline_w[dev] = _stats(sel)["median"]

        phases: list[dict[str, Any]] = []
        for label, a, b in self._phase_windows():
            row_samples = [r for r in self.samples if a <= r["t_rel_s"] <= b]
            pdev: dict[str, Any] = {}
            for dev, vals in device_values.items():
                joules, n_int = integrate(times, vals, a, b)
                sel = [v for r, v in zip(self.samples, vals)
                       if a <= r["t_rel_s"] <= b and v is not None]
                st = _stats(sel)
                entry = {"mean_w": st["mean"], "max_w": st["max"], "median_w": st["median"],
                         "joules_total": joules, "n_samples": st["n"],
                         "integral_intervals": n_int}
                base = baseline_w.get(dev)
                entry["joules_above_baseline"] = (
                    None if base is None else joules - base * (b - a))
                entry["baseline_median_w"] = base
                pdev[dev] = entry
            phases.append({"label": label, "t_start_s": a, "t_end_s": b,
                           "wall_seconds": b - a, "samples": len(row_samples),
                           "devices": pdev})

        series: dict[str, Any] = {"t_rel_s": times, "phase": [r["phase"] for r in self.samples]}
        for dev in self.device_columns:
            series[dev] = [None if r.get(dev) is None else round(r[dev], series_decimals)
                           for r in self.samples]
        for extra in [c for c in (self.samples[0] if self.samples else {})
                      if c.endswith(":util_pct")]:
            series[extra] = [r.get(extra) for r in self.samples]

        return {
            "track": "hardware",
            "track_note": ("Actual measured hardware energy. Never to be conflated with the "
                           "biological-equivalent track (PROJECT-VYBFLY.md §17)."),
            "sampler": {
                "interval_target_s": self.interval_s,
                "interval_stats_s": iv_stats,
                "achieved_hz": (None if not iv_stats["mean"] else 1.0 / iv_stats["mean"]),
                "n_samples": len(self.samples),
                "duration_s": duration,
                "device_columns": self.device_columns,
                "gpu_backend": self.gpu_backend,
                "baseline_phase": baseline_phase,
                "baseline_w": baseline_w,
            },
            "device_availability": self.probe,
            "devices": devices,
            "phases": phases,
            "marks": self.marks,
            "series": series,
            "collected_data": bool(self.samples and any(
                v is not None for vals in device_values.values() for v in vals)),
            "notes": self.notes,
            "generated_utc": _utcnow(),
        }


# --------------------------------------------------------------------------------------
# biological-equivalent model (track A)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetabolicAnchor:
    """A published metabolic measurement (or clearly-labeled derived estimate)."""
    key: str
    label: str
    p_watts: float
    n_neurons: float
    reported_value: str
    species: str
    method: str
    citation_title: str
    authors: str
    year: int
    doi: str
    url: str
    is_drosophila_measurement: bool
    not_a_drosophila_measurement: bool
    caveats: tuple[str, ...]
    provenance_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def per_neuron_watts(self) -> float:
        return self.p_watts / self.n_neurons

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_neuron_watts"] = self.per_neuron_watts
        d["per_neuron_picowatts"] = self.per_neuron_watts * 1e12
        return d


#: Free energy of ATP hydrolysis under cellular conditions. 50 kJ/mol is the value used
#: in the neuroenergetics literature; standard-state 30.5 kJ/mol would give 0.61x this,
#: so every ATP-derived joule figure below carries a +-40% assumption band.
ATP_HYDROLYSIS_J = 8.30e-20
ATP_HYDROLYSIS_ASSUMPTION = ("50 kJ/mol free energy of ATP hydrolysis under cellular "
                             "conditions / N_A = 8.30e-20 J per ATP; standard-state "
                             "30.5 kJ/mol would scale all ATP-derived joules by 0.61")

#: FlyWire whole-brain neuron count used as N0 (see FLYWIRE_V783_NEURONS).
ANCHORS: dict[str, MetabolicAnchor] = {
    "drosophila_brain_heat_panda2026": MetabolicAnchor(
        key="drosophila_brain_heat_panda2026",
        label="Drosophila whole-brain metabolic heat output (direct calorimetry)",
        p_watts=256e-9,
        n_neurons=FLYWIRE_V783_NEURONS,
        reported_value="~256 nW per brain (female, 10-day-old, y sc v)",
        species="Drosophila melanogaster",
        method=("nanowatt-resolution biocalorimeter, live explanted brain perfused with "
                "Schneider's medium at 4 uL/min; Q = G_th * dT; ~1 h per measurement"),
        citation_title=("Direct quantification of the metabolic heat output of individual "
                        "Drosophila brains"),
        authors=("Panda K, Mittapally R, Chen Q, Bhaskaran A, Reddy P, Meyhofer E, "
                 "Yadlapalli S"),
        year=2026,
        doi="10.1016/j.crmeth.2026.101501",
        url="https://doi.org/10.1016/j.crmeth.2026.101501",
        is_drosophila_measurement=True,
        not_a_drosophila_measurement=False,
        caveats=(
            "ex vivo: the brain is explanted and buffer-perfused, not in a behaving fly",
            "one genotype (y sc v), one sex (female), one age (10 days) -- sex/age/genotype "
            "differences of 10-15% are reported within the same study",
            "calorimeter time constant ~40 s and resolution ~7.6 nW: fast (sub-second) "
            "metabolic transients such as spike-driven ATP flux are smoothed away, so this "
            "is a slow/steady-state figure",
            "pairing the measured organ with N0 = 139,255 is an assumption: the neuron count "
            "is the FlyWire FAFB v783 whole-brain count, not a count of the measured animal",
            "metabolic heat output is total tissue metabolism (neurons + glia + housekeeping), "
            "not a measure of spiking computation",
        ),
        provenance_extra={
            "preprint": "bioRxiv 2025.08.08.669302 (DOI 10.1101/2025.08.08.669302)",
            "dry_mass": "~6.6 ug per female brain; ~38.7 nW/ug",
            "buffer_flow_sensitivity": "2 uL/min -> ~62 nW and unstable; 4 uL/min -> ~256 nW; "
                                       "6 uL/min -> ~247 nW (oxygen supply limited at 2)",
        },
    ),
    "mammalian_per_neuron_herculano_houzel2011": MetabolicAnchor(
        key="mammalian_per_neuron_herculano_houzel2011",
        label="Mammalian whole-brain metabolic cost per neuron (the §17 null model)",
        p_watts=6.0e-9 * 4184.0 / 86400.0,   # kcal/day per neuron -> W per neuron
        n_neurons=1.0,
        reported_value="~6 kcal/day per billion neurons (= 6e-9 kcal/neuron/day = 0.291 nW/neuron)",
        species="rodents and primates (mouse, rat, squirrel, monkey, baboon, human)",
        method=("published glucose+oxygen metabolic rates of awake whole brains (Karbowski "
                "2007 data) divided by counted neuron numbers; glucose use per neuron "
                "constant within 40% across a 1000x span in neuron number; total brain "
                "glucose use linear in N with power exponent 0.988"),
        citation_title=("Scaling of brain metabolism with a fixed energy budget per neuron: "
                        "implications for neuronal activity, plasticity and evolution"),
        authors="Herculano-Houzel S",
        year=2011,
        doi="10.1371/journal.pone.0017514",
        url="https://doi.org/10.1371/journal.pone.0017514",
        is_drosophila_measurement=False,
        not_a_drosophila_measurement=True,
        caveats=(
            "MAMMALIAN ONLY: this is evidence for the linear-per-neuron null model, and "
            "PROJECT-VYBFLY.md §17 explicitly forbids presenting it as a Drosophila measurement",
            "method is a derived quantity: measured whole-brain glucose/oxygen uptake divided "
            "by counted neurons, not a direct per-neuron measurement",
            "whole-brain average; cerebral cortex is >=10x the per-neuron cost of cerebellum",
            "'awake resting', no task; task/activity scaling is not included",
            "the value pairs with the fly only as a null-model ratio, so the resulting "
            "P_bio(fly-scale) is ~158x the directly measured fly calorimetry value",
        ),
        provenance_extra={
            "watt_conversion": "6e-9 kcal/neuron/day * 4184 J/kcal / 86400 s = 2.9056e-10 W",
            "human_check": "86e9 neurons * 6 kcal/day = 516 kcal/day ~ 25 W (paper's figure)",
        },
    ),
    "fly_nervous_system_estimate_scheffer2021": MetabolicAnchor(
        key="fly_nervous_system_estimate_scheffer2021",
        label="Published *estimate* of Drosophila nervous-system compute power (~120 nW)",
        p_watts=120e-9,
        n_neurons=FLYWIRE_V783_NEURONS,
        reported_value="~1.2e-7 W (~120 nW) for the Drosophila nervous system",
        species="Drosophila melanogaster",
        method=("derived: Chadwick's resting oxygen consumption of 26 mm^3/g/min for a ~1 mg "
                "fly, scaled to D. melanogaster mass and to the ~5% of resting metabolism "
                "attributed to the nervous system; NOT a calorimetric or electrophysiological "
                "measurement"),
        citation_title=("The Physical Design of Biological Systems - Insights from the Fly Brain"),
        authors="Scheffer LK",
        year=2021,
        doi="10.1145/3439706.3446898",
        url="https://doi.org/10.1145/3439706.3446898",
        is_drosophila_measurement=False,
        not_a_drosophila_measurement=False,
        caveats=(
            "derived estimate, not a measurement (oxygen-consumption proxy x assumed 5% "
            "nervous-system share x assumed body mass)",
            "same order of magnitude as the direct calorimetry value (120 nW vs 256 nW), so it "
            "corroborates the fly-scale anchor to within ~2.1x; used here as a cross-check only",
            "no neuron count is stated in the source; paired here with the FlyWire v783 count",
        ),
        provenance_extra={"cross_check_ratio_vs_calorimetry": 120e-9 / 256e-9},
    ),
}

#: Mammalian cortical gray-matter cost per spike and per synaptic release event.
#: Both are mammalian-cortex derived (Attwell & Laughlin 2001) -- flagged as such.
MAMMALIAN_CORTEX_EVENTS: dict[str, dict[str, Any]] = {
    "per_action_potential": {
        "atp_per_neuron_per_spike": 7.1e8,
        "joules": 7.1e8 * ATP_HYDROLYSIS_J,
        "citation_title": "An energy budget for signaling in the gray matter of the brain",
        "authors": "Attwell D, Laughlin SB",
        "year": 2001,
        "doi": "10.1097/00004647-200110000-00001",
        "url": "https://doi.org/10.1097/00004647-200110000-00001",
        "not_a_drosophila_measurement": True,
        "caveats": ("mammalian cortical gray matter; 'total ATP consumption when a neuron "
                    "fires an action potential ... 7.1e8 ATP/neuron/spike', i.e. axon + "
                    "presynaptic + postsynaptic ion fluxes, not the axonal spike alone"),
    },
    "per_vesicle_release": {
        "atp_per_vesicle_released": 1.64e5,
        "joules": 1.64e5 * ATP_HYDROLYSIS_J,
        "citation_title": "An energy budget for signaling in the gray matter of the brain",
        "authors": "Attwell D, Laughlin SB",
        "year": 2001,
        "doi": "10.1097/00004647-200110000-00001",
        "url": "https://doi.org/10.1097/00004647-200110000-00001",
        "not_a_drosophila_measurement": True,
        "caveats": ("mammalian cortical gray matter; energy per glutamate vesicle actually "
                    "released (presynaptic Ca2+ entry, vesicle cycling, postsynaptic actions, "
                    "glutamate recycling), 84% of it postsynaptic ion pumping. This is a "
                    "release event, not one anatomical synapse of the connectome"),
    },
}

#: Biological scaling-experiment scales from §18 (N relative to N0).
SCALING_SCALES = (1, 2, 5, 10, 25, 50, 100)


class BiologicalEnergyModel:
    """``P_bio(N) = P0 * N / N0`` with the anchor's citation and provenance attached.

    ``P0`` is the anchor's measured power, ``N0`` the neuron count it was measured with
    (for the mammalian per-neuron anchor ``N0 = 1`` neuron). Every returned value is
    biology-side only; nothing here says anything about this machine's electricity use.
    """

    track = "biological"

    def __init__(self, anchor_key: str = "drosophila_brain_heat_panda2026",
                 n0: float | None = None):
        if anchor_key not in ANCHORS:
            raise KeyError(f"unknown anchor {anchor_key!r}; have {sorted(ANCHORS)}")
        self.anchor = ANCHORS[anchor_key]
        self.anchor_key = anchor_key
        self.n0 = float(self.anchor.n_neurons if n0 is None else n0)

    @property
    def p0_watts(self) -> float:
        return self.anchor.p_watts

    @property
    def per_neuron_watts(self) -> float:
        return self.anchor.per_neuron_watts

    def power_w(self, n_neurons: float) -> float:
        return self.p0_watts * (n_neurons / self.n0)

    def joules(self, n_neurons: float, seconds: float) -> float:
        return self.power_w(n_neurons) * seconds

    def at_scales(self, n1: float = float(FLYWIRE_V783_NEURONS),
                  scales: Sequence[float] = SCALING_SCALES) -> list[dict[str, float]]:
        """P_bio at the §18 scales (1x..100x of the FlyWire neuron count)."""
        return [{"scale": float(s), "n_neurons": n1 * s, "p_bio_w": self.power_w(n1 * s),
                 "p_bio_nw": self.power_w(n1 * s) * 1e9} for s in scales]

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": "biological",
            "track_note": ("Biological-equivalent energy (PROJECT-VYBFLY.md §17 A). NOT this "
                           "machine's electricity use; the two tracks must never be conflated."),
            "model": "P_bio(N) = P0 * N / N0",
            "anchor_key": self.anchor_key,
            "P0_watts": self.p0_watts,
            "N0_neurons": self.n0,
            "per_neuron_watts": self.per_neuron_watts,
            "per_neuron_picowatts": self.per_neuron_watts * 1e12,
            "units": "watts (J/s), continuous metabolic power",
            "anchor": self.anchor.as_dict(),
            "not_a_drosophila_measurement": self.anchor.not_a_drosophila_measurement,
            "is_drosophila_measurement": self.anchor.is_drosophila_measurement,
            "flywire_v783_neuron_count": FLYWIRE_V783_NEURONS,
            "comparison_anchors": {k: v.as_dict() for k, v in ANCHORS.items()
                                   if k != self.anchor_key},
            "comparison_per_neuron_picowatts": {k: v.per_neuron_watts * 1e12
                                                for k, v in ANCHORS.items()},
            "mammalian_over_fly_calorimetry_ratio": (
                ANCHORS["mammalian_per_neuron_herculano_houzel2011"].per_neuron_watts
                / ANCHORS["drosophila_brain_heat_panda2026"].per_neuron_watts),
            "scaling_table": self.at_scales(),
            "provenance": {
                "atp_hydrolysis_j": ATP_HYDROLYSIS_J,
                "atp_hydrolysis_assumption": ATP_HYDROLYSIS_ASSUMPTION,
            },
            "mammalian_cortex_per_event": {
                k: dict(v, joules=v["joules"]) for k, v in MAMMALIAN_CORTEX_EVENTS.items()},
            "derived_fly_per_spike": {
                "note": ("NOT a published measurement: 256 nW / 139,255 neurons divided by an "
                         "assumed mean firing rate, i.e. total tissue power attributed to "
                         "spikes. Given only to contrast orders of magnitude; the calorimeter "
                         "cannot resolve spikes."),
                "assumed_mean_rate_hz": 1.0,
                "joules_per_neuron_spike_at_1hz": self.per_neuron_watts / 1.0,
                "not_a_drosophila_measurement": False,
                "is_derived_not_measured": True,
            },
            "generated_utc": _utcnow(),
        }
