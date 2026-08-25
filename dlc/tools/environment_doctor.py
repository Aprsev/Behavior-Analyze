#!/usr/bin/env python
"""Diagnose and optionally complete the Hybrid YOLO + SR + DLC environment.

This script intentionally depends only on the Python standard library so it can
run inside a partially configured Conda environment.

Examples
--------
    python environment_doctor.py
    python environment_doctor.py --install --torch cu126
    python environment_doctor.py --install --torch cpu --yes
    python environment_doctor.py --install --dry-run --torch-index-url URL
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


EXIT_OK = 0
EXIT_ISSUES = 1
EXIT_UNSAFE_ENVIRONMENT = 2
EXIT_INSTALL_FAILED = 3

PYTHON_MIN = (3, 10)
PYTHON_MAX = (3, 12)

TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


@dataclass(frozen=True)
class Requirement:
    label: str
    distributions: tuple[str, ...]
    module: str
    install_spec: str
    minimum: tuple[int, ...] | None = None
    maximum_exclusive: tuple[int, ...] | None = None
    capability: str | None = None


@dataclass
class CheckResult:
    label: str
    status: str
    version: str | None
    detail: str
    install_spec: str | None = None


REQUIREMENTS = (
    Requirement(
        "DeepLabCut (GUI + Model Zoo)",
        ("deeplabcut",),
        "deeplabcut",
        "deeplabcut[gui,modelzoo]>=3.0.0,<4",
        (3, 0, 0),
        (4, 0, 0),
    ),
    Requirement(
        "Ultralytics YOLO",
        ("ultralytics",),
        "ultralytics",
        "ultralytics>=8.3",
        (8, 3),
    ),
    Requirement("PySide6", ("PySide6",), "PySide6", "PySide6>=6.6", (6, 6)),
    Requirement(
        "OpenCV contrib",
        ("opencv-contrib-python", "opencv-contrib-python-headless"),
        "cv2",
        "opencv-contrib-python>=4.8",
        (4, 8),
        capability="dnn_superres",
    ),
    Requirement("NumPy", ("numpy",), "numpy", "numpy>=1.24", (1, 24)),
    Requirement("pandas", ("pandas",), "pandas", "pandas>=2.0,<3", (2, 0), (3, 0)),
    Requirement("PyTables", ("tables",), "tables", "tables>=3.9", (3, 9)),
)

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def numeric_version(value: str) -> tuple[int, ...]:
    """Return the leading numeric release components without external helpers."""
    match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def version_in_range(
    version: str,
    minimum: tuple[int, ...] | None,
    maximum_exclusive: tuple[int, ...] | None,
) -> bool:
    parsed = numeric_version(version)
    if not parsed:
        return False
    width = max(
        len(parsed),
        len(minimum or ()),
        len(maximum_exclusive or ()),
    )
    padded = parsed + (0,) * (width - len(parsed))
    if minimum:
        lower = minimum + (0,) * (width - len(minimum))
        if padded < lower:
            return False
    if maximum_exclusive:
        upper = maximum_exclusive + (0,) * (width - len(maximum_exclusive))
        if padded >= upper:
            return False
    return True


def distribution_version(names: Iterable[str]) -> tuple[str | None, str | None]:
    for name in names:
        try:
            return metadata.version(name), name
        except metadata.PackageNotFoundError:
            continue
    return None, None


def run_process(command: Sequence[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def probe_module(module: str, capability: str | None, timeout: int) -> tuple[bool, str]:
    code = (
        "import importlib, json\n"
        f"m = importlib.import_module({module!r})\n"
        f"cap = {capability!r}\n"
        "ok = cap is None or hasattr(m, cap)\n"
        "print(json.dumps({'ok': ok, 'version': str(getattr(m, '__version__', 'unknown'))}))\n"
    )
    try:
        result = run_process((sys.executable, "-c", code), timeout)
    except subprocess.TimeoutExpired:
        return False, f"import timed out after {timeout}s"
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().splitlines()
        return False, error[-1] if error else f"import returned {result.returncode}"
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return False, "import probe produced no readable result"
    if not payload.get("ok"):
        return False, f"module imports but capability '{capability}' is missing"
    return True, f"import OK ({payload.get('version', 'unknown')})"


def inspect_requirement(requirement: Requirement, timeout: int) -> CheckResult:
    version, distribution = distribution_version(requirement.distributions)
    if version is None:
        return CheckResult(
            requirement.label,
            "MISSING",
            None,
            "package metadata not found",
            requirement.install_spec,
        )
    if not version_in_range(version, requirement.minimum, requirement.maximum_exclusive):
        return CheckResult(
            requirement.label,
            "INCOMPATIBLE",
            version,
            f"installed as {distribution}; required by this project: {requirement.install_spec}",
            requirement.install_spec,
        )
    import_ok, detail = probe_module(requirement.module, requirement.capability, timeout)
    if not import_ok:
        return CheckResult(
            requirement.label,
            "BROKEN",
            version,
            detail,
            requirement.install_spec,
        )
    return CheckResult(requirement.label, "OK", version, detail)


def conda_information() -> dict[str, Any]:
    prefix = os.environ.get("CONDA_PREFIX")
    name = os.environ.get("CONDA_DEFAULT_ENV")
    active = bool(prefix)
    same_prefix = False
    if prefix:
        current = os.path.normcase(os.path.abspath(sys.prefix))
        expected = os.path.normcase(os.path.abspath(prefix))
        same_prefix = current == expected
    return {
        "active": active,
        "name": name,
        "conda_prefix": prefix,
        "python_prefix": sys.prefix,
        "python_executable": sys.executable,
        "same_prefix": same_prefix,
        "is_base": bool(name and name.lower() == "base"),
    }


def torch_information(timeout: int) -> dict[str, Any]:
    version, _ = distribution_version(("torch",))
    torchvision, _ = distribution_version(("torchvision",))
    result: dict[str, Any] = {
        "installed": version is not None,
        "version": version,
        "torchvision": torchvision,
        "import_ok": False,
        "cuda_available": False,
    }
    if version is None:
        result["detail"] = "PyTorch is not installed"
        return result
    code = """
import json, torch
data = {
    'import_ok': True,
    'torch_version': str(torch.__version__),
    'compiled_cuda': str(torch.version.cuda),
    'cuda_available': bool(torch.cuda.is_available()),
    'cudnn': str(torch.backends.cudnn.version()),
}
if data['cuda_available']:
    data['device_count'] = torch.cuda.device_count()
    data['device_name'] = torch.cuda.get_device_name(0)
    x = torch.ones((8, 8), device='cuda')
    data['allocation_test'] = float(x.sum().item()) == 64.0
print(json.dumps(data))
"""
    try:
        probe = run_process((sys.executable, "-c", code), timeout)
    except subprocess.TimeoutExpired:
        result["detail"] = f"PyTorch probe timed out after {timeout}s"
        return result
    if probe.returncode != 0:
        lines = (probe.stderr or probe.stdout).strip().splitlines()
        result["detail"] = lines[-1] if lines else "PyTorch import failed"
        return result
    try:
        result.update(json.loads(probe.stdout.strip().splitlines()[-1]))
        result["detail"] = "PyTorch import and tensor probe passed"
    except (IndexError, json.JSONDecodeError):
        result["detail"] = "PyTorch probe produced no readable result"
    return result


def nvidia_information(timeout: int) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "detail": "nvidia-smi was not found"}
    command = (
        executable,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    try:
        result = run_process(command, min(timeout, 30))
    except subprocess.TimeoutExpired:
        return {"available": False, "detail": "nvidia-smi timed out"}
    if result.returncode != 0:
        return {"available": False, "detail": (result.stderr or result.stdout).strip()}
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": True, "executable": executable, "devices": devices}


def installed_opencv_distributions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in OPENCV_DISTRIBUTIONS:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def pip_check(timeout: int) -> dict[str, Any]:
    try:
        result = run_process((sys.executable, "-m", "pip", "check"), timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "pip check timed out"}
    output = (result.stdout or result.stderr).strip()
    return {"ok": result.returncode == 0, "detail": output or "No broken requirements found."}


def quote_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def build_install_commands(
    results: Sequence[CheckResult],
    torch_choice: str,
    torch_index_url: str | None,
    fix_opencv: bool,
    upgrade: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    pip = [sys.executable, "-m", "pip"]
    torch_url = torch_index_url or TORCH_INDEXES.get(torch_choice)
    if torch_choice != "keep" or torch_index_url:
        if not torch_url:
            raise ValueError("A valid --torch-index-url is required for this torch selection")
        command = pip + ["install"]
        if upgrade:
            command.append("--upgrade")
        command += ["torch", "torchvision", "--index-url", torch_url]
        commands.append(command)

    specs = [item.install_spec for item in results if item.install_spec]
    opencv_needs_work = any(item.label == "OpenCV contrib" and item.install_spec for item in results)
    if fix_opencv and opencv_needs_work:
        conflicts = list(installed_opencv_distributions())
        if conflicts:
            commands.append(pip + ["uninstall", "-y", *conflicts])
    if specs:
        command = pip + ["install"]
        if upgrade:
            command.append("--upgrade")
        command.extend(dict.fromkeys(specs))
        commands.append(command)
    return commands


def print_header(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def print_report(report: dict[str, Any]) -> None:
    conda = report["conda"]
    print_header("Conda environment")
    print(f"Environment:       {conda.get('name') or '(not detected)'}")
    print(f"Python executable: {conda['python_executable']}")
    print(f"Python version:    {report['python_version']}")
    print(f"Conda prefix:      {conda.get('conda_prefix') or '(not detected)'}")
    print(f"Prefix match:      {'yes' if conda['same_prefix'] else 'no'}")

    nvidia = report["nvidia"]
    torch = report["torch"]
    print_header("GPU and PyTorch")
    if nvidia.get("available"):
        for device in nvidia.get("devices", []):
            print(f"NVIDIA GPU:        {device}")
    else:
        print(f"NVIDIA GPU:        not detected ({nvidia.get('detail')})")
    print(f"PyTorch:           {torch.get('version') or 'missing'}")
    print(f"torchvision:       {torch.get('torchvision') or 'missing'}")
    print(f"Compiled CUDA:     {torch.get('compiled_cuda', 'unknown')}")
    print(f"CUDA usable:       {'yes' if torch.get('cuda_available') else 'no'}")
    if torch.get("device_name"):
        print(f"Torch GPU:         {torch['device_name']}")
    print(f"Torch probe:       {torch.get('detail', 'unknown')}")

    print_header("Hybrid pipeline packages")
    for item in report["packages"]:
        version = item.get("version") or "-"
        print(f"{item['status']:<12} {item['label']:<28} {version:<15} {item['detail']}")
    opencv = report["opencv_distributions"]
    if opencv:
        print("OpenCV wheels:     " + ", ".join(f"{key}={value}" for key, value in opencv.items()))
    print(f"pip check:         {'OK' if report['pip_check']['ok'] else 'FAILED'}")
    if not report["pip_check"]["ok"]:
        print(report["pip_check"]["detail"])


def write_json_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to: {destination}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and optionally complete the Conda environment for the Hybrid YOLO + SR + DLC GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--install", action="store_true", help="Install missing or incompatible packages.")
    parser.add_argument("--yes", action="store_true", help="Run the displayed installation commands without prompting.")
    parser.add_argument("--dry-run", action="store_true", help="Display installation commands without executing them.")
    parser.add_argument("--upgrade", action="store_true", help="Pass --upgrade to pip installation commands.")
    parser.add_argument(
        "--torch",
        choices=("keep", *TORCH_INDEXES),
        default="keep",
        help="Keep existing PyTorch, or explicitly install a CPU/CUDA wheel build.",
    )
    parser.add_argument(
        "--torch-index-url",
        help="Official/custom PyTorch wheel index. Overrides the --torch preset and explicitly enables PyTorch installation.",
    )
    parser.add_argument(
        "--fix-opencv",
        action="store_true",
        help="When OpenCV contrib is broken, remove installed OpenCV wheel variants before reinstalling it.",
    )
    parser.add_argument("--allow-base", action="store_true", help="Allow installation into the Conda base environment.")
    parser.add_argument("--allow-non-conda", action="store_true", help="Allow installation outside an active Conda environment.")
    parser.add_argument("--require-gpu", action="store_true", help="Treat an unavailable CUDA device as an error.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for each import/command probe.")
    parser.add_argument("--json-report", metavar="PATH", help="Write the diagnostic result as JSON.")
    return parser.parse_args(argv)


def collect_report(timeout: int) -> dict[str, Any]:
    packages = [inspect_requirement(requirement, timeout) for requirement in REQUIREMENTS]
    return {
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_supported": PYTHON_MIN <= sys.version_info[:2] <= PYTHON_MAX,
        "conda": conda_information(),
        "nvidia": nvidia_information(timeout),
        "torch": torch_information(timeout),
        "packages": [asdict(item) for item in packages],
        "opencv_distributions": installed_opencv_distributions(),
        "pip_check": pip_check(timeout),
    }


def report_has_issues(report: dict[str, Any], require_gpu: bool) -> bool:
    package_issue = any(item["status"] != "OK" for item in report["packages"])
    torch_issue = not report["torch"].get("import_ok") or not report["torch"].get("torchvision")
    gpu_issue = require_gpu and not report["torch"].get("cuda_available")
    return package_issue or torch_issue or gpu_issue or not report["pip_check"]["ok"]


def installation_safety_error(args: argparse.Namespace, report: dict[str, Any]) -> str | None:
    if not report["python_supported"]:
        return "DeepLabCut requires Python 3.10-3.12; create a compatible Conda environment first."
    conda = report["conda"]
    if (not conda["active"] or not conda["same_prefix"]) and not args.allow_non_conda:
        return "The active Python does not match an active Conda environment. Activate it first, or use --allow-non-conda."
    if conda["is_base"] and not args.allow_base:
        return "Installation into Conda base is blocked. Activate a project environment, or use --allow-base."
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout < 5:
        print("ERROR: --timeout must be at least 5 seconds.", file=sys.stderr)
        return EXIT_UNSAFE_ENVIRONMENT

    report = collect_report(args.timeout)
    print_report(report)
    write_json_report(args.json_report, report)

    if not report["python_supported"]:
        print("\nERROR: Supported DeepLabCut Python range is 3.10 through 3.12.")

    if not args.install:
        if not report["torch"].get("installed"):
            print("\nPyTorch is missing. Choose the correct build explicitly, for example:")
            print(f"  {quote_command((sys.executable, str(Path(__file__).resolve()), '--install', '--torch', 'cu126'))}")
            print("Use --torch cpu only when GPU acceleration is not required.")
        elif report["nvidia"].get("available") and not report["torch"].get("cuda_available"):
            print("\nWARNING: NVIDIA hardware is visible, but the installed PyTorch cannot use CUDA.")
            print("Re-run with --install and an appropriate --torch preset or --torch-index-url.")
        print("\nDiagnostic mode made no changes. Add --install to apply the displayed repair plan.")
        return EXIT_ISSUES if report_has_issues(report, args.require_gpu) else EXIT_OK

    safety_error = installation_safety_error(args, report)
    if safety_error:
        print(f"\nINSTALLATION BLOCKED: {safety_error}", file=sys.stderr)
        return EXIT_UNSAFE_ENVIRONMENT

    result_objects = [CheckResult(**item) for item in report["packages"]]
    try:
        commands = build_install_commands(
            result_objects,
            args.torch,
            args.torch_index_url,
            args.fix_opencv,
            args.upgrade,
        )
    except ValueError as error:
        print(f"\nINSTALLATION BLOCKED: {error}", file=sys.stderr)
        return EXIT_UNSAFE_ENVIRONMENT

    if not report["torch"].get("installed") and args.torch == "keep" and not args.torch_index_url:
        print("\nNOTICE: PyTorch is missing and was not added to the plan.")
        print("Select --torch cpu/cu126/cu128, or pass the index shown by the official PyTorch selector.")

    print_header("Installation plan")
    if not commands:
        print("No installation commands are required.")
        return EXIT_ISSUES if report_has_issues(report, args.require_gpu) else EXIT_OK
    for command in commands:
        print(quote_command(command))
    if args.dry_run:
        print("\nDry run complete; no changes were made.")
        return EXIT_ISSUES if report_has_issues(report, args.require_gpu) else EXIT_OK

    if not args.yes:
        if not sys.stdin.isatty():
            print("\nINSTALLATION BLOCKED: Non-interactive input requires --yes.", file=sys.stderr)
            return EXIT_UNSAFE_ENVIRONMENT
        answer = input("\nRun these commands in the current environment? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Installation cancelled.")
            return EXIT_ISSUES

    for command in commands:
        print(f"\nRunning: {quote_command(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"Installation failed with exit code {completed.returncode}.", file=sys.stderr)
            return EXIT_INSTALL_FAILED

    print("\nInstallation finished. Re-running all probes...")
    final_report = collect_report(args.timeout)
    print_report(final_report)
    write_json_report(args.json_report, final_report)
    if report_has_issues(final_report, args.require_gpu):
        print("\nInstallation completed, but one or more checks still need attention.")
        return EXIT_ISSUES
    print("\nEnvironment is ready for the Hybrid YOLO + SR + DLC GUI.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
