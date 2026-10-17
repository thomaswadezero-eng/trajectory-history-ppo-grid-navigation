#!/usr/bin/env python
"""Manual, gated runtime preflight.  Never called by static audit."""

from __future__ import annotations

import os

# Preserve the user's process environment before importing audit/orchestration
# modules.  In particular, CUDA_VISIBLE_DEVICES must survive any audit import.
ORIGINAL_CHILD_ENV = os.environ.copy()

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import common
from run_external_baselines_8seeds import MASKABLE_PYTHON, VALUE_C51_PYTHON, current_identity


PASS_MARKER = common.OUTPUT_ROOT / "preflight" / "RUNTIME_PREFLIGHT_PASS.json"


def child_environment(snapshot: Mapping[str, str]) -> dict[str, str]:
    return {**snapshot, "PYTHONUNBUFFERED": "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External-baseline isolated runtime preflight")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-runtime-preflight", action="store_true")
    return parser.parse_args()


def _interpreter(method: str) -> Path:
    if method == "MaskablePPO":
        if not MASKABLE_PYTHON.is_file():
            raise common.AuditError(f"frozen MaskablePPO interpreter missing: {MASKABLE_PYTHON}")
        return MASKABLE_PYTHON
    if not VALUE_C51_PYTHON.is_file():
        raise common.AuditError(f"frozen value/C51 interpreter missing: {VALUE_C51_PYTHON}")
    return VALUE_C51_PYTHON


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _write_probe_log(
    path: Path,
    context: Mapping[str, Any],
    stdout: str,
    stderr: str,
) -> None:
    header = [
        "[PREFLIGHT METADATA]",
        f"method={context['method']}",
        f"python={context['python_executable']}",
        f"device={context['device']}",
        f"working_directory={context['working_directory']}",
        f"cuda_visible_devices_present={context['cuda_visible_devices_present']}",
        f"cuda_visible_devices={context['cuda_visible_devices']}",
        f"pythonpath_present={context['pythonpath_present']}",
        f"pythonpath={context['pythonpath']}",
        f"command={context['command_display']}",
        "[CHILD STDOUT]",
    ]
    body = "\n".join(header) + "\n" + stdout
    if stdout and not stdout.endswith("\n"):
        body += "\n"
    body += "[CHILD STDERR]\n" + stderr
    if stderr and not stderr.endswith("\n"):
        body += "\n"
    path.write_text(body, encoding="utf-8")


def _last_error(stdout: str, stderr: str) -> str:
    for payload in (stderr, stdout):
        lines = [line for line in payload.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "child exited without diagnostic output"


def _run_streaming(
    command: list[str], child_env: Mapping[str, str], working_directory: Path
) -> tuple[int, str, str]:
    """Tee both child streams live while retaining their exact text."""
    process = subprocess.Popen(
        command,
        cwd=str(working_directory),
        env=dict(child_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def pump(stream: Any, destination: Any, captured: list[str]) -> None:
        for line in stream:
            captured.append(line)
            destination.write(line)
            destination.flush()

    stdout_thread = threading.Thread(
        target=pump, args=(process.stdout, sys.stdout, stdout_parts), daemon=True
    )
    stderr_thread = threading.Thread(
        target=pump, args=(process.stderr, sys.stderr, stderr_parts), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return_code = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    return return_code, "".join(stdout_parts), "".join(stderr_parts)


def main() -> None:
    args = parse_args()
    if not args.confirm_runtime_preflight:
        raise common.AuditError("runtime preflight requires --confirm-runtime-preflight")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = common.OUTPUT_ROOT / "preflight" / "work" / f"session_{stamp}"
    session.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    active_context: dict[str, Any] = {}
    try:
        bootstrap_result_path = session / "MaskablePPO_formal_wrapper_bootstrap.json"
        bootstrap_log_path = session / "MaskablePPO_formal_wrapper_bootstrap.log"
        bootstrap_command = [
            str(MASKABLE_PYTHON), "-u",
            str(common.REVISION_ROOT / "run_maskable_ppo_wrapper.py"),
            "--bootstrap-check", "--bootstrap-result", str(bootstrap_result_path),
        ]
        child_env = child_environment(ORIGINAL_CHILD_ENV)
        active_context = {
            "method": "MaskablePPO_formal_wrapper_bootstrap",
            "python_executable": str(MASKABLE_PYTHON),
            "device": args.device,
            "command": bootstrap_command,
            "command_display": _display_command(bootstrap_command),
            "working_directory": str(session),
            "cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in child_env,
            "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
            "pythonpath_present": "PYTHONPATH" in child_env,
            "pythonpath": child_env.get("PYTHONPATH"),
            "log": str(bootstrap_log_path),
        }
        print("[PREFLIGHT FORMAL-WRAPPER BOOTSTRAP]", flush=True)
        print(f"python={MASKABLE_PYTHON}", flush=True)
        print(f"working_directory={session}", flush=True)
        print(f"command={active_context['command_display']}", flush=True)
        return_code, child_stdout, child_stderr = _run_streaming(
            bootstrap_command, child_env, session
        )
        _write_probe_log(
            bootstrap_log_path, active_context, child_stdout, child_stderr
        )
        if return_code != 0:
            active_context.update({
                "return_code": return_code,
                "last_error": _last_error(child_stdout, child_stderr),
            })
            raise common.AuditError(
                f"MaskablePPO formal-wrapper bootstrap failed with code {return_code}"
            )
        wrapper_bootstrap = common.read_json(bootstrap_result_path)
        expected_common = str((common.REVISION_ROOT / "common.py").resolve())
        expected_authority = str(common.AUTHORITY["MaskablePPO"]["program"].resolve())
        if (
            wrapper_bootstrap.get("status") != "PASS"
            or wrapper_bootstrap.get("common_path") != expected_common
            or wrapper_bootstrap.get("authority_path") != expected_authority
            or wrapper_bootstrap.get("authority_sha256")
            != common.AUTHORITY["MaskablePPO"]["program_sha256"]
            or wrapper_bootstrap.get("legacy_module_imported") is not False
            or wrapper_bootstrap.get("torch_imported") is not False
            or wrapper_bootstrap.get("formal_training_interactions") != 0
            or wrapper_bootstrap.get("formal_result_created") is not False
        ):
            raise common.AuditError("invalid MaskablePPO formal-wrapper bootstrap evidence")
        for method in common.METHODS:
            result_path = session / f"{method}.json"
            interpreter = _interpreter(method)
            command = [
                str(interpreter), "-u", str(common.REVISION_ROOT / "runtime_probe.py"),
                "--method", method, "--device", args.device,
                "--work-dir", str(session / method), "--result-json", str(result_path),
                "--confirm-runtime-probe",
            ]
            child_env = child_environment(ORIGINAL_CHILD_ENV)
            log_path = session / f"{method}.log"
            active_context = {
                "method": method,
                "python_executable": str(interpreter),
                "device": args.device,
                "command": command,
                "command_display": _display_command(command),
                "working_directory": str(common.PROJECT_ROOT),
                "cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in child_env,
                "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
                "pythonpath_present": "PYTHONPATH" in child_env,
                "pythonpath": child_env.get("PYTHONPATH"),
                "log": str(log_path),
            }
            print("[PREFLIGHT]", flush=True)
            print(f"method={method}", flush=True)
            print(f"python={interpreter}", flush=True)
            print(f"device={args.device}", flush=True)
            print(f"command={active_context['command_display']}", flush=True)
            return_code, child_stdout, child_stderr = _run_streaming(
                command, child_env, common.PROJECT_ROOT
            )
            _write_probe_log(log_path, active_context, child_stdout, child_stderr)
            if return_code != 0:
                last_error = _last_error(child_stdout, child_stderr)
                active_context.update({
                    "return_code": return_code,
                    "last_error": last_error,
                })
                print("[PREFLIGHT FAILED]", file=sys.stderr, flush=True)
                print(f"method={method}", file=sys.stderr, flush=True)
                print(f"return_code={return_code}", file=sys.stderr, flush=True)
                print(f"log={log_path}", file=sys.stderr, flush=True)
                print(f"last_error={last_error}", file=sys.stderr, flush=True)
                raise common.AuditError(f"{method} preflight failed with code {return_code}")
            result = common.read_json(result_path)
            if result.get("status") != "PASS" or result.get("formal_training_interactions") != 0:
                raise common.AuditError(f"{method} returned an invalid preflight result")
            results[method] = result
        marker = {
            "schema_version": "external_baseline_runtime_preflight.v1",
            "status": "PASS",
            "created_at": common.now_iso(),
            "device": args.device,
            "methods": list(common.METHODS),
            "results": results,
            "formal_wrapper_bootstrap": wrapper_bootstrap,
            "revision_identity": current_identity(),
            "original_cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in ORIGINAL_CHILD_ENV,
            "original_cuda_visible_devices": ORIGINAL_CHILD_ENV.get("CUDA_VISIBLE_DEVICES"),
            "formal_training_interactions": 0,
            "formal_result_root_written": False,
            "preflight_session": str(session),
        }
        common.write_json_atomic(PASS_MARKER, marker)
        print(PASS_MARKER)
    except Exception as exc:
        failed = session / "RUNTIME_PREFLIGHT_FAILED.json"
        common.write_json_atomic(failed, {
            "status": "FAILED", "created_at": common.now_iso(),
            "device": args.device, "error_type": type(exc).__name__, "error": str(exc),
            "child": active_context,
            "original_cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in ORIGINAL_CHILD_ENV,
            "original_cuda_visible_devices": ORIGINAL_CHILD_ENV.get("CUDA_VISIBLE_DEVICES"),
            "formal_training_interactions": 0, "formal_result_root_written": False,
        })
        raise


if __name__ == "__main__":
    main()
