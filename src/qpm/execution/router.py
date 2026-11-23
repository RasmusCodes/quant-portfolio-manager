"""Adapter for an external smart order router (a separate repository, never imported at build time).

Two transports, both speaking the JSON contract in ``interface.py``:

* ``python``: a callable given as "package.module:function" that takes a list of parent-order dicts and
  returns a list of execution-report dicts.  The package must be installed in the same environment.
* ``cli``: a command that reads parent orders as JSON lines on stdin and writes execution reports as
  JSON lines on stdout (exit code 0).  Example: ``["python", "-m", "router.cli", "route"]``.

Every report is checked with ``validate_report``; a router that breaks the contract raises
``RouterContractError`` instead of silently corrupting a backtest.
"""
from __future__ import annotations

import importlib
import json
import subprocess
from typing import Callable, Sequence

from .interface import ExecutionReport, ParentOrder, validate_report


class RouterContractError(RuntimeError):
    pass


def _load_callable(path: str) -> Callable:
    module, _, attr = path.partition(":")
    if not module or not attr:
        raise ValueError("callable path must look like 'package.module:function'")
    fn = getattr(importlib.import_module(module), attr)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


class SmartOrderRouterAdapter:
    def __init__(self, transport: str = "cli", command: Sequence[str] | None = None, callable_path: str | None = None,
                 timeout: float = 120.0, name: str = "smart-order-router", strict: bool = True):
        if transport not in ("cli", "python"):
            raise ValueError("transport must be 'cli' or 'python'")
        if transport == "cli" and not command:
            raise ValueError("the cli transport needs a command")
        if transport == "python" and not callable_path:
            raise ValueError("the python transport needs callable_path")
        self.transport = transport
        self.command = list(command or [])
        self.callable_path = callable_path
        self.timeout = timeout
        self.name = name
        self.strict = strict
        self._fn = None

    def _send(self, payload: list[dict]) -> list[dict]:
        if self.transport == "python":
            if self._fn is None:
                self._fn = _load_callable(self.callable_path)
            return list(self._fn(payload))
        text = "\n".join(json.dumps(p) for p in payload) + "\n"
        proc = subprocess.run(self.command, input=text, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RouterContractError(f"router exited with {proc.returncode}: {proc.stderr.strip()[:500]}")
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

    def execute(self, orders: list[ParentOrder]) -> list[ExecutionReport]:
        if not orders:
            return []
        raw = self._send([o.to_dict() for o in orders])
        by_id = {}
        for d in raw:
            rep = ExecutionReport.from_dict(d)
            by_id[rep.order_id] = rep
        missing = [o.order_id for o in orders if o.order_id not in by_id]
        if missing:
            raise RouterContractError(f"router returned no report for {missing[:5]}")
        reports = []
        for o in orders:
            rep = by_id[o.order_id]
            problems = validate_report(o, rep)
            if problems and self.strict:
                raise RouterContractError(f"report for {o.order_id} breaks the contract: {'; '.join(problems)}")
            rep.executor = rep.executor or self.name
            reports.append(rep)
        return reports
