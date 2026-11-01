"""Code-execution sandbox.

Architecture
------------
* A long-lived *worker* process is started with the `spawn` method BEFORE any GPU model is
  loaded (so we never fork a process that holds CUDA state / many threads).
* The worker keeps the business tables in memory. For every code snippet it `fork`s a short
  lived child (copy-on-write => cheap even for 300k-row tables), runs the snippet with a
  timeout, and returns a JSON result. The child can never corrupt the loaded data.
* Snippets get a restricted `__import__`/`open` and only whitelisted modules.
  NOTE: this is a *safety net against accidents*, not a security boundary - the Docker
  container (offline, read-only /data) is the real boundary.
"""
from __future__ import annotations

import builtins
import contextlib
import datetime as dt
import io
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

ALLOWED_MODULES = {
    "pandas", "numpy", "math", "re", "datetime", "collections", "itertools", "functools",
    "statistics", "decimal", "fractions", "json", "unicodedata", "difflib", "string",
    "operator", "calendar", "time", "typing", "dateutil", "duckdb", "heapq", "bisect", "copy",
}
RESERVED_NAMES = {"pd", "np", "dfs", "rulebook", "calendar_df", "readme", "as_of", "AS_OF", "answer", "answer_type", "sql", "dt"}
MAX_STDOUT = 3000


# --------------------------------------------------------------------------- results
@dataclass
class ExecResult:
    ok: bool
    answer_type: str | None = None
    answer: Any = None
    reason: str | None = None
    stdout: str = ""
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class LoadInfo:
    profile: str = ""
    tables: dict = field(default_factory=dict)
    readme: str = ""
    warnings: list = field(default_factory=list)
    lunar_ok: bool = False


class SandboxError(RuntimeError):
    pass


# --------------------------------------------------------------------------- restricted builtins
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top not in ALLOWED_MODULES:
        raise ImportError(f"import of '{top}' is not allowed in the sandbox (allowed: pandas, numpy, math, re, datetime, collections, itertools, statistics, json, ...)")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _blocked_open(*a, **k):
    raise PermissionError("file access is disabled in the sandbox: use the preloaded DataFrames (see the table list)")


def _make_safe_builtins() -> dict:
    b = {k: v for k, v in vars(builtins).items() if k not in {"input", "breakpoint", "exit", "quit", "help", "open", "__import__"}}
    b["__import__"] = _safe_import
    b["open"] = _blocked_open
    return b


# --------------------------------------------------------------------------- sql helper
def _make_sql(tables: dict):
    state: dict = {"con": None, "kind": None}

    def sql(query: str):
        """Run a SQL query (DuckDB dialect) over ALL tables by name and return a pandas DataFrame."""
        if state["con"] is None:
            try:
                import duckdb

                con = duckdb.connect()
                for n, df in tables.items():
                    con.register(n, df)
                state["con"], state["kind"] = con, "duckdb"
            except ImportError:
                import sqlite3

                con = sqlite3.connect(":memory:")
                for n, df in tables.items():
                    df.to_sql(n, con, index=False)
                state["con"], state["kind"] = con, "sqlite"
        if state["kind"] == "duckdb":
            return state["con"].execute(query).df()
        import pandas as pd

        return pd.read_sql_query(query, state["con"])

    return sql


# --------------------------------------------------------------------------- child execution
def _format_error(e: BaseException, code: str) -> str:
    lines = code.splitlines()
    msg = f"{type(e).__name__}: {e}"
    for fr in reversed(traceback.extract_tb(e.__traceback__)):
        if fr.filename == "<llm_code>":
            src = lines[fr.lineno - 1].strip() if 0 < fr.lineno <= len(lines) else ""
            msg += f"\n  at line {fr.lineno}: {src}"
            break
    return msg[:1500]


def _exec_child(base_ns, lunar, code, as_of_iso, w):
    from .answer_format import sanitize_answer
    from .helpers import build_helpers

    res = {"ok": False, "answer_type": None, "answer": None, "reason": None, "stdout": "", "error": None}
    buf = io.StringIO()
    try:
        as_of = dt.date.fromisoformat(as_of_iso)
        ns = dict(base_ns)
        ns.update(build_helpers(as_of, lunar))
        ns.update({"as_of": as_of, "AS_OF": as_of_iso, "dt": dt, "sql": _make_sql(base_ns["dfs"]),
                   "__builtins__": _make_safe_builtins(), "__name__": "__sandbox__"})

        def declare_unanswerable(reason: str = ""):
            """Declare that the data cannot support this question."""
            ns["answer_type"], ns["answer"], ns["answer_reason"] = "unanswerable", None, str(reason)

        ns["declare_unanswerable"] = declare_unanswerable
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<llm_code>", "exec"), ns)
        atype = ns.get("answer_type")
        if atype is None:
            raise ValueError("variables `answer_type` and `answer` were not set")
        if atype == "unanswerable":
            res.update(ok=True, answer_type="unanswerable", reason=ns.get("answer_reason", ""))
        else:
            if "answer" not in ns:
                raise ValueError("variable `answer` was not set")
            val = sanitize_answer(str(atype), ns["answer"])
            res.update(ok=True, answer_type=str(atype), answer=val)
    except BaseException as e:  # noqa: BLE001 - we must report anything the snippet raises
        res["error"] = _format_error(e, code)
    res["stdout"] = buf.getvalue()[:MAX_STDOUT]
    try:
        w.send(json.dumps(res, ensure_ascii=False, default=str))
    finally:
        w.close()


# --------------------------------------------------------------------------- worker process
def _worker_main(conn):
    import pandas as pd
    import numpy as np

    from .data_loader import load_business, read_csv_robust
    from .helpers import LunarCalendar
    from .profile import profile_tables

    state: dict = {"ns": None, "lunar": None}
    fork = mp.get_context("fork")
    cal_cache: dict = {}
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        cmd = msg[0]
        try:
            if cmd == "close":
                break
            if cmd == "ping":
                conn.send(("ok", "pong"))
            elif cmd == "load":
                a = msg[1]
                biz = load_business(a["business_dir"], a.get("business_id"))
                rulebook = ""
                if a.get("rulebook_path"):
                    try:
                        with open(a["rulebook_path"], encoding="utf-8-sig", errors="replace") as f:
                            rulebook = f.read()
                    except OSError:
                        rulebook = ""
                cal_df, lunar = None, None
                cp = a.get("calendar_path")
                if cp:
                    if cp not in cal_cache:
                        try:
                            cdf = read_csv_robust(__import__("pathlib").Path(cp))
                            try:
                                lun = LunarCalendar(cdf)
                            except Exception:
                                lun = None
                            cal_cache[cp] = (cdf, lun)
                        except Exception:
                            cal_cache[cp] = (None, None)
                    cal_df, lunar = cal_cache[cp]
                ns = {"pd": pd, "np": np, "dfs": dict(biz.tables), "rulebook": rulebook, "calendar_df": cal_df, "readme": biz.readme}
                for n, df in biz.tables.items():
                    if n not in RESERVED_NAMES:
                        ns[n] = df
                state["ns"], state["lunar"] = ns, lunar
                prof = profile_tables(biz.tables, max_chars=int(a.get("profile_max_chars", 7000)))
                info = {"profile": prof, "tables": {n: [len(df), [str(c) for c in df.columns]] for n, df in biz.tables.items()},
                        "readme": biz.readme, "warnings": biz.warnings, "lunar_ok": lunar is not None}
                conn.send(("ok", info))
            elif cmd == "run":
                _, code, as_of_iso, timeout = msg
                if state["ns"] is None:
                    conn.send(("ok", {"ok": False, "error": "no business loaded", "stdout": ""}))
                    continue
                r, w = fork.Pipe(duplex=False)
                p = fork.Process(target=_exec_child, args=(state["ns"], state["lunar"], code, as_of_iso, w))
                t0 = time.time()
                p.start()
                w.close()
                out = None
                if r.poll(timeout):
                    try:
                        out = json.loads(r.recv())
                    except (EOFError, OSError):
                        out = {"ok": False, "error": "execution process crashed (possibly out of memory)", "stdout": ""}
                else:
                    out = {"ok": False, "error": f"TimeoutError: code exceeded {timeout}s - use vectorised pandas / SQL instead of row loops", "stdout": ""}
                if p.is_alive():
                    p.kill()
                p.join(5)
                r.close()
                out["elapsed"] = round(time.time() - t0, 3)
                conn.send(("ok", out))
            else:
                conn.send(("err", f"unknown command {cmd}"))
        except BaseException:  # noqa: BLE001
            try:
                conn.send(("err", traceback.format_exc()[-2000:]))
            except Exception:
                break


# --------------------------------------------------------------------------- client
@contextlib.contextmanager
def _safe_main():
    """`spawn` re-imports __main__ from its file; hide a non-existent one (e.g. '<stdin>')."""
    import os
    import sys

    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    hide = f is not None and not os.path.exists(f)
    if hide:
        saved = main.__file__
        del main.__file__
    try:
        yield
    finally:
        if hide:
            main.__file__ = saved


class SandboxClient:
    """Parent-side handle. Create it before loading the LLM."""

    def __init__(self, startup_timeout: float = 120.0):
        self._ctx = mp.get_context("spawn")
        self._start(startup_timeout)

    def _start(self, startup_timeout: float):
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_worker_main, args=(child,))
        with _safe_main():
            self._proc.start()
        child.close()
        self._call(("ping",), startup_timeout)
        self._last_load = None

    def _call(self, msg, timeout: float):
        self._conn.send(msg)
        if not self._conn.poll(timeout):
            raise SandboxError(f"sandbox worker did not answer within {timeout}s")
        status, payload = self._conn.recv()
        if status != "ok":
            raise SandboxError(str(payload))
        return payload

    def load(self, business_dir, business_id=None, rulebook_path=None, calendar_path=None, profile_max_chars=7000) -> LoadInfo:
        args = {"business_dir": str(business_dir), "business_id": business_id,
                "rulebook_path": str(rulebook_path) if rulebook_path else None,
                "calendar_path": str(calendar_path) if calendar_path else None,
                "profile_max_chars": profile_max_chars}
        self._last_load = args
        d = self._call(("load", args), 600)
        return LoadInfo(**d)

    def run(self, code: str, as_of: str, timeout: float = 60.0) -> ExecResult:
        try:
            d = self._call(("run", code, as_of, timeout), timeout + 30)
        except (SandboxError, EOFError, OSError, BrokenPipeError) as e:
            self._recover()
            return ExecResult(ok=False, error=f"sandbox worker failure: {e}")
        return ExecResult(ok=bool(d.get("ok")), answer_type=d.get("answer_type"), answer=d.get("answer"), reason=d.get("reason"),
                          stdout=d.get("stdout", ""), error=d.get("error"), elapsed=float(d.get("elapsed", 0.0)))

    def _recover(self):
        """Restart a dead/stuck worker and reload the last business."""
        last = self._last_load
        try:
            self._proc.kill()
        except Exception:
            pass
        self._start(120.0)
        if last:
            self._last_load = last
            self._call(("load", last), 600)

    def close(self):
        try:
            self._conn.send(("close",))
        except Exception:
            pass
        try:
            self._proc.join(10)
            if self._proc.is_alive():
                self._proc.kill()
        except Exception:
            pass
