#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

"""diff_runs.py OLD_OUT NEW_OUT [--md PATH] [--json PATH] [--cap N]

Compare two usage-map output directories (out-<build>/ written by report2.py and
candidates.py) and report what changed: function class shifts, functions added or
removed (B and E flagged), component and file class-count changes, unit runtime and
rc changes, and removal-candidate tier changes (candidates.json).

Stdlib only. Reads data.json, functions.csv and candidates.json; writes nothing
except the --md and --json paths.

Exit status: 0 no differences, 1 differences found, 2 missing or invalid input.
A missing candidates.json skips the tier comparison with a warning.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

CLASSES = ("A", "P", "C", "B", "E")
REQ_DATA = {"build", "groups", "real", "public", "filtered", "units", "components", "files"}
REQ_COMP = {"api", "comp", "lines", *CLASSES}
REQ_FILE = {"api", "comp", "sub", "lines", "file", *CLASSES}
REQ_UNIT = {"uid", "consumer", "seconds", "rc"}
REQ_CAND = {"tier", "api", "component", "sub", "removable"}
REQ_FN_COLS = ["class", "abi_exported", "api", "component", "sub", "file", "line",
               "instances", "function", "groups"]
SUPPORTED_SCHEMA = {1}          # data.json "schema_version"; absent = pre-versioning output, warned


class InputError(Exception):
    pass


def _rows_ok(name, rows, req, path):
    for i, r in enumerate(rows):
        miss = req - r.keys()
        if miss:
            raise InputError(f"{path}: {name}[{i}] missing {sorted(miss)}")


def load_candidates(out_dir, warnings):
    """(api, component, sub) -> row of candidates.json, or None when the file is absent."""
    p = os.path.join(out_dir, "candidates.json")
    if not os.path.isfile(p):
        warnings.append(f"{p}: missing, tier comparison skipped (run candidates.py)")
        return None
    try:
        with open(p) as fh:
            rows = json.load(fh)
    except json.JSONDecodeError as e:
        raise InputError(f"{p}: {e}")
    _rows_ok("candidates", rows, REQ_CAND, p)
    return {(r["api"], r["component"], r["sub"]): r for r in rows}


def load_run(out_dir, warnings):
    """Return dict with data, functions {(file, function, occurrence): (cls, line, abi)}, candidates."""
    dpath = os.path.join(out_dir, "data.json")
    fpath = os.path.join(out_dir, "functions.csv")
    for p in (dpath, fpath):
        if not os.path.isfile(p):
            raise InputError(f"{p}: missing")
    try:
        with open(dpath) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        raise InputError(f"{dpath}: {e}")
    miss = REQ_DATA - data.keys()
    if miss:
        raise InputError(f"{dpath}: missing keys {sorted(miss)}")
    sv = data.get("schema_version")
    if sv is None:
        warnings.append(f"{dpath}: no schema_version (pre-versioning output); "
                        "report2.py should write one")
    elif sv not in SUPPORTED_SCHEMA:
        raise InputError(f"{dpath}: schema_version {sv!r} not supported")
    _rows_ok("components", data["components"], REQ_COMP, dpath)
    _rows_ok("files", data["files"], REQ_FILE, dpath)
    _rows_ok("units", data["units"], REQ_UNIT, dpath)

    funcs = {}
    seen = Counter()
    with open(fpath, newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd, None)
        if hdr != REQ_FN_COLS:
            raise InputError(f"{fpath}: header {hdr} != {REQ_FN_COLS}")
        for row in rd:
            r = dict(zip(hdr, row))
            k = (r["file"], r["function"])
            seen[k] += 1
            funcs[(*k, seen[k])] = (r["class"], int(r["line"]), r["abi_exported"] == "1")
    return {"data": data, "functions": funcs, "candidates": load_candidates(out_dir, warnings)}


def diff(old, new):
    od, nd = old["data"], new["data"]
    out = {}

    # functions: occurrence-indexed key (file, demangled name, n); line numbers are not part of the key
    of, nf = old["functions"], new["functions"]
    shifts, added, removed = [], [], []
    for k in sorted(of.keys() & nf.keys()):
        oc, nc = of[k][0], nf[k][0]
        if oc != nc:
            shifts.append({"file": k[0], "function": k[1], "from": oc, "to": nc,
                           "abi_exported": nf[k][2]})
    for k in sorted(nf.keys() - of.keys()):
        added.append({"file": k[0], "function": k[1], "class": nf[k][0], "abi_exported": nf[k][2]})
    for k in sorted(of.keys() - nf.keys()):
        removed.append({"file": k[0], "function": k[1], "class": of[k][0], "abi_exported": of[k][2]})
    out["function_shifts"] = shifts
    out["functions_added"] = added
    out["functions_removed"] = removed

    # components and files: class counts
    def count_diff(oldrows, newrows, keyf):
        o = {keyf(r): r for r in oldrows}
        n = {keyf(r): r for r in newrows}
        changed, add, rem = [], [], []
        for k in sorted(o.keys() & n.keys()):
            delta = {c: n[k][c] - o[k][c] for c in CLASSES if n[k][c] != o[k][c]}
            if delta or n[k]["lines"] != o[k]["lines"]:
                changed.append({"key": k, "delta": delta, "lines": [o[k]["lines"], n[k]["lines"]]})
        add = [{"key": k, **{c: n[k][c] for c in CLASSES}} for k in sorted(n.keys() - o.keys())]
        rem = [{"key": k, **{c: o[k][c] for c in CLASSES}} for k in sorted(o.keys() - n.keys())]
        return changed, add, rem

    out["components_changed"], out["components_added"], out["components_removed"] = count_diff(
        od["components"], nd["components"], lambda r: f"{r['api']}/{r['comp']}")
    out["files_changed"], out["files_added"], out["files_removed"] = count_diff(
        od["files"], nd["files"], lambda r: r["file"])

    # removal candidates: tier changes per (api, component, sub); skipped without candidates.json
    oc, nc = old["candidates"], new["candidates"]
    t_changed, t_added, t_removed = [], [], []
    if oc is not None and nc is not None:
        for k in sorted(oc.keys() & nc.keys()):
            if oc[k]["tier"] != nc[k]["tier"]:
                t_changed.append({"key": "/".join(k), "from": oc[k]["tier"], "to": nc[k]["tier"],
                                  "removable": [oc[k]["removable"], nc[k]["removable"]]})
        t_added = [{"key": "/".join(k), "tier": nc[k]["tier"]} for k in sorted(nc.keys() - oc.keys())]
        t_removed = [{"key": "/".join(k), "tier": oc[k]["tier"]} for k in sorted(oc.keys() - nc.keys())]
    out["candidates_tier_changed"] = t_changed
    out["candidates_added"] = t_added
    out["candidates_removed"] = t_removed

    # units: runtime change beyond noise (abs > 1 s and > 20 %), rc change, add/remove
    ou = {u["uid"]: u for u in od["units"]}
    nu = {u["uid"]: u for u in nd["units"]}
    u_sec, u_rc = [], []
    for k in sorted(ou.keys() & nu.keys()):
        a, b = ou[k]["seconds"], nu[k]["seconds"]
        if abs(a - b) > max(1.0, 0.2 * a):
            u_sec.append({"uid": k, "old": a, "new": b})
        if ou[k]["rc"] != nu[k]["rc"]:
            u_rc.append({"uid": k, "old": ou[k]["rc"], "new": nu[k]["rc"]})
    out["units_seconds_changed"] = u_sec
    out["units_rc_changed"] = u_rc
    out["units_added"] = sorted(nu.keys() - ou.keys())
    out["units_removed"] = sorted(ou.keys() - nu.keys())
    return out


def has_changes(d):
    return any(d[k] for k in d)


def render_md(d, old_dir, new_dir, cap):
    L = ["# usage-map run diff", "", f"- old: `{old_dir}`", f"- new: `{new_dir}`", ""]
    L.append("| section | count |")
    L.append("|---|---|")
    for k in d:
        L.append(f"| {k} | {len(d[k])} |")
    L.append("")

    def sec(title, rows, fmt):
        if not rows:
            return
        L.append(f"## {title} ({len(rows)})")
        L.append("")
        for r in rows[:cap]:
            L.append(fmt(r))
        if len(rows) > cap:
            L.append(f"... {len(rows) - cap} more (see --json)")
        L.append("")

    sec("Function class shifts", d["function_shifts"],
        lambda r: f"- `{r['file']}`: {r['from']} -> {r['to']}"
                  + (" (ABI-exported)" if r["abi_exported"] else "")
                  + f" `{r['function'][:120]}`")
    sec("Functions added", d["functions_added"],
        lambda r: f"- {r['class']}{' **new B/E**' if r['class'] in ('B', 'E') else ''} "
                  f"`{r['file']}` `{r['function'][:120]}`")
    sec("Functions removed", d["functions_removed"],
        lambda r: f"- {r['class']} `{r['file']}` `{r['function'][:120]}`")
    sec("Components with changed counts", d["components_changed"],
        lambda r: f"- `{r['key']}` delta {r['delta']} lines {r['lines'][0]} -> {r['lines'][1]}")
    sec("Components added", d["components_added"], lambda r: f"- `{r['key']}`")
    sec("Components removed", d["components_removed"], lambda r: f"- `{r['key']}`")
    sec("Files with changed counts", d["files_changed"],
        lambda r: f"- `{r['key']}` delta {r['delta']}")
    sec("Files added", d["files_added"], lambda r: f"- `{r['key']}`")
    sec("Files removed", d["files_removed"], lambda r: f"- `{r['key']}`")
    sec("Removal candidates changed tier", d["candidates_tier_changed"],
        lambda r: f"- `{r['key']}` {r['from']} -> {r['to']} (removable {r['removable'][0]} -> {r['removable'][1]})")
    sec("Removal candidates added", d["candidates_added"], lambda r: f"- `{r['key']}` {r['tier']}")
    sec("Removal candidates removed", d["candidates_removed"], lambda r: f"- `{r['key']}` {r['tier']}")
    sec("Unit seconds changed", d["units_seconds_changed"],
        lambda r: f"- `{r['uid']}` {r['old']} -> {r['new']}")
    sec("Unit rc changed", d["units_rc_changed"],
        lambda r: f"- `{r['uid']}` rc {r['old']} -> {r['new']}")
    sec("Units added", d["units_added"], lambda r: f"- `{r}`")
    sec("Units removed", d["units_removed"], lambda r: f"- `{r}`")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--md", help="write markdown report here (default: stdout)")
    ap.add_argument("--json", help="write full diff as JSON here")
    ap.add_argument("--cap", type=int, default=50, help="rows per section in markdown")
    a = ap.parse_args(argv)

    warnings = []
    try:
        old = load_run(a.old, warnings)
        new = load_run(a.new, warnings)
    except InputError as e:
        print(f"diff_runs: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"diff_runs: warning: {w}", file=sys.stderr)

    d = diff(old, new)
    md = render_md(d, a.old, a.new, a.cap)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
    return 1 if has_changes(d) else 0


if __name__ == "__main__":
    sys.exit(main())
