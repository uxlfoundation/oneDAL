#!/usr/bin/env python3
# ******************************************************************************
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
# ******************************************************************************

"""Whole-product (bundle) ABI gate for oneDAL — capture and gate halves.

oneDAL is one header tree across six interdependent shared objects. abicheck's
ADR-023 bundle layer is built for that shape and names oneDAL as its motivating
case, but its stored-baseline consumer is reachable only from Python and, as
written, forwards no ``CompileContext`` (defaulting to castxml, which cannot
parse this clang/icpx-only toolchain) and applies one ``headers``/``includes``
set uniformly to every library. So this module drives the same Tier-2
chokepoints (``service.resolve_input`` / ``service.compare_snapshots`` /
``bundle_facts.compare_bundle_from_facts``) with per-library scoping instead.
Full rationale, and what would make this file deletable:
``.github/abicheck/README.md``.

Two subcommands, sharing the one ``LIBRARIES`` table in
``onedal_libraries.py`` so the two sides cannot drift — a baseline captured from
a different header set than the gate uses reports every header-only difference
as a spurious add/remove:

``capture``  dump one header-scoped snapshot per library and pack them into a
             single ``BundleFacts`` document (one release asset: 13.2 MiB
             compressed against 902 MiB of raw JSON, so the ``.json.zst``
             suffix that selects compression is load-bearing).
``gate``     resolve each library's new side with its own header roots, diff it
             against the stored snapshot under ``policy.yaml``, fold the six
             diffs into one bundle comparison, exit on the bundle verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

# Resolves because Python puts a script's own directory on sys.path, so
# `python3 .github/abicheck/bundle_gate.py` works from any cwd.
from onedal_libraries import LIBRARIES, SYSTEM_PROVIDERS

log = logging.getLogger("onedal.bundle_gate")


def _peak_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _compile_context(options: list[str], compiler: str):
    from abicheck.compile_context import CompileContext

    return CompileContext(
        gcc_path=compiler,
        gcc_option_tokens=tuple(options),
        frontend="clang",
    )


def _resolve(lib_path: Path, spec: dict, src: Path, version: str, compiler: str):
    """Dump one library's snapshot with that library's own scoping."""
    from abicheck import service

    headers = [src / h for h in spec["headers"]] or None
    includes = [src / i for i in spec["includes"]] or None
    t0 = time.time()
    snap = service.resolve_input(
        lib_path,
        headers=headers,
        includes=includes,
        version=version,
        lang="c++",
        # Not resolve_input's own bare-Python default of True. Both sides here
        # come from this one table, so any consistent value would compare, but
        # False is what a dependency-scoped `compare --bundle-facts-out` file
        # carries -- keeping it means a facts file from the ordinary CLI flow
        # stays comparable instead of raising ScopeMismatchError.
        include_dependencies=False,
        compile=_compile_context(spec["options"], compiler) if headers else None,
    )
    log.info(
        "resolved %s headers=%d in %.1fs (peak %d MB)",
        lib_path.name,
        len(headers or []),
        time.time() - t0,
        _peak_mb(),
    )
    return snap


def _match_map(lib_dir: Path) -> dict[str, Path]:
    from abicheck.cli_helpers_compare import _build_match_map
    from abicheck.package import discover_shared_libraries

    found = discover_shared_libraries(lib_dir, include_private=False)
    matched, warnings = _build_match_map(found)
    # The .so/.so.4/.so.4.0 symlink chain makes every oneDAL library
    # "ambiguous"; version-aware resolution picks the real file. Logged rather
    # than silenced so a genuine two-version directory is still visible.
    for warning in warnings:
        log.debug("match: %s", warning)
    return matched


def _selected(only: str | None) -> dict[str, dict[str, list[str]]]:
    """The library table, optionally narrowed for local debugging.

    A narrowed capture is not a valid baseline (the bundle graph would be
    missing real intra-bundle edges), so ``capture`` says so out loud rather
    than writing a quietly incomplete asset.
    """
    if not only:
        return LIBRARIES
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    if unknown := [name for name in wanted if name not in LIBRARIES]:
        raise SystemExit(f"--only names no known library: {', '.join(unknown)}")
    return {name: LIBRARIES[name] for name in wanted}


def _missing(matched: dict[str, Path], table: dict) -> list[str]:
    return sorted(set(table) - set(matched))


def cmd_capture(args: argparse.Namespace) -> int:
    from abicheck.bundle_facts import capture_bundle_facts
    from abicheck.serialization import save_bundle_facts

    src, lib_dir = Path(args.source).resolve(), Path(args.lib_dir).resolve()
    table = _selected(args.only)
    if table is not LIBRARIES:
        log.warning("--only: capturing %d of %d libraries; NOT a valid baseline",
                    len(table), len(LIBRARIES))
    matched = _match_map(lib_dir)
    if missing := _missing(matched, table):
        log.error("no library in %s matched: %s", lib_dir, ", ".join(missing))
        return 1

    snapshots, paths = {}, {}
    for key, spec in table.items():
        snapshots[key] = _resolve(matched[key], spec, src, args.version, args.compiler)
        paths[key] = matched[key]

    facts = capture_bundle_facts(snapshots, library_paths=paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle_facts(facts, out)
    log.info("wrote %s (%d bytes, %d libraries)", out, out.stat().st_size, len(snapshots))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from abicheck import service
    from abicheck.bundle import build_bundle_snapshot
    from abicheck.bundle_facts import compare_bundle_from_facts
    from abicheck.bundle_models import BundleSignatureEvidence
    from abicheck.serialization import load_bundle_facts

    src, lib_dir = Path(args.source).resolve(), Path(args.lib_dir).resolve()
    table = _selected(args.only)
    facts = load_bundle_facts(Path(args.facts))
    matched = _match_map(lib_dir)

    _, policy_file = service.load_suppression_and_policy(
        None, policy_file_path=Path(args.policy) if args.policy else None
    )

    per_library, evidence = [], {}
    for key, old_snapshot in facts.per_library_snapshots.items():
        new_path = matched.get(key)
        if new_path is None:
            log.warning("%s is in the baseline but not in %s; not diffed", key, lib_dir)
            continue
        spec = table.get(key)
        if spec is None:
            log.warning("%s is in the baseline but not in the library table; not diffed", key)
            continue
        new_snapshot = _resolve(new_path, spec, src, args.version, args.compiler)
        per_library.append(
            service.compare_snapshots(old_snapshot, new_snapshot, policy_file=policy_file)
        )
        evidence[key] = BundleSignatureEvidence.from_snapshot(new_snapshot)

    result = compare_bundle_from_facts(
        facts,
        build_bundle_snapshot(matched),
        per_library,
        system_providers=SYSTEM_PROVIDERS,
        new_signature_evidence=evidence,
    )
    return _report(result, facts, matched, args)


def _verdict(value) -> str:
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def _canonical(name: str) -> str:
    """``libonedal.so.4.0`` -> ``libonedal.so``.

    ``DiffResult.library`` carries the versioned filename off disk while
    ``LIBRARIES`` and ``BundleFacts.per_library_snapshots`` are keyed by the
    soname stem, so reporting the raw name makes every row look like a table
    miss -- i.e. header-less, i.e. "ELF-only".
    """
    base = name
    while True:
        stem, _, suffix = base.rpartition(".")
        if not stem or not suffix.isdigit():
            return base
        base = stem


def _kinds(items) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = _verdict(item.kind)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _summarize(result, facts, matched) -> dict:
    libraries = [
        {
            "library": _canonical(diff.library),
            "artifact": diff.library,
            "verdict": _verdict(diff.verdict),
            "changes": len(diff.changes),
            "change_kinds": _kinds(diff.changes),
            # False means --scope-public-headers was requested but the public
            # surface could not be resolved, so the diff fell back to the whole
            # export table: compatibility is unconfirmed, not confirmed clean.
            "scope_resolved": bool(diff.scope_resolved),
        }
        for diff in result.per_library
    ]

    return {
        "bundle_verdict": _verdict(result.bundle_verdict),
        "per_library_verdict": _verdict(result.per_library_verdict),
        "aggregate_verdict": _verdict(result.verdict),
        "bundle_findings": len(result.bundle_findings),
        "bundle_finding_kinds": _kinds(result.bundle_findings),
        # An empty list here is a positive claim that bundle analysis completed
        # cleanly (abicheck G38 Phase 11), not merely "nothing to say" -- a
        # non-empty one means the verdict above is from a partial analysis and
        # must not read as a clean run.
        "analysis_errors": list(result.analysis_errors),
        "libraries": sorted(libraries, key=lambda entry: entry["library"]),
        "baseline_libraries": sorted(facts.per_library_snapshots),
        "not_in_baseline": sorted(set(matched) - set(facts.per_library_snapshots)),
        "peak_rss_mb": _peak_mb(),
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _markdown(result, summary) -> str:
    """The report text the three per-library `scan` steps used to print."""
    from abicheck.bundle import render_bundle_findings_markdown

    lines = [
        "## oneDAL whole-product ABI gate",
        "",
        f"Bundle verdict: **{summary['bundle_verdict']}** — "
        f"worst per-library {summary['per_library_verdict']}, "
        f"aggregate {summary['aggregate_verdict']}.",
        "",
    ]
    if summary["analysis_errors"]:
        lines += [
            "> [!WARNING]",
            "> Bundle analysis degraded — the verdict above is from a *partial*",
            "> analysis and must not be read as a clean run:",
            *(f"> - {err}" for err in summary["analysis_errors"]),
            "",
        ]
    lines += [
        "| Library | Verdict | Findings | Public surface |",
        "| --- | --- | --- | --- |",
    ]
    for entry in summary["libraries"]:
        surface = "scoped" if entry["scope_resolved"] else "**UNRESOLVED**"
        if not LIBRARIES.get(entry["library"], {}).get("headers"):
            surface = "ELF-only"
        lines.append(
            f"| `{entry['library']}` | {entry['verdict']} "
            f"| {entry['changes']} | {surface} |"
        )

    lines += ["", f"### Cross-library findings ({summary['bundle_findings']})", ""]
    lines += render_bundle_findings_markdown(result.bundle_findings) or [
        "None — every intra-bundle dependency edge resolved.",
    ]
    return "\n".join(lines)


def _repo_relative(uri: str, source: Path, cache: dict[str, str]) -> str:
    """Map one absolute header path onto the file it is in *this* checkout.

    Longest-suffix match, not a prefix strip: the two sides recorded their paths
    under different roots (the workspace, and whatever the baseline was captured
    from), and both have to land on the same repo-relative path or code scanning
    treats one finding's two locations as two files. A path matching nothing --
    a header this branch deleted -- is left as it was, since inventing a path
    for a file that is not there is worse than an unresolvable one.
    """
    if uri in cache:
        return cache[uri]
    parts = Path(uri).parts
    resolved = uri
    for start in range(1, len(parts)):
        candidate = Path(*parts[start:])
        if (source / candidate).exists():
            resolved = candidate.as_posix()
            break
    cache[uri] = resolved
    return resolved


def _relativize_uris(node, source: Path, cache: dict[str, str]) -> None:
    """Rewrite every absolute ``uri`` under ``node`` to a repo-relative one.

    Walks the whole run rather than just ``locations`` so ``relatedLocations``,
    ``fixes`` and anything else carrying an ``artifactLocation`` are covered too.
    """
    if isinstance(node, dict):
        uri = node.get("uri")
        if isinstance(uri, str) and uri.startswith("/"):
            node["uri"] = _repo_relative(uri, source, cache)
        for value in node.values():
            _relativize_uris(value, source, cache)
    elif isinstance(node, list):
        for value in node:
            _relativize_uris(value, source, cache)


def _finalize_sarif_run(
    run: dict, label: str, source: Path, cache: dict[str, str]
) -> None:
    """Fix the two things ``to_sarif`` cannot know about code scanning.

    The automation id abicheck writes embeds both version strings, and CI passes
    ``--version $GITHUB_SHA``, so it would change every commit. GitHub keys an
    analysis on that id (everything before its last ``/`` is the category), so a
    per-commit id makes every run a new category whose predecessor's alerts are
    never correlated nor marked fixed; ``upload-sarif``'s ``category:`` input
    cannot fix it, since it only fills a *missing* id. Paths are absolute because
    that is what the snapshot recorded, and ``upload-sarif`` relativizes them
    only while this gate runs from the workspace root.
    """
    run.setdefault("automationDetails", {})["id"] = f"abicheck/{label}/"
    _relativize_uris(run, source, cache)


def _sarif(result, source: Path) -> str:
    """One SARIF document, one run per library plus one for the bundle layer.

    Bundle findings have no SARIF renderer of their own, but ``BundleFinding.
    to_change()`` is what the CLI already folds into the verdict, so they ride
    in a synthetic ``bundle`` run rather than being dropped from code scanning.
    """
    from abicheck.checker_types import DiffResult
    from abicheck.sarif import to_sarif

    documents = [
        (_canonical(diff.library), to_sarif(diff)) for diff in result.per_library
    ]
    if result.bundle_findings:
        documents.append(
            (
                "bundle",
                to_sarif(
                    DiffResult(
                        old_version="baseline",
                        new_version="new",
                        library="bundle",
                        changes=[f.to_change() for f in result.bundle_findings],
                        verdict=result.bundle_verdict,
                        policy=result.policy,
                    )
                ),
            )
        )
    if not documents:
        raise SystemExit("nothing was diffed; refusing to write an empty SARIF report")
    cache: dict[str, str] = {}
    merged = dict(documents[0][1])
    merged["runs"] = []
    for label, document in documents:
        for run in document.get("runs", []):
            _finalize_sarif_run(run, label, source, cache)
            merged["runs"].append(run)
    return json.dumps(merged, indent=2)


def _report(result, facts, matched, args) -> int:
    summary = _summarize(result, facts, matched)
    text = json.dumps(summary, indent=2, sort_keys=False)
    print(text)
    if args.output:
        _write(Path(args.output), text)
    if args.report_md:
        _write(Path(args.report_md), _markdown(result, summary))
    if args.sarif:
        _write(Path(args.sarif), _sarif(result, Path(args.source)))

    if summary["analysis_errors"]:
        log.error("bundle analysis degraded: %s", "; ".join(summary["analysis_errors"]))
        return 3
    # Gate on the bundle verdict alone. The aggregate verdict folds in the
    # per-library diffs, which the three per-library scans already own; this
    # job exists for the cross-DSO findings no per-library scan can see.
    if summary["bundle_verdict"] == "BREAKING":
        log.error("bundle verdict is BREAKING")
        return 4
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("capture", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True, help="oneDAL source tree root")
        p.add_argument("--lib-dir", required=True, help="directory of built .so files")
        p.add_argument("--version", default="", help="version label for the snapshots")
        p.add_argument("--compiler", default="icpx")
        p.add_argument("--verbose", action="store_true")
        p.add_argument(
            "--only",
            default=None,
            help="comma-separated library subset, for local debugging only",
        )
        if name == "capture":
            p.add_argument("--output", required=True, help="BundleFacts path to write")
        else:
            p.add_argument("--facts", required=True, help="baseline BundleFacts path")
            p.add_argument("--policy", default=None, help="policy.yaml to apply")
            p.add_argument("--output", default=None, help="write the JSON summary here")
            p.add_argument("--report-md", default=None, help="write the report text here")
            p.add_argument("--sarif", default=None, help="write a merged SARIF report here")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    return cmd_capture(args) if args.command == "capture" else cmd_gate(args)


if __name__ == "__main__":
    sys.exit(main())
