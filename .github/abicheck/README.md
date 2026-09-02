<!-- file: README.md
******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# oneDAL ABI checks

oneDAL's `CI` workflow runs **two independent ABI gates**, in parallel, both
depending only on `LinuxMakeDPCPP` and never on each other:

| | `ABI Conformance(avx2)` | `Abicheck (multilib)` |
|---|---|---|
| job | `LinuxABICheck` | `LinuxAbicheckScan` — **one** job, one check step |
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck)'s own Action, `mode: compare` |
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as one abicheck snapshot asset per library, fetched from the release |
| operands | two directories, compared library by library by the script | two directories, compared library by library by abicheck's release fan-out |
| evidence | ELF symbols and type info from the two binaries | ELF exported symbols and metadata from the snapshots and the new binaries |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) |
| cost | 20 min timeout | 12 s, ~205 MiB |

They are complementary, not redundant. `abidiff`'s baseline is `main`, so drift
that accumulates one merged PR at a time across a release cycle never appears in
it; this gate answers the question users actually ask — "can I drop this build in
for the release I have installed?" — and answers it per library, with a verdict
that distinguishes *break* from *risk* under a policy whose every downgrade is
evidence-backed. Its blind spots are equally concrete: it knows only what the
baseline snapshot recorded, it reads no headers, and it needs a published release
asset to compare against at all.

## Where everything lives

| path | what it is |
|---|---|
| `.github/workflows/ci.yml`, job `LinuxAbicheckScan` | the PR-side check: a single `mode: compare` step with two directory operands |
| `.github/workflows/abicheck-baseline.yml` | the baseline publisher: one `mode: dump` step per library |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as the Action's `policy-file` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | digests of the release assets for `<tag>` — the trust anchor, and the only baseline artifact in git |
| `.github/.abignore` | libabigail suppressions, used by the *other* job |

There is no oneDAL-owned Python, no oneDAL-owned driver script and no
oneDAL-owned abicheck config file: both workflows call `abicheck/abicheck` at a
pinned SHA and pass everything as Action inputs.

## What one run compares

Both operands are directories, which is abicheck's own multi-library ("release")
comparison: one process fans out over every shared object in
`__release_lnx/daal/latest/lib/intel64` and matches each to the baseline snapshot
of the same name. Six libraries, one step, one verdict, plus a per-library verdict
table:

| library | verdict | breaking | risk | additions | dominant kinds |
|---|---|---|---|---|---|
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 0 | 254 | 114 | 237 `func_removed_elf_only`, 102 `func_added`, 16 `imported_symbol_added` |
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 1415 | 62 | 1414 `func_removed_elf_only`, 36 `var_added`, 24 `func_added` |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 305 | 128 | 272 `func_removed_elf_only`, 114 `func_added`, 27 `imported_symbol_added` |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 0 | 13 | 2 | 12 `func_removed_elf_only` |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 18 | 2 | 17 `func_removed_elf_only` |
| `libonedal_thread.so` | `COMPATIBLE` | 0 | 0 | 1 | 1 `visibility_leak` |

Measured on `main` against the `2026.0.0` baseline, through the same root Action
the job invokes, with `policy.yaml` in effect: **exit 0**, folded verdict
`COMPATIBLE_WITH_RISK`, 2314 findings in total, nothing removed from the report.
Four runs of the same shape: 11.4–12.2 s, 203–206 MiB peak RSS.

`jobs: 1` is not a tuning choice. The default (`0`) means one worker per CPU, and
six libraries parsed concurrently is where this shape stops fitting a runner.

`extra-args: --no-bundle-analysis` is also load-bearing. The cross-library bundle
graph is built from real ELF binaries; the old side here is six JSON snapshots, so
abicheck has no old bundle to diff against and reports all six libraries as
`bundle_library_added` plus (without an explicit `--bundle-system-providers` list)
255 `bundle_intra_dep_removed`, i.e. `BREAKING`, on an unchanged tree. Two facts
worth keeping for whoever revisits cross-library checking: `DT_NEEDED` shows only
two sibling edges (`libonedal_parameters*.so → libonedal*.so`) — `libonedal_dpc.so`
does not link `libonedal_core.so` despite importing ~800 symbols from it, because
applications link both — and abicheck's `DEFAULT_SYSTEM_PROVIDERS` omits
`libtbbmalloc`, the MKL libraries and the Intel runtime, while **one** unmatched
`DT_NEEDED` edge disables the system-edge exemption for a whole library (measured:
the difference between 862 and 58 `bundle_unresolved_intra_dependency` findings,
and between `BREAKING` and `COMPATIBLE_WITH_RISK`).

## Binary depth

The release fan-out is **binary depth by contract**. abicheck's own
`check-target` reference says a bundle's baseline "is always raw binaries with no
historical header/build/source evidence staged per member"; the CLI rejects
`--depth` outright for a directory operand (exit 64, "the per-library fan-out does
not collect inline build/source evidence"), and the Action drops a `depth:` input
silently on that path rather than forwarding it — which is why `ci.yml` passes
none. `LinuxMakeDPCPP` emits no DWARF at all (`readelf -SW` finds no
`.debug_info` in `libonedal_core.so`), so the default resolves to the same thing
today: a run with no depth pinned produced an identical library table in an
identical 12 s.

What that gives up, in the report's own words — one such line per library:

> Binary-only analysis without debug info; many ABI changes cannot be detected
> (struct layout, enum values, type changes)

Concretely, versus the header-scoped per-library matrix this replaced:

* **no public-header AST.** No `exported_not_public`/`public_not_exported`
  provenance, no declaration-vs-export mismatch, and struct layout, enum values
  and type changes are not compared. The detectors that need those facts announce
  themselves as disabled (`vtable_layout`, `dwarf_layout_coherence`, `sycl`,
  plus the kABI/PE/Mach-O/Python ones that never applied) — nine coverage
  warnings per library, which is most of the report's length.
* **no SARIF, so no code scanning.** `--format sarif` and `html` are rejected for
  a directory operand (exit 64: "sarif/html/review require a single-pair
  comparison"); `json`, `junit` and `markdown` are what remain.
* **no per-symbol detail in the markdown.** A release comparison's markdown is
  per-library counts plus coverage warnings — 7.3 KB, no symbol names anywhere.
  The gate therefore also writes the json report
  (`extra-args: --write json=…`) and archives it: 717 KB, one annotation entry
  per finding, all 2314 of them, naming every symbol. Note the json's own
  `findings` array is capped at 10 per library (`findings_truncated: true`); the
  `annotations` array is not.

The header-scoped alternative was measured, not assumed. A single multilib
comparison at header depth needs the L2 compile context (`-fsycl`,
`-DONEDAL_DATA_PARALLEL`, include roots), and the Action refuses those inputs on
a release operand outright (`::error::` + exit 1, because "the per-library fan-out
never threads the L2 compile context to each pair's header dump"), so it is
reachable only by smuggling the flags through `extra-args` past that guard. Doing
that works — exit 0, `COMPATIBLE_WITH_RISK` — and costs **50 min 54 s and
10.7 GiB peak RSS**, against a 16 GB runner, per PR. Upstream documents multilib
comparison as binary-depth; running it otherwise means depending on a shape
upstream deliberately blocks. Hence this gate is binary depth, and the
header-scoped surface stays the domain of a local run.

## Baselines

The published release binaries are a multi-ISA (`sse2 sse42 avx2 avx512`)
release-mode build while CI compiles a single-ISA `avx2` one, so comparing
against them would compare build configurations rather than ABI changes. Hence
`.github/workflows/abicheck-baseline.yml` ("Publish Abicheck Baseline") checks out
the release **tag** and rebuilds it with the same three targets, ISA and debug
setting the PR-side build uses. It runs on `workflow_dispatch` (with a
`baseline_tag` input) and on `release: published`, and it:

1. builds the tag — `daal`, `oneapi_c`, `oneapi_dpc`, which produce all six
   libraries;
2. runs the abicheck Action in `mode: dump` once per library (`dump` takes one
   library at a time; only `compare` fans a directory out) — measured 0.69 s to
   7.42 s each, 17.5 s together, peak 184 MiB on the largest;
3. checks a compression tripwire: the six snapshots are 5.8 KiB to 148 KiB, 372 KB
   together, so a 5 MiB ceiling catches a snapshot written uncompressed;
4. uploads each `<library>.abicheck.json.zst` as a **release asset**;
5. commits only its `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`, one line per asset.

**The asset name is the matcher.** abicheck keys a snapshot to a shared object by
the filename up to `.so`, so `libonedal_core.so.abicheck.json.zst` pairs with
`libonedal_core.so.4.0` — and a decorated name (a `<tag>-` prefix, say) pairs with
nothing, silently. The `.json.zst` suffix is load-bearing too: abicheck infers the
compression envelope from it when writing and from the bytes when reading.

The snapshots themselves are deliberately **not** in git. The digest file is a few
hundred bytes, is reviewable, and is the trust anchor: `LinuxAbicheckScan`
downloads exactly the asset names it lists and then runs
`sha256sum --check --strict`, so a replaced or corrupted asset fails the job
instead of silently changing every PR's verdict. Publishing the baseline
*binaries* instead — abicheck's own bundle model — was rejected as a 144 MB
per-release product decision, and the 2026.0.0 release carries no assets at all
today.

Published assets are **immutable**: a run that would overwrite one stops, because
every already-merged PR's "compatible with `<tag>`" verdict was computed against
the published bytes. Existence, not a digest compare, is the rule — and has to be,
since a snapshot is content-reproducible but not byte-reproducible (two dumps of
the same `libonedal.so` by the same pin: 87,584 bytes each, different sha256,
differing only in the recorded `created_at`). A re-run keeps every asset already on
the release and re-hashes the published bytes, so it converges instead of
deadlocking.

### Bootstrap

`workflow_dispatch` only becomes available once the workflow file is on the
default branch, so the digest file cannot exist before this lands. Until it does,
`LinuxAbicheckScan` **skips** the gate and says so in a warning annotation and the
job summary — inert and visibly inert, rather than red (which would block the very
PR that delivers the publishing workflow) or quietly green. Once the digest file is
on `main`, the gate arms itself with no further edit.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update
`ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml`.

### The abicheck pin and the baseline must move together

The `uses: abicheck/abicheck@<sha>` pin in `ci.yml` and in
`abicheck-baseline.yml` must name the **same** commit. `uses:` accepts no
expression, so the SHA cannot be shared through an env var and is written out at
each call site; a bump has to touch all of them. A snapshot records a
`schema_version`, and detectors whose evidence postdates it decline to run rather
than trust stale facts, so a baseline dumped by an *older* abicheck than the
reader does not fail — it silently **under-reports**. The reverse direction, a
snapshot newer than the reader, is a hard reject.

The loud failure mode is worse and `schema_version` does not protect against it: a
fix in the **dumper** changes recorded facts without changing the schema, so the
two sides disagree about a value neither side changed. Measured while validating an
earlier bump — an enumerator-value fix made baselines dumped by the previous pin
report **280** breaking `enum_member_value_changed` findings on an *unchanged*
tree, `schema_version` 25 on both sides; re-dumping took all 280 to zero. (That
particular kind cannot recur at binary depth, which collects no enum facts. The
mechanism can, for any fact this depth does collect.)

So bumping the pin obliges re-*verifying* the published baselines: compare the
unchanged tree with the new pin against the current baselines and diff the report
against the old pin's. Both bumps done so far came out identical line-for-line
apart from timings, so neither needed a re-capture. Any difference means
re-dispatching the baseline workflow for every published tag first. If a bump ever
produces a wave of findings in one kind across an unchanged tree, suspect this
before suspecting oneDAL.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is in the report. Nothing is filtered out.
* *Did oneDAL break its ABI* — the verdict, computed after `policy.yaml`
  re-classifies findings by kind and ELF linkage.

The job gates on the folded release verdict with the Action's defaults: a binary
ABI break (`BREAKING`, exit 4) fails it; a source-level API break (exit 2) does
not, since `fail-on-api-break` stays off. `fail-on-removed-library: true` is set —
a library dropped from the release is a break, not a coverage gap.

**`policy.yaml` is load-bearing, and measured to be.** The same comparison with no
`--policy` at all is `BREAKING` on five of the six libraries — 1952 breaking
findings, exit 4. With it: 0 breaking, exit 0, and all 1952 still printed as risk.
Every one of them is `func_removed_elf_only` on a WEAK symbol: the `2026.0.0`
baseline predates `makefile` gaining `-fvisibility-inlines-hidden`, so `main`
stopped *exporting* a large set of COMDAT inline and template symbols that are
still defined as LOCAL FUNC in the new binaries' `.symtab`. The evidence for
tolerating that, and the exact scope of what the two rules give up, is in
`policy.yaml` itself. Both rules should be **deleted** once a
post-`-fvisibility-inlines-hidden` release becomes the baseline.

The policy is accountable rather than a blanket mute, and the linkage scoping is
what makes it so. Negative control on the shipped shape: take one tolerated
symbol in the baseline snapshot — `oneapi::dal::v1::exception::~exception()`, WEAK,
no longer exported — and flip only its recorded linkage to `global`.
`libonedal_parameters.so` moves to `BREAKING` with exactly one breaking finding
(its risk count drops 13 → 12, so that one finding changed bucket and nothing else
did) and the run exits 4. A STRONG export disappearing gates; a WEAK one is
reported and does not.

A **suppression** file was tried first and rejected. A suppression rule removes the
matching change *before* the verdict and the counts are computed, and leaves no
trace in text output — not the finding, not a count, not even a note that a
suppression file was in effect. The failure mode is silence, not noise: a
synthesized STRONG-symbol visibility regression produced a run whose visible
finding set was identical to a clean one's, exit 0.

Five rules that earlier revisions carried are gone with the header-scoped shape
they were measured against (three named types' `type_vtable_changed`,
`oneapi::dal::preview`'s `experimental_removed_without_replacement`, one internal
`constant_changed`): at binary depth no type, layout, enum or constant facts reach
the comparison, so those rules were config that could never fire. Verified by
running both shapes side by side — the two-rule policy produces a library table
identical to the seven-rule one's, and an identical negative-control failure. If
this gate ever regains header depth they have to be re-measured, not restored from
git history.

Two abicheck behaviours are worth knowing before reading a job summary. The
Action's own summary block and its `verdict` output fold a release verdict down to
the plain one — measured, `Verdict: COMPATIBLE` printed for a run whose real
verdict is `COMPATIBLE_WITH_RISK`; the report the job pastes into the summary is
where the real one is. And the per-library rows are keyed by the *snapshot*
filename (`libonedal_core.so.abicheck.json.zst`), not the shared object's.

`require-complete-analysis` is deliberately **not** set: `libonedal_thread.so` has
no installed public header, so its analysis can never be "complete", and the flag
would fail `ANALYSIS_INCOMPLETE` (exit 1) on a library that has nothing wrong with
it.

Until the `risk` bucket has had a burn-in period, treat a change in these counts as
something to read rather than as a regression in itself.

## Why not abicheck's declarative project configuration

abicheck's paved road for a multi-library project (G30/ADR-047) is a
`targets:`/`bundles:`/`profiles:`/`baseline:` block in `.abicheck.yml`, consumed by
the reusable `check-project.yml` and `publish-baseline.yml` workflows. Three of the
four obstacles an earlier, header-scoped revision of this gate documented no longer
apply at binary depth: `public_headers:` never reaching a run-plan cell, `bundles:`
being restricted to `depth: binary`, and `-fsycl` being inexpressible in the
`compile:` block all stop mattering when nothing parses a header.

What remains is the build contract. Both reusable workflows are "build once, scan
many": each expects one `<prefix><profile-id>` artifact per contract profile
containing a `build-output.json` plus the binaries it references (G30 P1.1), and
oneDAL's makefile build emits no such manifest. Producing one means oneDAL-owned
build glue — the class of thing this revision exists to remove — so the two
workflows here keep calling the root Action directly. Everything the paved road
expects to be portable is already in place: release-asset baselines with a
committed digest anchor, a release-triggered (never `pull_request`) publishing
workflow, SHA-pinned Actions, abicheck's own Action rather than a project driver,
and policy over suppression. If oneDAL's build ever emits `build-output.json`,
this collapses into two `workflow_call` jobs.

## Known gaps

* **The old side is never re-parsed, and that is what makes this affordable.** A
  directory-vs-directory comparison with binaries on *both* sides re-derives every
  fact for both; comparing against stored snapshots reads the new side only. Do not
  "simplify" this into a binary-vs-binary directory compare, and do not drop the
  published baselines in favour of building the tag in the PR job.
* **Header-scoped evidence is gone from CI**, with everything that implies:
  declaration-vs-export provenance, layout, enum values and constants are not
  compared at all, and `abidiff` is the only gate reading type information. Getting
  it back inside one multilib run costs 51 min / 10.7 GiB and requires bypassing an
  upstream guard (see [Binary depth](#binary-depth)); a per-library matrix costs
  ~12 min wall and ~33 min of runner time. Neither was judged worth it against a
  12 s check plus `abidiff`.
* **`scan --artifact-set DIR` remains the cheap fallback** if the baseline assets
  are ever unavailable: all six libraries, no old side, in 10.8 s with
  `--depth binary`. Its residual findings are by design, so gate on kinds rather
  than counts if it is ever adopted.
* **The public-header provenance cross-checks stay off** — they need header roots,
  which this shape cannot pass. When they were reachable they were measured as
  volume rather than capability: `libonedal_parameters.so` alone (the smallest, one
  header) yielded 129 findings, 68 of 97 exports undocumented and 61 declarations
  whose export obligation the binary does not satisfy. They are *intra-version*
  observations, not drift since the baseline, so adopting them means a policy pass
  over the reason buckets first.
* **`binding:` is not accepted as a rule's only scope.** A `reclassify:` entry must
  name at least one of `symbol`, `symbol_pattern`, `type_pattern`, `member_name`,
  `source_location`, `namespace` or `finding_id`, so the two linkage-scoped rules
  in `policy.yaml` carry a `symbol_pattern: ".*"` that means nothing beyond
  satisfying the validator. `Change.symbol_binding` *is* stamped on both the removal
  and the visibility branch, so the selector itself works.
* **A per-library options input on `actions/baseline` would collapse the baseline
  workflow's six `mode: dump` steps into one composite call.** It passes no
  per-library configuration to its dump loop, so the six explicit steps stay. Same
  for the pin: `uses:` takes no expression, so the SHA is repeated per step.

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the
rationale. If you change the pinned commit, the baseline storage, the policy or the
shape of the comparison, update this file in the same PR.
