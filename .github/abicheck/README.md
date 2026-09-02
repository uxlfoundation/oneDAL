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

| | `ABI Conformance(avx2)` | `Abicheck (<library>)` |
|---|---|---|
| job | `LinuxABICheck` | `LinuxAbicheckScan`, a six-way matrix (one job per library) |
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck)'s own Action, `mode: compare` |
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as one abicheck snapshot asset per library, fetched from the release |
| evidence | the two binaries only (ELF symbols + whatever DWARF they carry) | this PR's **public-header AST** plus the binary's exported symbols |
| artifacts | every `lib*.so` in the release directory | all six product libraries, one job each |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) |
| timeout | 20 min | 45 min per library |

They are complementary, not redundant, and neither subsumes the other.
`abidiff` never reads a header, so a declaration that ships in `include/` but is
exported by nothing — or an export with no public declaration — is invisible to
it, and because its baseline is `main`, drift that accumulates one merged PR at a
time across a release cycle never appears. abicheck answers the question users
ask instead ("can I drop this build in for the release I have installed?"), and
it reads each library against the headers that actually declare its surface, so
an export with no public declaration is a finding rather than a silent pass. Its
own blind spots: it knows only what the baseline snapshot recorded, at the depth
this repository runs it reasons about layout from header ASTs rather than DWARF,
and it compares one library at a time (see
[What one job per library does not see](#what-one-job-per-library-does-not-see)).

## Where everything lives

| path | what it is |
|---|---|
| `.github/workflows/ci.yml`, job `LinuxAbicheckScan` | the PR-side check: one `mode: compare` Action step per library, driven by the job's matrix |
| `.github/workflows/abicheck-baseline.yml` | the baseline publisher: one `mode: dump` Action step per library |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as the Action's `policy-file` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | digests of the release assets for `<tag>` — the trust anchor, and the only baseline artifact in git |
| `.github/.abignore` | libabigail suppressions, used by the *other* job |

There is no oneDAL-owned Python and no oneDAL-owned abicheck config: both
workflows call `abicheck/abicheck` at a pinned SHA and pass everything as Action
inputs. The scoping table below therefore lives twice, once in each workflow —
duplicated on purpose, because the alternative (a config file, or a script both
read) is what the previous revision of this gate did and what reviewers asked to
remove. The duplication cannot drift silently: the scoping enters the snapshot's
profile fingerprint, so a baseline captured with a different header set, include
root or compiler option makes the comparison fail as `NOT_COMPARABLE` rather
than report a false break.

## Evidence depth — what is actually enabled

abicheck stacks evidence in layers. This repository runs `depth: headers`,
which is **L2**. Observed layer status from a real run of this configuration:

| layer | status here | why |
|---|---|---|
| `L0_binary` | **present** | ELF exported symbols, read from the artifact |
| `L1_debug` | **present** for `libonedal_core.so`, *not collected* for the other five | `LinuxMakeDPCPP` builds without `--debug symbols`, but `icx` emits DWARF for the `daal` target anyway |
| `L2_header` | **present** | public-header AST, parsed with `icpx` |
| `L3_build` | *not collected* | needs real compile flags (`--sources`/`--build-info`) |
| `L4_source_abi` | *not collected* | needs a Clang plugin or a compiler wrapper |
| `L5_source_graph` | **present** | reachability over the declared surface |
| `pattern_scan` | **present** | lexical pre-scan of the changed headers |

Not requiring `L1_debug` is deliberate: L2 needs no DWARF, which is what makes
scanning `libonedal_dpc.so` affordable — a DPC++ library built with device debug
info is far too large to dump on a standard runner, and without it the same
header-AST evidence applies unchanged. Where DWARF happens to be there abicheck
uses it; nothing in the gate depends on that. One consequence: any command
reading the *binary* without a snapshot pays for that DWARF (see
`scan --artifact-set` under [Known gaps](#known-gaps)). L3 and L4 are measured
and deferred, also below.

## Which artifacts are checked, and with which headers

All six product libraries, one CI job each, each with **its own** header roots
and compile flags. The table is written out twice: as `LinuxAbicheckScan`'s
`strategy.matrix.include` in `ci.yml` and as the six `Dump <library>` steps in
`abicheck-baseline.yml`. Keep them equal — the PR-side check compares against the
baseline the other one captured, and the scoping is part of what makes the two
comparable at all.

| library | headers | include roots | extra compile flags |
|---|---|---|---|
| `libonedal_core.so` | 7 `daal` roots | `cpp/daal/include` | `-std=c++17` |
| `libonedal.so` | 7 `oneapi::dal` roots | `cpp` | `-fsycl -std=c++17` |
| `libonedal_dpc.so` | same 7 | `cpp` | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |
| `libonedal_parameters.so` | `detail/parameters/system_parameters.hpp` | `cpp` | `-fsycl -std=c++17` |
| `libonedal_parameters_dpc.so` | same 1 | `cpp` | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |
| `libonedal_thread.so` | none — ELF-only | — | — |

`-DONEDAL_DATA_PARALLEL` makes the `sycl::queue` overloads visible, and those
overloads are exactly the part of the public API only `libonedal_dpc.so`
exports; without it this scan would report the whole SYCL surface as
exported-but-not-declared. Equally, scanning one library against another's
headers would report every declaration as missing — hence per-library roots
rather than one shared list (against the shared tree, every declaration the
*other* libraries implement counts as "declared here but not exported here":
2787 such on `libonedal.so`, 4974 on `libonedal_core.so`, which is why those
cross-checks stay advisory).

Two libraries get less than the others, and only one of those is a real design
boundary. `libonedal_thread.so` (290 exports) has no public header surface at
all — it is the internal C ABI between `libonedal_core.so` and its threading
backend, declared only in the uninstalled `cpp/daal/src/threading/threading.h`
— so it rides in the bundle ELF-only, for its dependency edges. The two
`parameters` libraries (75 and 124 exports) are scoped to
`system_parameters.hpp` only because of an **oneDAL packaging bug**: the other
12 installed `parameters` headers include `oneapi/dal/backend/dispatcher.hpp`
and `oneapi/dal/backend/` is not installed, so no consumer can include them as
shipped (verified by compiling each against the release include tree alone: 12
of 14 fail "file not found"). Fixing that packaging is worth doing on its own
merits and is what would let the rest in.

No rebuild happens on the PR side: the job downloads the `__release_lnx`
artifact `LinuxMakeDPCPP` already uploaded and installs DPC++ only so `icpx` can
parse headers.

### Why the header roots are an explicit list, not the directories

Handing abicheck a directory expands to every header in the tree, parsed as one
aggregate translation unit, and both of oneDAL's trees fail that way outright:
`cpp/daal/include` contains `arrow_numeric_table.h`, which includes
`<arrow/table.h>` — Arrow dev headers are not in the `ubuntu-24.04` apt archive
and oneDAL's own build never compiles that header either — and `cpp/oneapi/dal`
contains headers resolving only relative to `cpp`, several additionally needing
`-I cpp/daal/include`. abicheck's auto-exclusion cannot rescue either case: it
drops only headers raising a `#error` direct-inclusion guard.

So `LIBRARIES` lists, per library, its umbrella header plus exactly those
installed public headers the umbrella does not reach. That list is load-bearing:
on the `oneapi::dal` side the six extra headers add 687 functions and 130 types
(5289 → 5976 functions), because `dbscan`, `linear_regression`,
`logistic_regression`, `correlation_distance`, `cosine_distance` and
`csr_accessor` all ship in `include/` but are unreachable from `dal.hpp`.

**When a public header is added** and the umbrella does not reach it, add it to
that library's entry in *both* workflows, then re-capture the baselines (see
[Rotation](#rotating-to-a-newer-baseline)): a baseline captured before the
addition was fingerprinted against the old header sequence, and a check that
parses more headers than the baseline did stops being comparable to it.

## Baselines

The published release binaries are a multi-ISA (`sse2 sse42 avx2 avx512`)
release-mode build while CI compiles a single-ISA `avx2` one, so comparing
against them would compare build configurations rather than ABI changes. Hence
`.github/workflows/abicheck-baseline.yml` ("Publish Abicheck Baseline") checks
out the release **tag** and rebuilds it with the same three targets the PR-side
build uses. It runs on `workflow_dispatch` (with a `baseline_tag` input) and on
`release: published`, and it:

1. builds the tag, three targets, no debug info (those three produce all six
   libraries);
2. runs the abicheck Action in `mode: dump` once per library, with the same
   header roots, include root and compiler options the PR-side matrix uses;
3. checks a compression tripwire — a snapshot is a few MiB compressed against
   hundreds of MiB of raw JSON, so anything near the 100 MiB limit means the
   `.json.zst` envelope did not apply;
4. uploads each `<tag>-<library>.abicheck.json.zst` as a **release asset** —
   six per baseline tag;
5. commits only its `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`, one line per asset.

The snapshots themselves are deliberately **not** in git: compressed binary per
release is not something review can inspect, and uncompressed they are multiples
of GitHub's 100 MiB per-file limit. The digest file is a few hundred bytes, is
reviewable, and is the trust anchor — `LinuxAbicheckScan` downloads exactly the
asset names it lists and then runs `sha256sum --check --strict`, so a replaced or
corrupted asset fails the job instead of silently changing every PR's verdict.
Each matrix job verifies all six digests and then uses its own library's asset.

Published assets are **immutable**: a run that would overwrite one stops,
because every already-merged PR's "compatible with `<tag>`" verdict was computed
against the published bytes. Existence, not a digest compare, is the rule — and
has to be, since a snapshot is *not* byte-reproducible: two dumps of the same
tree by the same pin differ in size and sha256 while decompressing to identical
content (the L5 source-graph decl nodes reorder). Recovering from a
partially-published run means deleting the asset on purpose; a re-run keeps every
asset already on the release and re-hashes the published bytes, so it converges
instead of deadlocking.

### Bootstrap

`workflow_dispatch` only becomes available once the workflow file is on the
default branch, so the digest file cannot exist before this lands. Until it
does, `LinuxAbicheckScan` **skips** the gate and says so in a warning annotation
and the job summary — inert and visibly inert, rather than red (which would
block the very PR that delivers the publishing workflow) or quietly green. Once
the digest file is on `main`, the gate arms itself with no further edit.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update
`ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml`.

### The abicheck pin and the baseline must move together

The `uses: abicheck/abicheck@<sha>` pin in `ci.yml` and in
`abicheck-baseline.yml` must name the **same** commit. `uses:` accepts no
expression, so the SHA cannot be shared through an env var and is written out at
each call site; a bump has to touch all of them. A snapshot
records a `schema_version`, and detectors whose evidence postdates it decline to
run rather than trust stale facts, so a baseline dumped by an *older* abicheck
than the scanner does not fail — it silently **under-reports**. abicheck warns
on stderr naming each degraded fact, but a warning is not a gate; the reverse
direction, a snapshot newer than the reader, is a hard reject.

The loud failure mode is worse and `schema_version` does not protect against it:
a fix in the **dumper** changes recorded facts without changing the schema, so
the two sides disagree about a value neither side changed. Measured while
validating an earlier bump — an enumerator-value fix made baselines dumped by
the previous pin report **280** breaking `enum_member_value_changed` findings
across `libonedal_core.so` and `libonedal_dpc.so` on an *unchanged* tree,
`schema_version` 25 on both sides; re-dumping took all 280 to zero.

So bumping the pin obliges re-*verifying* the published baselines, with no
`schema_version` shortcut: scan the unchanged tree with the new pin against the
current baselines and diff the report against the old pin's. Both bumps done so
far came out identical line-for-line apart from timings, so neither needed a
re-capture — most recently the bump to the pin now in use (`c599daf7`, 428
commits past its predecessor), where the gate was additionally re-run against
facts freshly captured by the new pin and produced the same report again, from
both baselines. Any difference means re-dispatching the baseline workflow for
every published tag first. If a bump ever produces a wave of findings in one
kind across an unchanged tree, suspect this before suspecting oneDAL.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is printed with its symbol and source
  location. Nothing is filtered out of the report.
* *Did oneDAL break its ABI* — each library's verdict, computed after
  `policy.yaml` re-classifies findings per kind.

Each matrix job gates on its own library, with the Action's defaults: a binary
ABI break (`BREAKING`, exit 4) fails that job; a source-level API break (exit 2)
does not, since `fail-on-api-break` stays off and the header surface is
deliberately wider than the ABI contract. `fail-fast: false` keeps the other five
libraries running, so one PR run reports every affected library rather than the
first one alphabetically.

Each job uploads its own SARIF under `category: abicheck-<library>`. The explicit
category is load-bearing: abicheck's own automation id embeds `--version`, i.e.
the commit sha, so without it every run would open a fresh category whose
predecessor's alerts are never closed — and six jobs uploading with the Action's
own hardcoded `abicheck` category would collide into one result set, each
overwriting the last. Header paths in the SARIF are repo-relative because the
matrix passes repo-relative header roots.

A **suppression** file was tried first and rejected. A suppression rule removes
the matching change *before* the verdict and the counts are computed, and a
suppressed finding then leaves no trace in text output — not the finding, not a
count, not even a note that a suppression file was in effect. The failure mode
is silence, not noise: a synthesized STRONG-symbol visibility regression
produced a run whose visible finding set was identical to a clean one's, exit 0.
A policy re-classification changes only the severity, so the finding stays in
the list, and the gate's SARIF carries a `policyReclassify` record naming the
rule that fired.

`policy.yaml` records each downgrade with its evidence and the class of break it
gives up gating on, and every rule is **scoped**: four name the specific type,
namespace or constant, so any other type's vtable change, any experimental
removal outside `preview` and every other constant still gate. The two rules
covering the `-fvisibility-inlines-hidden` fallout are scoped by ELF linkage
instead (`binding: weak`), which is the evidence for tolerating them — a
`GLOBAL`/STRONG export that is hidden or removed matches neither rule and still
gates. There is no `overrides:` block: no change kind is demoted for every
symbol it could ever fire on.

### Current state

Measured `main` against the `2026.0.0` baseline through the same root Action
this job invokes, on a no-debug-info build of both sides, with `policy.yaml` in
effect: **all six jobs exit 0**, `suppressed_count` 0 on every one, nothing
removed from any report.

| library | verdict | findings | dominant kinds | wall | peak RSS |
|---|---|---|---|---|---|
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 79 | 77 `func_visibility_changed` | 7m46s | 3.66 GiB |
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 68 | 22 `func_added`, 16 `imported_symbol_added`, 12 `type_added`, 6 `func_visibility_changed` | 4m17s | 2.60 GiB |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 86 | 22 `func_added`, 20 `imported_symbol_added`, 12 `type_added`, 10 `func_visibility_changed` | 11m43s | 6.86 GiB |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 2 | 1 `enum_member_added`, 1 `imported_symbol_added` | 1m04s | 0.66 GiB |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 2 | same two | 8m05s | 4.38 GiB |
| `libonedal_thread.so` | `COMPATIBLE` | 1 | 1 `visibility_leak` (ELF-only) | 0.9s | 0.05 GiB |

Because the six run as matrix jobs, the wall-clock cost of the gate is the
slowest single library (~12m), not the 33m their sum would be — but the runner
minutes are the sum, roughly 3× what one sequential job would spend. The peak to
plan for is per job, not aggregate: 6.86 GiB on `libonedal_dpc.so` against a
16 GB runner. `-fsycl -DONEDAL_DATA_PARALLEL` is what costs this — the same
single `system_parameters.hpp` is 1m04s / 0.66 GiB without it and 8m05s /
4.38 GiB with it.

`libonedal_core.so` being almost entirely `risk` is expected and temporary: the
`2026.0.0` baseline predates `makefile` gaining `-fvisibility-inlines-hidden`,
so `main` stopped exporting a large number of WEAK COMDAT inline/template
symbols. The code was not removed — those symbols are still defined as LOCAL
FUNC in the new binaries' `.symtab` — only their export visibility was demoted.
This clears itself once a post-`-fvisibility-inlines-hidden` release becomes the
baseline, at which point the two `inlines-hidden-demotion` rules in
`policy.yaml` should be deleted.

The policy's effect is accountable rather than a blanket mute, and the linkage
scoping is what makes it so. Flipping a single symbol's ELF linkage from WEAK to
GLOBAL on a copy of the real `libonedal.so` baseline snapshot —
`oneapi::dal::detail::v1::homogen_table_builder::build()`, a genuine non-inline,
non-template export — turns the same comparison into `BREAKING`, exit 4, with
exactly one error-level finding naming that symbol. Nothing else about the run
changes. Until the `risk` bucket has had a burn-in period, treat a change in
these counts as something to read, not as a regression.

`require-complete-analysis` is deliberately **not** set. `libonedal_thread.so`
has no installed public header, so its analysis can never be "complete": with
the flag on it fails `ANALYSIS_INCOMPLETE`, exit 1, on a library that has
nothing wrong with it.

One abicheck behaviour is worth knowing before reading a job summary: the
Action's step summary prints `Verdict: COMPATIBLE — No binary ABI break
detected` for a `COMPATIBLE_WITH_RISK` run. The distinction is in the SARIF
(`abiVerdict`) and in the alerts, not in that line.

Findings of kind `func_removed_elf_only` will show up in a **local**
reproduction and not in CI, which is abicheck's design and not a
misconfiguration. Its L0 export-delta fold re-resolves the *binary* each
snapshot was dumped from, located through the snapshot's recorded `source_path`
and identity-checked against the recorded mtime and size. On a PR runner the
baseline binary never existed, so the fold declines silently; with both trees on
disk it fires and reports every WEAK export the demotion above dropped —
measured, 1948 findings across the five header-scoped libraries (1414 + 268 +
237 + 17 + 12), all WEAK in the baseline's `.dynsym` and all still LOCAL FUNC in
the new `.symtab`. `policy.yaml`'s second `inlines-hidden-demotion` rule is what
keeps that local run green; it is the reason the rule exists despite matching
nothing in CI.

## What one job per library does not see

Comparing each library on its own gives up the whole-product view an earlier
revision of this gate had: if `libonedal_thread.so` dropped
`_daal_parallel_sort_int32`, `libonedal_core.so` would fail to load at runtime,
and all six per-library comparisons would still be clean. abicheck can report
that (`bundle_intra_dep_removed`, ADR-023), but only at `depth: binary` and only
from stored bundle facts — which means publishing the baseline **binaries**
alongside the snapshots and a driver to feed them in, i.e. exactly the
oneDAL-owned Python this revision removes. The trade was made deliberately: the
header-scoped per-library comparison is where the findings are, and the ELF-level
"provider disappeared" case is the one `abidiff` against `main` is best placed to
catch anyway.

Two things about oneDAL's DSO graph are worth writing down for whoever revisits
this. `DT_NEEDED` shows only two sibling edges
(`libonedal_parameters*.so → libonedal*.so`): `libonedal_core.so` does not link
`libonedal_thread.so`, and `libonedal_dpc.so` does not link `libonedal_core.so`
despite importing ~800 symbols from it — applications link both, so any
cross-DSO rule has to treat under-linking as normal. And abicheck's
`DEFAULT_SYSTEM_PROVIDERS` omits `libtbbmalloc`, the MKL libraries and the Intel
runtime, while **one** unmatched `DT_NEEDED` edge disables the system-edge
exemption for a whole library — measured, the difference between 862 and 58
`bundle_unresolved_intra_dependency` findings, and between `BREAKING` and
`COMPATIBLE_WITH_RISK`. Any bundle check added later needs that list supplied
explicitly.

## Why not abicheck's declarative project configuration

abicheck's paved road for a multi-library project (G30/ADR-047) is a
`targets:`/`bundles:`/`profiles:`/`baseline:` block in `.abicheck.yml`, validated
by `abicheck project validate`, expanded by `abicheck project plan`, and consumed
by the reusable `check-project.yml` and `publish-baseline.yml` workflows. That is
one step further than the two workflows here go: they call abicheck's root Action
directly and pass the scoping as inputs. oneDAL cannot use the declarative
topology yet, for four reasons upstream records in its own README ("Migrating a
multi-library project onto the declarative topology"), each verified against
abicheck's code at the pinned commit rather than tracked as scheduled work:

* a target's `public_headers:` is validated but never projected into a run-plan
  cell (`buildsource/run_plan.py` never reads it), so per-library header roots
  reach no invocation — and header-scoped comparison is this gate's whole value;
* a `bundles:` check is restricted to `depth: binary`, and `headers` is rejected
  at validation time (`BUNDLE_CHECK_DEPTHS`, `buildsource/project_targets.py`);
* the `compile:` block accepts only
  `{frontend, std, sysroot, nostdinc, include_dirs, defines}`, so `-fsycl` — the
  flag without which three of the six libraries' headers do not parse at all —
  is inexpressible in the config, while the Action's `gcc-options` takes it
  directly;
* `publish-baseline.yml` consumes one `build-output.json` artifact per contract
  profile (G30 P1.1), which oneDAL's makefile build does not emit, and
  `actions/baseline` passes no per-library compiler options to its dump loop
  (`actions/baseline/run.sh`), so it cannot capture the `-fsycl` lanes either.

Having *three* build lanes (plain C++, `-fsycl`,
`-fsycl -DONEDAL_DATA_PARALLEL`) across five libraries plus an ELF-only sixth is
not itself the obstacle — that is three `profiles:`, which the schema handles.
The obstacles are the four above, so a declarative block added today would be
configuration nothing reads; this gate ships none. What it does keep is
everything the paved road expects to be portable: the release-asset baseline with
a committed digest anchor, a release-triggered (never `pull_request`) publishing
workflow, SHA-pinned Actions, abicheck's own Action rather than a project driver,
and policy over suppression. If `public_headers:` ever reaches the run-plan and
`compile:` grows a pass-through for arbitrary flags, this collapses into one
`check-project.yml` call and both matrices go away.

## Known gaps

* **The old side is never re-parsed, and that is what makes this affordable.** A
  directory-vs-directory header-scoped compare has a snapshot on neither side,
  so it parses both — 12 full parses of the union header set, which plateaued at
  **~38 GB** and ran 2.5 hours without finishing. Comparing against a stored
  snapshot parses the new side only. Do not "simplify" this into a directory
  compare, and do not drop the published baselines in favour of building the tag
  in the PR job.
* **`scan --artifact-set DIR` remains the cheap fallback** if the baseline asset
  is ever unavailable: all six libraries, no old side, in **10.8s / 383 MB** with
  `--depth binary` (without it, 8m15s / 1.12 GB for identical findings, reading
  `libonedal_core.so`'s DWARF). Its residual 58 findings are by design, so gate
  on kinds rather than counts if it is ever adopted.
* **The public-header provenance cross-checks are reachable now and still
  deliberately off.** `service.resolve_input` takes `public_header_dirs` as
  provenance only, and passing it moves `exported_not_public`,
  `public_not_exported` and `rtti_for_internal_type` from skipped to present at
  **no** parse cost (measured on `libonedal_parameters.so`: 58.8s against
  59.3s). Volume, not capability, is what stops it: that one library — the
  smallest, one header — yields **129** findings, 68 of 97 exports undocumented
  (42 template instantiations, 25 external-dependency artifacts, 1 genuinely
  undeclared) and 61 declarations whose export obligation the binary does not
  satisfy. They are *intra-version* observations, not drift since the baseline,
  so wiring them on means a policy pass over the reason buckets first, then
  re-captured baselines, since `public_header_dirs` enters the scope
  fingerprint. `header_build_context_mismatch`, `odr_type_variant`,
  `identity_collision_detected` and `compile_context_conflict` stay skipped
  regardless — they need L3/L4.
* **A per-library compiler-options input on `actions/baseline` would collapse
  the baseline workflow's six `mode: dump` steps into one composite call.** Its
  dump loop passes no compiler options at all, and three of the six libraries do
  not parse without `-fsycl`, so the six explicit steps are the whole reason that
  composite is unusable here. Same for the pin: `uses:` takes no expression, so
  the SHA is repeated per step.
* **`binding:` is not accepted as a rule's only scope.** A `reclassify:` entry
  must name at least one of `symbol`, `symbol_pattern`, `type_pattern`,
  `member_name`, `source_location`, `namespace` or `finding_id`, and `binding:`
  is not on that list — so the two linkage-scoped rules in `policy.yaml` carry a
  `symbol_pattern: ".*"` that means nothing beyond satisfying the validator.
  Verified at the pin: `Change.symbol_binding` *is* stamped on both the removal
  and the visibility branch, so the selector itself works; only the
  "at-least-one-identity-selector" check forces the noise.
* **The L0 export-delta fold cannot run on a downloaded baseline.** It
  re-resolves the binary a snapshot was dumped from, so it silently declines
  whenever that binary is not on the local filesystem with matching mtime and
  size — always, in this job. A local reproduction with both trees present
  therefore reports 1948 `func_removed_elf_only` findings that CI does not; see
  Current state. Folding the same fact out of the snapshot's own recorded
  `.dynsym` (which both sides already carry) would close the divergence.
* **L3 build context** (`--depth build --sources . --build-info`) would remove
  the header/binary context-drift findings by feeding the scan the real compile
  flags, and names `-fvisibility-inlines-hidden` directly as one
  `abi_relevant_build_flag_changed`. Measured cost: roughly 20 minutes added to
  the `oneapi::dal` scan, in the preprocessor stage.
* **L4 source facts** (macro / `constexpr` / inline-body diffing) stay out of
  the PR path: the Clang plugin is 2.39× compile time, the `abicheck-cc` wrapper
  is a second frontend per TU, and fact volume is the open risk — one daal TU
  emitted 33 MB, so ~2200 TUs is tens of GB without public-root scoping and
  cross-TU dedup. Intel's oneAPI apt package also ships no `LLVMConfig.cmake`,
  so the plugin cannot be built against `icpx` from it.

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the
rationale. If you change the header lists, the pinned commit, the baseline
storage or the policy, update this file — and the *other* workflow's copy of the
scoping table — in the same PR.
