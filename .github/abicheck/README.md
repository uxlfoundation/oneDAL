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

| | `ABI Conformance(avx2)` | `Abicheck (header scan + binary, avx2)` |
|---|---|---|
| job | `LinuxABICheck` | `LinuxAbicheckScan` |
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck) bundle comparison, via `.github/abicheck/bundle_gate.py` |
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as an abicheck `BundleFacts` asset fetched from the release |
| evidence | the two binaries only (ELF symbols + whatever DWARF they carry) | this PR's **public-header AST** plus the binary's exported symbols, per library, plus the cross-library dependency graph |
| artifacts | every `lib*.so` in the release directory | all six product libraries, compared as one bundle |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) |
| timeout | 20 min | 75 min |

They are complementary, not redundant, and neither subsumes the other.
`abidiff` never reads a header, so a declaration that ships in `include/` but is
exported by nothing — or an export with no public declaration — is invisible to
it, and because its baseline is `main`, drift that accumulates one merged PR at a
time across a release cycle never appears. abicheck answers the question users
ask instead ("can I drop this build in for the release I have installed?") and
reasons about the six libraries as one product, so a symbol moving between
siblings is a finding rather than two independently-clean diffs. Its own blind
spot: it knows only what the baseline snapshot recorded, and at the depth this
repository runs it reasons about layout from header ASTs rather than DWARF.

## Where everything lives

| path | what it is |
|---|---|
| `.github/abicheck/bundle_gate.py` | the gate itself, in `capture` and `gate` halves |
| `.github/abicheck/onedal_libraries.py` | the six-library scoping table both halves read |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as `--policy` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | digests of the release assets for `<tag>` — the trust anchor, and the only baseline artifact in git |
| `.github/abicheck/.abicheck.yml` | project config: public header surface + L2 compile context |
| `.github/.abignore` | libabigail suppressions, used by the *other* job |

Everything abicheck needs is under `.github/abicheck/`, including the project
config — that last part needs abicheck ≥ the pinned commit, since
[abicheck#828](https://github.com/abicheck/abicheck/pull/828) is what extended
config discovery beyond a root-level `.abicheck.yml`. The filename must stay
`.abicheck.yml`, and relative paths inside it resolve against the project root.

CI does not read that config: discovery is a CLI-layer feature and the gate
drives abicheck as a library, so `bundle_gate.py` is authoritative for CI and
restates the settings itself (`compile.*` becomes the per-library
`CompileContext`, `scope.public` is `compare_snapshots`'s
`scope_to_public_surface`). What the config still buys is a bare local
`abicheck scan` reproducing CI's scope and severity instead of silently running
with defaults. The duplication is deliberate; keep the two in step.

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

All six product libraries, in one run, each with **its own** header roots and
compile flags. The single source of truth is the `LIBRARIES` table in
`onedal_libraries.py`, which the baseline `capture` and the PR-side `gate` both
read, so they cannot drift — and drift is not cosmetic, since a baseline
captured from a different header set reports every header-only difference as a
spurious add/remove.

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
rather than one shared list, which also keeps the per-library halves meaningful
(against the shared tree, every declaration the *other* libraries implement
counts as "declared here but not exported here": 2787 such on `libonedal.so`,
4974 on `libonedal_core.so`, which is why those cross-checks stay advisory).

Two libraries get less than the others, and only one of those is a real design
boundary. `libonedal_thread.so` (290 exports) has no public header surface at
all — it is the internal C ABI between `libonedal_core.so` and its threading
backend, declared only in the uninstalled `cpp/daal/src/threading/threading.h`
— so it rides in the bundle ELF-only, for its dependency edges. The two
`parameters` libraries (75 and 124 exports) are scoped to
`system_parameters.hpp` only because of an **oneDAL packaging bug**: the other
12 installed `parameters` headers include `oneapi/dal/backend/dispatcher.hpp`
and `oneapi/dal/backend/` is not installed, so no consumer can include them as
shipped (measured in `onedal_libraries.py`). Fixing that packaging is worth
doing on its own merits and is what would let the rest in.

The bundle layer covers what per-library diffs cannot: if `libonedal_thread.so`
dropped `_daal_parallel_sort_int32`, `libonedal_core.so` would fail to load, six
independent diffs would each be clean, and the bundle comparison reports
`bundle_intra_dep_removed`.

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
that library's entry in `onedal_libraries.py`, then re-capture the baselines
(see [Rotation](#rotating-to-a-newer-baseline)) — one table feeds both sides, but
a baseline captured before the addition still reports it as a spurious add.

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
2. captures **one** `BundleFacts` document — a header-scoped snapshot per
   library, using the same `LIBRARIES` table the gate uses;
3. checks a compression tripwire — the document is ~13 MiB compressed against
   902 MiB of raw JSON, so anything near the 100 MiB limit means the `.json.zst`
   envelope did not apply;
4. uploads `<tag>-bundle.abicheck.json.zst` as a **release asset**;
5. commits only its `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`.

The facts themselves are deliberately **not** in git: 13 MiB of compressed
binary per release is not something review can inspect, and uncompressed it is
nine times GitHub's 100 MiB per-file limit. The digest file is a few hundred
bytes, is reviewable, and is the trust anchor — `LinuxAbicheckScan` downloads
exactly the asset names it lists and then runs `sha256sum --check --strict`, so
a replaced or corrupted asset fails the job instead of silently changing every
PR's verdict.

Published assets are **immutable**: a run that would overwrite one stops,
because every already-merged PR's "compatible with `<tag>`" verdict was computed
against the published bytes. Existence, not a digest compare, is the rule — and
has to be, since the document is *not* byte-reproducible: two captures of the
same tree measured 13,800,260 against 13,799,608 bytes with different sha256s
while decompressing to identical content (the L5 source-graph decl nodes
reorder). Recovering from a partially-published run means deleting the asset on
purpose.

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

`ABICHECK_PIN` in `ci.yml` and in `abicheck-baseline.yml` must name the **same**
commit. (It is a `pip install` pin rather than a `uses:` one because the gate
drives abicheck as a library — see [Known gaps](#known-gaps).) A snapshot
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
current baselines and diff the report against the old pin's. The bump to the pin
now in use came out identical line-for-line apart from timings, so it needed no
re-capture. Any difference means re-dispatching the baseline workflow for every
published tag first. If a bump ever produces a wave of findings in one kind
across an unchanged tree, suspect this before suspecting oneDAL.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is printed with its symbol and source
  location. Nothing is filtered out of the report.
* *Did oneDAL break its ABI* — the whole-product verdict, computed after
  `policy.yaml` re-classifies findings per kind.

The exit code comes from the **bundle** verdict, not the per-library ones:
`bundle_gate.py` exits 4 when the whole-product comparison says `BREAKING`, 3
when `analysis_errors` is non-empty — a *partial* analysis whose verdict must
not be read as a clean run — and 0 otherwise. The per-library verdicts are
reported in full, in both the job summary and the SARIF upload, but they are not
what fails the job: the libabigail gate already owns per-artifact regressions,
and this one exists for the cross-DSO findings no single-artifact diff can see.

The SARIF carries one run per library, each given a stable `abicheck/<library>/`
automation id and repo-relative paths by `bundle_gate.py`. Both matter to code
scanning: abicheck's own id embeds `--version`, i.e. the commit sha, which would
make every run a fresh category whose predecessor's alerts are never closed, and
the two sides record header paths under different roots, which would split one
finding's locations across two "files".

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
gives up gating on. Four of the five are **selector-scoped** — naming the
specific type, namespace, header or constant — so any other type's vtable
change, any experimental removal outside `preview`, and every other constant all
still gate. Only `func_visibility_changed` is kind-global, because abicheck
cannot yet narrow it (below).

### Current state

Measured against the `2026.0.0` baseline, on a no-debug-info build of both
sides, with `policy.yaml` in effect: **bundle verdict `NO_CHANGE`, zero
cross-library findings, no `analysis_errors`, exit 0**, in 31m28s / 8.03 GiB
peak for the whole product. Resolving the new side is 29m02s of that, serially:
`libonedal_dpc.so` 10m03s, `libonedal_parameters_dpc.so` 7m32s,
`libonedal_core.so` 6m45s, `libonedal.so` 3m42s, `libonedal_parameters.so`
1m00s, `libonedal_thread.so` 0.1s (ELF-only). The peak is the number to watch:
the whole-product baseline stays resident (~3–4 GiB) while each new side is
resolved on top of it, so it is higher than the 6.25 GiB the three per-library
scans needed, and the margin on a 16 GB runner is now ~8 GiB.

| library | verdict | findings | surface | dominant kinds |
|---|---|---|---|---|
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 79 | scoped | 77 `func_visibility_changed` |
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 59 | scoped | 22 `func_added`, 16 `imported_symbol_added`, 6 `func_visibility_changed` |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 80 | scoped | 22 `func_added`, 20 `imported_symbol_added`, 13 `func_visibility_changed` |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 2 | scoped | 1 `enum_member_added`, 1 `imported_symbol_added` |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 2 | scoped | same two |
| `libonedal_thread.so` | `COMPATIBLE` | 1 | ELF-only | 1 `visibility_leak` |

The two `parameters` libraries and `libonedal_thread.so` are new coverage — the
three per-library scans this replaced never looked at them. Against those scans
the risk-and-above findings are **identical, library for library** (78 + 1, 27
and 47: same kinds, same counts). The `compatible` bucket is 13 and 16 lower on
the two `oneapi::dal` libraries, a scope difference worth naming: the CLI dump
applied `.abicheck.yml`'s `sources` filter at *dump* time, recording 320 types
where the capture records 330, the extra ten all libstdc++ types pulled in
transitively by `dal.hpp`. Payload, not findings — compare-time public scoping
keeps them out of the verdict, which is why the risk sets match. The capture
narrows the *compared* surface, no longer the *recorded* one.

`libonedal_core.so` being almost entirely `risk` is expected and temporary: the
`2026.0.0` baseline predates `makefile` gaining `-fvisibility-inlines-hidden`,
so `main` dropped a large number of WEAK COMDAT inline/template exports. The
code was not removed — those symbols are still defined as LOCAL FUNC in the new
binaries' `.symtab` — only their export visibility was demoted. This clears
itself once a post-`-fvisibility-inlines-hidden` release becomes the baseline.

The policy's effect is accountable rather than a blanket mute. On `libonedal.so`
the same scan without the policy file reports
`breaking=6 api_break=4 risk=17 compatible=45` and exits 4; with it,
`breaking=0 api_break=0 risk=27 compatible=45` and exit 0 — the 6 + 4
re-classified findings are exactly the 17 → 27 difference, and they stay in the
printed report either way. Until the `risk` bucket has had a burn-in period,
treat a change in these counts as something to read, not as a regression.

## Why not abicheck's declarative project configuration

abicheck's paved road for a multi-library project (G30/ADR-047) is a
`targets:`/`bundles:`/`profiles:`/`baseline:` block in `.abicheck.yml`, validated
by `abicheck project validate`, expanded by `abicheck project plan`, and consumed
by the reusable `check-project.yml` and `publish-baseline.yml` workflows — no
project-owned Python at all. oneDAL cannot use it yet, for four reasons that are
all in abicheck's own code at the pinned commit:

* a `bundles:` check is restricted to `depth: binary`, and `headers` is rejected
  at validation time (`BUNDLE_CHECK_DEPTHS`, `buildsource/project_targets.py`) —
  because the declarative path assumes a bundle baseline is raw binaries with no
  historical header snapshot. This gate's whole value is the header-scoped
  comparison, against a baseline that *does* carry one per library;
* a target's `public_headers:` is validated but never projected into a run-plan
  cell (`buildsource/run_plan.py` never reads it), so per-library header roots
  reach no invocation;
* `BundleFacts` appears nowhere in the run-plan, the composite Action or
  `check-project.yml`: the stored-facts bundle comparison is Python-only;
* `publish-baseline.yml` consumes one `build-output.json` artifact per contract
  profile (G30 P1.1), which oneDAL's makefile build does not emit.

Per-library compile contexts are the same story from the other side: a
`profiles:` overlay describes a build *lane*, not a library, so "these two of six
libraries need `-fsycl -DONEDAL_DATA_PARALLEL`" is not expressible without one
profile per library. Adding the declarative block today would therefore be
configuration nothing reads, so this PR does not ship one. What it does keep is
everything the paved road expects to be portable: the release-asset baseline with
a committed digest anchor, a release-triggered (never `pull_request`) publishing
workflow, SHA-pinned Actions, and policy over suppression.

## Known gaps

* **The bundle layer has no CLI, so this gate is a committed Python script.**
  `compare_release_against_bundle_facts` is implemented and parity-tested but
  deliberately not on `abicheck compare` (every file that would host the
  dispatch is within two lines of an upstream 2000-line cap), it forwards no
  `CompileContext` (defaulting to castxml, which cannot parse this
  clang/`icpx`-only toolchain), and it applies one `headers`/`includes` set to
  every library. `bundle_gate.py` drives the same Tier-2 chokepoints
  (`service.resolve_input`, `service.compare_snapshots`,
  `bundle_facts.compare_bundle_from_facts`) itself. The cost is a `pip install`
  pin and a script to review; the upside is that per-library scoping and
  `policy.yaml` both survive. A CLI consumer carrying both would make most of
  that file deletable.
* **The system-provider list is load-bearing, not belt-and-braces.** abicheck's
  `DEFAULT_SYSTEM_PROVIDERS` omits `libtbbmalloc`, the MKL libraries and the
  Intel runtime, and **one** unmatched `DT_NEEDED` edge disables the system-edge
  exemption for a whole library — measured, the difference between 862 and 58
  `bundle_unresolved_intra_dependency` findings, and between `BREAKING` and
  `COMPATIBLE_WITH_RISK`. A new external dependency has to be added to
  `SYSTEM_PROVIDERS`. Upstream has recorded the default-list gap as an open
  action item, unfixed.
* **`bundle_intra_dep_signature_unverified` fires by construction** for the
  ELF-only member: it is a C-boundary signature-evidence check, so it reports
  wherever a side has no type evidence, which is `libonedal_thread.so`'s
  permanent state. Never breaking; read the count, do not gate on it.
* **"Consumer under-links its provider" is oneDAL's normal.** `DT_NEEDED` shows
  only two sibling edges (`libonedal_parameters*.so → libonedal*.so`);
  `libonedal_core.so` does not link `libonedal_thread.so`, and
  `libonedal_dpc.so` does not link `libonedal_core.so` despite importing ~800
  symbols from it. Applications link both. Any tightening of the cross-DSO rules
  has to keep treating that as normal.
* **The old side is never re-parsed, and that is what makes this affordable.** A
  directory-vs-directory header-scoped compare has a snapshot on neither side,
  so it parses both — 12 full parses of the union header set, which plateaued at
  **~38 GB** and ran 2.5 hours without finishing. Comparing against stored
  `BundleFacts` parses the new side only. Do not "simplify" this into a
  directory compare.
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
* **`symbol_binding` is not stamped on the visibility branch.** The `binding:`
  selector reads `Change.symbol_binding`, stamped only on the removal kinds, and
  the matcher fails closed when it is `None`. Stamping that one field upstream
  turns the last kind-global override into a scoped rule.
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
storage or the policy, update this file in the same PR.
