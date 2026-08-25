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
| artifacts | every `lib*.so` in the release directory | all six product libraries, compared as one bundle (below) |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) |
| timeout | 20 min | 75 min |

They are complementary, not redundant, and neither subsumes the other.

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
config. That last part requires abicheck ≥ the pinned commit:
[abicheck#828](https://github.com/abicheck/abicheck/pull/828) extended config
discovery from a root-level `.abicheck.yml` only to also recognise
`.github/.abicheck.yml` and `.github/abicheck/.abicheck.yml`, with relative paths
inside resolving against the **project root**, not the config's own directory.
The filename must stay `.abicheck.yml` — discovery matches that exact name at
each location, so `config.yml` would not be found.

CI does not read the config at all: discovery is a CLI-layer feature and the gate
drives abicheck as a library, so `bundle_gate.py` is authoritative for CI and
restates those settings itself — `compile.frontend`/`compile.std` become the
per-library `CompileContext`, `scope.public` is `compare_snapshots`'s
`scope_to_public_surface` (True by default). What the config still does is make a
bare local `abicheck scan` at the repo root reproduce CI's scope and severity
settings instead of silently running with defaults. The duplication is
deliberate, but it is duplication; keep the two in step.

## What each check sees that the other cannot

**`abidiff` only.** It compares two *built* libraries, so it sees the real
recorded type layout on both sides, and it iterates over every shared object
the build produced — including `libonedal_thread.so`,
`libonedal_parameters*.so` and `libonedal_dpc.so`. It is also short-range and
cheap: PR versus the immediately preceding `main`.

Its blind spots follow from the same design. It never reads a header, so a
declaration that ships in `include/` but is not exported by any library — or an
export with no public declaration behind it — is invisible to it. And because
the baseline is `main`, drift that accumulates one merged PR at a time across a
whole release cycle never appears: each individual step is compatible with the
step before it.

**abicheck only.** It parses the public headers, so it can compare the
*declared* surface with the *exported* surface, and its baseline is the last
release, so it answers the question users actually ask — "can I drop this build
in for the release I have installed?" That is a different question from "did
this PR change anything", and it is the one a per-PR `main`-relative diff
structurally cannot answer. It also reasons about the six libraries as one
product rather than six unrelated files, so a symbol moving between siblings, or
a sibling dropping something another sibling imports, is a finding rather than
two independently-clean diffs.

Its blind spots: it only knows about the baseline what the snapshot recorded,
and at the evidence depth this repository runs (below) it does not read debug
info, so it reasons about layout from header ASTs rather than from DWARF.

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
header-AST evidence applies unchanged. Where DWARF happens to be there, as in
`libonedal_core.so`, abicheck uses it; nothing in the gate depends on that. One
consequence to remember: any command reading the *binary* without a snapshot
pays for that DWARF (see `scan --artifact-set` under [Known gaps](#known-gaps)).

L3 and L4 are measured and deferred — see [Known gaps](#known-gaps).

## Which artifacts are checked, and with which headers

All six product libraries, in one run, each with **its own** header roots and
compile flags. The single source of truth is the `LIBRARIES` table in
`onedal_libraries.py`; the baseline `capture` and the PR-side `gate` both read it,
so they cannot drift apart — and drift is not cosmetic, since a baseline captured
from a different header set reports every header-only difference as a spurious
add/remove.

| library | headers | include roots | extra compile flags |
|---|---|---|---|
| `libonedal_core.so` | 7 `daal` roots | `cpp/daal/include` | `-std=c++17` |
| `libonedal.so` | 7 `oneapi::dal` roots | `cpp` | `-fsycl -std=c++17` |
| `libonedal_dpc.so` | same 7 | `cpp` | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |
| `libonedal_parameters.so` | `detail/parameters/system_parameters.hpp` | `cpp` | `-fsycl -std=c++17` |
| `libonedal_parameters_dpc.so` | same 1 | `cpp` | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |
| `libonedal_thread.so` | none — ELF-only | — | — |

`-DONEDAL_DATA_PARALLEL` is what makes the `sycl::queue` overloads in
`cpp/oneapi/dal` visible to the parser, and those overloads are precisely the
part of the public API only `libonedal_dpc.so` exports. Without the macro this
scan would report the entire SYCL surface as exported-but-not-declared.
Equally, scanning one of these against another's headers would report every
declaration as missing — which is why the roots are not shared.

**Why two libraries get less than the others**, and why only one of those is a
real design boundary:

| library | exports | consumable public header surface? |
|---|---|---|
| `libonedal_thread.so` | 290 | **no** — the `_daal_*` threading entry points are declared only in `cpp/daal/src/threading/threading.h`, which is not installed |
| `libonedal_parameters.so` | 75 | **partly** — `class ONEDAL_EXPORT system_parameters` only |
| `libonedal_parameters_dpc.so` | 124 | **partly**, same header under `-DONEDAL_DATA_PARALLEL` |

`libonedal_thread.so` genuinely has no public header surface: it is the internal
C ABI between `libonedal_core.so` and its threading backend, so a
header-versus-exports comparison has nothing to compare. It is still in the
bundle, ELF-only, because its *dependency edges* are what matter there.

The two `parameters` libraries are a different case, and the reason is an oneDAL
packaging bug rather than an abicheck limit. oneDAL installs 14 `parameters`
headers, but the 12 algorithm ones
(`include/oneapi/dal/algo/*/parameters/{cpu,gpu}/*.hpp`) all include
`oneapi/dal/backend/dispatcher.hpp`, and `include/oneapi/dal/backend/` is not
installed at all. Compiling each installed header against the release include
tree alone, 12 of the 14 fail with `'oneapi/dal/backend/dispatcher.hpp' file not
found`; from the *source* tree they instead fail on
`backend/dispatcher_cpu.hpp`'s build-generated
`$(WORKDIR)/oneapi/dal/_dal_cpu_dispatcher_gen.hpp`. No consumer can include them
as shipped, so header-scoping to them would scope against an unreachable surface.
`system_parameters.hpp` needs only `detail/` headers, all installed, so it is in.
Fixing the packaging — installing `oneapi/dal/backend/` and the generated
dispatcher header, or making those 12 headers self-contained — is what would let
the rest in, and is worth doing independently: it is a real defect in what users
can `#include`.

Per-library scoping, rather than one shared list, is what makes the per-library
halves meaningful: against the shared tree every declaration the *other*
libraries implement counts as "declared here but not exported here" (2787 such on
`libonedal.so`, 4974 on `libonedal_core.so`), which is why those cross-checks stay
advisory. The bundle layer covers the rest: if `libonedal_thread.so` dropped
`_daal_parallel_sort_int32`, `libonedal_core.so` would fail to load — six
independent per-library diffs would each be clean, and the bundle comparison
reports `bundle_intra_dep_removed`.

No rebuild happens on the PR side: this job downloads the `__release_lnx`
artifact `LinuxMakeDPCPP` already uploaded (which holds the `daal`, `oneapi_c`
**and** `oneapi_dpc` outputs) and installs DPC++ only so `icpx` can parse headers.

## Why the header roots are an explicit list, not the directories

Handing abicheck a directory expands to every header in the tree, parsed as one
aggregate translation unit. Both of oneDAL's fail that way outright:

* `cpp/daal/include` contains `arrow_numeric_table.h`, which includes
  `<arrow/table.h>`. Apache Arrow dev headers are not in the `ubuntu-24.04`
  apt archive, and oneDAL's own build never compiles that header either.
  abicheck's auto-exclusion only drops headers that raise a `#error`
  direct-inclusion guard — a missing include is a hard failure by design — so
  it cannot rescue this.
* `cpp/oneapi/dal` contains headers that only resolve relative to `cpp`, and
  several that additionally need `-I cpp/daal/include`.

So `LIBRARIES` lists, per library, its umbrella header plus exactly those
installed public headers the umbrella does not reach. That list is load-bearing rather than
cosmetic: on the `oneapi::dal` side the six extra headers add 687 functions and
130 types to the snapshot (5289 → 5976 functions). `dbscan`,
`linear_regression`, `logistic_regression`, `correlation_distance`,
`cosine_distance` and `csr_accessor` all ship in `include/` but are unreachable
from `dal.hpp`, so the umbrella alone silently under-covers the public API.

**When a public header is added** and it is not reachable from the umbrella, add
it to that library's entry in `onedal_libraries.py`, then re-capture the
baselines (see [Rotation](#rotating-to-a-newer-baseline)) — one table feeds both
sides, so they cannot disagree, but a baseline captured before the addition still
reports it as a spurious add.

## Baselines

### Why the release is rebuilt instead of downloaded

The published release binaries are a multi-ISA (`sse2 sse42 avx2 avx512`)
release-mode build. This repository's CI compiles a single-ISA `avx2` build.
Comparing the two would compare build configurations, not ABI changes. So
`.github/workflows/abicheck-baseline.yml` checks out the release **tag** and
builds it with the same targets the PR-side build uses (`daal`, `oneapi_c`,
`oneapi_dpc`) before capturing.

### How they are produced and stored

`.github/workflows/abicheck-baseline.yml` ("Publish Abicheck Baseline") runs on
`workflow_dispatch` (with a `baseline_tag` input) and on `release: published`,
so baselines refresh automatically for each new release. It:

1. builds the tag, three targets, no debug info (those three produce all six
   libraries);
2. captures **one** `BundleFacts` document — a header-scoped snapshot per
   library, using the same `LIBRARIES` table the PR-side gate uses, so the two
   sides cannot disagree about scope;
3. checks a compression tripwire — the document is ~13 MiB compressed against
   902 MiB of raw JSON, so anything near the 100 MiB limit means the `.json.zst`
   envelope did not apply;
4. uploads `<tag>-bundle.abicheck.json.zst` as a **release asset**;
5. commits only its `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`.

The facts themselves are deliberately **not** in git: 13 MiB of compressed
binary refreshed every release is not something review can meaningfully
inspect, and uncompressed it is nine times GitHub's 100 MiB per-file push
limit. The digest file is a few hundred bytes, is reviewable, and is the trust
anchor: `LinuxAbicheckScan` downloads exactly the asset names the digest file
lists and then runs `sha256sum --check --strict`, so a replaced, truncated or
corrupted asset fails the job instead of silently changing every PR's verdict.

Published assets are treated as **immutable**: a run that would overwrite one
stops the job, because every already-merged PR's "compatible with `<tag>`" verdict
was computed against the published bytes. Existence, not a digest compare, is the
rule — and has to be, since the document is *not* byte-reproducible: two captures
of the same tree measured 13,800,260 against 13,799,608 bytes with different
sha256s while decompressing to identical content (the L5 source-graph decl nodes
reorder). Recovering from a partially-published run therefore means deleting the
asset on purpose. That is the point.

### Bootstrap

`workflow_dispatch` only becomes available once the workflow file is on the
default branch, so the digest file cannot exist before this lands. Until it does,
`LinuxAbicheckScan` **skips** the gate and says so in a warning annotation and in
the job summary — inert and visibly inert, rather than red (which would block the
very PR that delivers the publishing workflow) or quietly green. Once the digest
file is on `main`, the gate arms itself with no further edit.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update
`ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml`.

### The abicheck pin and the baseline must move together

`ABICHECK_PIN` in `ci.yml` and in `abicheck-baseline.yml` must name the **same**
commit. (It is a `pip install` pin rather than a `uses:` one because the gate
drives abicheck as a library — see [Known gaps](#known-gaps).) A snapshot records a
`schema_version`, and detectors whose evidence postdates that version decline
to run rather than trust stale facts — so a baseline dumped by an *older*
abicheck than the scanner does not fail, it silently **under-reports**.
abicheck warns on stderr naming each degraded fact, but a warning is not a
gate. The reverse direction (a snapshot newer than the reader) is a hard
reject.

Under-reporting is the *quiet* failure mode. The loud one is worse, and
`schema_version` does not protect against it: a fix in the **dumper** changes the
recorded facts without changing the schema, so the two sides disagree about a
value neither side changed. Measured while validating an earlier bump — an
enumerator-value fix made baselines dumped by the previous pin report **280**
breaking `enum_member_value_changed` findings across `libonedal_core.so` and
`libonedal_dpc.so` on an *unchanged* source tree, `schema_version` 25 on both
sides; re-dumping took all 280 to zero. `abicheck-baseline.yml` records the
details at the pin itself.

So bumping the pin obliges re-*verifying* the published baselines, with no
`schema_version` shortcut. Scan the unchanged tree with the new pin against the
current baselines and diff the report against the old pin's; identical means the
baselines still hold. The bump to the pin now in use came out that way — reports
identical line-for-line apart from timings — so it needed no re-capture. Any
difference means re-dispatching the baseline workflow for every published tag
first. If a bump ever produces a wave of findings in one kind across an unchanged
tree, suspect this before suspecting oneDAL.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is printed with its symbol and source
  location. Nothing is filtered out of the report.
* *Did oneDAL break its ABI* — the whole-product verdict, computed after
  `.github/abicheck/policy.yaml` re-classifies findings per kind.

The exit code comes from the **bundle** verdict, not from the per-library ones:
`bundle_gate.py` exits 4 when the whole-product comparison says `BREAKING`, 3
when `analysis_errors` is non-empty — a *partial* analysis whose verdict must not
be read as a clean run — and 0 otherwise. The per-library verdicts are reported
in full, in both the job summary and the SARIF upload, but they are not what
fails the job: the libabigail gate already owns per-artifact regressions, and
this one exists for the cross-DSO findings no single-artifact diff can see.

The SARIF carries one run per library, each given a stable
`abicheck/<library>/` automation id and repo-relative paths by `bundle_gate.py`.
Both matter to code scanning: abicheck's own id embeds `--version`, i.e. the
commit sha, which would make every run a fresh category whose predecessor's
alerts are never closed, and the two sides record header paths under different
roots, which would split one finding's locations across two "files".

A **suppression** file was tried first and rejected. A suppression rule removes
the matching change *before* the verdict and the severity counts are computed,
and a suppressed finding then leaves no trace at all in text output — not the
finding, not a count, not even a note that a suppression file was in effect. The
failure mode is silence, not noise: a synthesized STRONG-symbol visibility
regression produced a run whose visible finding set was identical to a clean
one's, exit 0. A policy re-classification changes only the severity, so the
finding stays in the list. That is the whole reason for the swap.

`.github/abicheck/policy.yaml` records each downgrade with its evidence and,
importantly, the class of break it gives up gating on. Four of the five are
**selector-scoped** — naming the specific type, namespace, header or constant —
so the gate stays live for everything the evidence does not cover: any other
type's vtable change, any experimental removal outside `preview`, and every
other constant all still gate. Only `func_visibility_changed` is kind-global,
because abicheck cannot yet narrow it (see below).

### Current state

Measured against the `2026.0.0` baseline, on a no-debug-info build of both
sides, with `.github/abicheck/policy.yaml` in effect: **bundle verdict
`NO_CHANGE`, zero cross-library findings, no `analysis_errors`, exit 0**, in
31m28s / 8.03 GiB peak for the whole product.

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
the risk-and-above findings are **identical, library
for library** — 78 + 1, 27 and 47, the same kinds in the same counts. The
`compatible` bucket is 13 and 16 lower on the two `oneapi::dal` libraries,
which is a scope difference worth naming: the CLI dump applied `.abicheck.yml`'s
`sources` filter at *dump* time, recording 320 types where the capture records
330. The extra ten are all libstdc++ (`std::map`, `std::variant`, `error_code`,
…), pulled in transitively by `dal.hpp` — payload, not findings, since
compare-time public scoping keeps them out of the verdict, which is why the risk
sets match. The residual: the capture narrows the *compared* surface, no longer
the *recorded* one.

`libonedal_core.so` being almost entirely `risk` is expected and temporary: the
`2026.0.0` baseline predates `makefile` gaining `-fvisibility-inlines-hidden`,
so `main` dropped a large number of WEAK COMDAT inline/template exports. The
code was not removed — those symbols are still defined as LOCAL FUNC in the new
binaries' `.symtab` — only their export visibility was demoted. This clears
itself once a post-`-fvisibility-inlines-hidden` release becomes the baseline.

The policy's effect is accountable rather than a blanket mute. On
`libonedal.so` the same scan without the policy file reports
`breaking=6 api_break=4 risk=17 compatible=45` and exits 4; with it,
`breaking=0 api_break=0 risk=27 compatible=45` and exit 0 — the 6 + 4
re-classified findings are exactly the 17 → 27 difference, and they stay in the
printed report either way. Until the `risk` bucket has had a burn-in period,
treat a change in these counts as something to read, not as a regression itself.

## Known gaps

* **The bundle layer has no CLI, so this gate is a committed Python script.**
  abicheck's ADR-023 bundle layer names oneDAL as its motivating case, and its
  stored-baseline consumer (`compare_release_against_bundle_facts`) is implemented
  and parity-tested — but deliberately not exposed on `abicheck compare`, because
  every file that would host the dispatch is within two lines of an upstream
  2000-line cap. It is also unusable here as-is for two further reasons: it
  forwards no `CompileContext`, so it defaults to castxml and dies on this
  clang/`icpx`-only toolchain, and it applies one `headers`/`includes` set
  uniformly to every library, which cannot express the per-library roots above.
  `bundle_gate.py` therefore drives the same Tier-2 chokepoints
  (`service.resolve_input`, `service.compare_snapshots`,
  `bundle_facts.compare_bundle_from_facts`) itself. The cost is a `pip install`
  pin instead of a `uses:` pin, and a script to review; the upside is that
  per-library scoping and `policy.yaml` both survive. If a CLI consumer lands
  upstream with both, most of this file becomes deletable.
* **`--bundle-system-providers` is effectively mandatory, and its default list is
  short.** abicheck's `DEFAULT_SYSTEM_PROVIDERS` omits `libtbbmalloc.so.2`, the
  MKL libraries and the Intel runtime (`libimf`/`libsvml`/`libirng`/`libintlc`),
  and **one** unmatched `DT_NEEDED` edge disables the system-edge exemption for
  that whole library — measured, that is the difference between 862 and 58
  `bundle_unresolved_intra_dependency` findings on the audit path, and between
  `BREAKING` and `COMPATIBLE_WITH_RISK` on the compare path. So
  `SYSTEM_PROVIDERS` in `bundle_gate.py` is load-bearing, not belt-and-braces,
  and a new external dependency has to be added to it. Upstream has recorded the
  default-list gap as an open action item, unfixed.
* **`bundle_intra_dep_signature_unverified` fires by construction for the
  ELF-only member.** It is a C-boundary signature-evidence check, so it reports
  wherever a side has no type evidence — which is exactly
  `libonedal_thread.so`'s permanent state. Classified `COMPATIBLE_WITH_RISK`,
  never breaking, so the gate is unaffected; read the count, do not gate on it.
* **"Consumer under-links its provider" is oneDAL's normal.** `DT_NEEDED` shows
  only two sibling edges — `libonedal_parameters.so → libonedal.so` and
  `libonedal_parameters_dpc.so → libonedal_dpc.so`. `libonedal_core.so` does
  **not** link `libonedal_thread.so`, and `libonedal_dpc.so` does not link
  `libonedal_core.so` despite importing ~800 symbols from it; applications link
  both. Any tightening of the cross-DSO rules has to keep treating that as
  normal rather than as a finding.
* **The old side is never re-parsed, and that is what makes this affordable.**
  A directory-vs-directory header-scoped compare has a snapshot on neither side,
  so it parses both: six pairs × 2 = 12 full parses of the union header set. That
  plateaus at **~38 GB** and ran for over 2.5 hours without finishing — a 16 GB
  runner cannot host it. Comparing against stored `BundleFacts` parses the new
  side only, which is the same one-sided arithmetic `scan --against <snapshot>`
  used before. Do not "simplify" this into a directory compare.
* **`scan --artifact-set DIR` (ADR-056) remains the cheap fallback** if the
  baseline asset is ever unavailable: all six libraries, no old side at all, in
  **10.8s / 383 MB** — but only with `--depth binary`. Without it the default
  depth reads `libonedal_core.so`'s DWARF (that library does carry `.debug_info`)
  and the same audit takes **8m15s / 1.12 GB** for identical findings. Its
  residual 58 findings are *by design*: audit mode has no symbol-name-shape
  fallback, and its suppression path requires the consumer to have zero
  intra-bundle `DT_NEEDED` edges, which both `parameters` libraries fail. Gate on
  kinds, not counts, if this is ever adopted.
* **The public-header provenance cross-checks are reachable now, and still
  deliberately off.** Under the old `dump`/`scan` pair they could not run at all:
  `scan` had a provenance-only `--public-header-dir`, `dump` did not, and passing
  it on one side only changed the comparability hash (computed over exactly
  `("headers", "public_header_dirs")`), so the scan returned `NOT_COMPARABLE`
  instead of a verdict. Driving the library directly removes that:
  `service.resolve_input` takes `public_header_dirs` as provenance-only — not a
  parse root, so oneDAL's include tree does not have to survive being compiled —
  and `run_crosschecks` is an in-memory pass over a resolved snapshot. Measured
  on `libonedal_parameters.so`, passing it moves `exported_not_public`,
  `public_not_exported` and `rtti_for_internal_type` from `skipped — no
  public-header provenance` to `present` at **no** parse cost (58.8s against
  59.3s).

  Volume, not capability, is what stops it being wired. That one library — the
  smallest of the six, one header — yields **129** findings: 68 of 97 exports
  undocumented (42 template instantiations, 25 external-dependency artifacts, 1
  genuinely undeclared export) and 61 declarations whose export obligation the
  binary does not satisfy. They are *intra-version* observations about the
  current tree, not drift since the baseline, so they would arrive as thousands
  of untriaged advisory rows. Turning them on is two pieces of work, in order: a
  policy pass over the reason buckets (`template_instantiation` and
  `external_dependency` self-classify as artifacts and want suppressing
  wholesale), then re-captured baselines, since `public_header_dirs` enters the
  scope fingerprint and both sides must carry the same value.
  `header_build_context_mismatch`, `odr_type_variant`,
  `identity_collision_detected` and `compile_context_conflict` stay skipped
  regardless — they need L3/L4 evidence (below), not provenance.
* **`symbol_binding` is not stamped on the visibility branch.** The
  `binding:` selector reads `Change.symbol_binding`, which is stamped only on
  the removal kinds and not on the visibility-changed branch; the matcher fails
  closed when the field is `None`. Stamping that one field upstream turns the
  last kind-global override into a scoped rule.
* **`scan --format json` carries no policy-disclosure block** — the audit dict
  is emitted only on the compare/report path, so a reviewer sees the downgraded
  verdict but not which rule produced it.
* **L3 build context** (`--depth build --sources . --build-info`) would remove
  the header/binary context-drift findings by feeding the scan the real compile
  flags. Measured: roughly 20 minutes added to the `oneapi::dal` scan, in the
  preprocessor stage.
* **L4 source facts** (macro / `constexpr` / inline-body diffing) stay out of the
  PR path. The Clang plugin is 2.39× compile time; the `abicheck-cc` wrapper is a
  second full frontend per TU. Fact volume is the open risk — one daal TU emitted
  33 MB, so ~2200 TUs is tens of GB unless public-root scoping and cross-TU dedup
  land first. Intel's oneAPI apt package ships no `LLVMConfig.cmake`, so the
  plugin cannot be built against `icpx` from it; a nightly job over the daal half
  using apt `clang`/`libclang-dev` is the way to get real numbers.

## Keeping this document honest

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the
rationale. If you change the header lists, the pinned commit, the baseline
storage or the policy, update this file in the same PR.
