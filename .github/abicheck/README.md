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
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck) `scan` |
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as an abicheck snapshot fetched from release assets |
| evidence | the two binaries only (ELF symbols + whatever DWARF they carry) | this PR's **public-header AST** plus the binary's exported symbols |
| artifacts | every `lib*.so` in the release directory | three named libraries (below) |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) |
| timeout | 20 min | 45 min |

They are complementary, not redundant, and neither subsumes the other.

## Where everything lives

| path | what it is |
|---|---|
| `.github/abicheck/README.md` | this document |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as `policy-file:` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | digests of the release assets for `<tag>` — the trust anchor, and the only baseline artifact in git |
| `.github/abicheck/.abicheck.yml` | project config: public header surface + L2 compile context |
| `.github/.abignore` | libabigail suppressions, used by the *other* job |

Everything abicheck needs is under `.github/abicheck/`, including the project
config. That last part requires abicheck ≥ the pinned commit: config discovery
originally accepted only a root-level `.abicheck.yml`, and
[abicheck#828](https://github.com/abicheck/abicheck/pull/828) extended it to
also recognise `.github/.abicheck.yml` and `.github/abicheck/.abicheck.yml`,
with relative paths inside the config resolving against the **project root**
rather than the directory the config physically sits in.

Two consequences worth knowing:

* The filename must stay `.abicheck.yml`. Discovery matches that exact name in
  each of the three locations — `config.yml` would not be found.
* Both workflows still pass `build-config:` explicitly, so CI does not depend on
  discovery at all. Discovery is what makes a bare local `abicheck scan` at the
  repo root pick up the same scope and severity settings CI uses, instead of
  silently running with defaults.

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

**`abicheck scan` only.** It parses the public headers, so it can compare the
*declared* surface with the *exported* surface, and its baseline is the last
release, so it answers the question users actually ask — "can I drop this build
in for the release I have installed?" That is a different question from "did
this PR change anything", and it is the one a per-PR `main`-relative diff
structurally cannot answer.

Its blind spots: it only knows about the baseline what the snapshot recorded,
and at the evidence depth this repository runs (below) it does not read debug
info, so it reasons about layout from header ASTs rather than from DWARF.

## Evidence depth — what is actually enabled

abicheck stacks evidence in layers. This repository runs `depth: headers`,
which is **L2**. Observed layer status from a real run of this configuration:

| layer | status here | why |
|---|---|---|
| `L0_binary` | **present** | ELF exported symbols, read from the artifact |
| `L1_debug` | *not collected* | `LinuxMakeDPCPP` builds without `--debug symbols`, so there is no DWARF to read |
| `L2_header` | **present** | public-header AST, parsed with `icpx` |
| `L3_build` | *not collected* | needs real compile flags (`--sources`/`--build-info`) |
| `L4_source_abi` | *not collected* | needs a Clang plugin or a compiler wrapper |
| `L5_source_graph` | **present** | reachability over the declared surface |
| `pattern_scan` | **present** | lexical pre-scan of the changed headers |

That `L1_debug` is absent is a deliberate consequence, not an oversight: L2
needs no DWARF at all, which is exactly what makes scanning
`libonedal_dpc.so` affordable. A DPC++ library built with device debug info is
far too large to dump on a standard runner; without it, the same header-AST
evidence applies unchanged.

L3 and L4 are measured and deferred — see [Known gaps](#known-gaps).

## Which artifacts are scanned, and why three

`scan` analyses exactly one artifact per invocation (there is no per-library
fan-out), and oneDAL exports three independent public surfaces:

| library | header root | extra compile flags |
|---|---|---|
| `libonedal_core.so` | `cpp/daal/include` | `-std=c++17` |
| `libonedal.so` | `cpp/oneapi` | `-fsycl -std=c++17` |
| `libonedal_dpc.so` | `cpp/oneapi` | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |

`-DONEDAL_DATA_PARALLEL` is what makes the `sycl::queue` overloads in
`cpp/oneapi/dal` visible to the parser, and those overloads are precisely the
part of the public API only `libonedal_dpc.so` exports. Without the macro this
scan would report the entire SYCL surface as exported-but-not-declared.

Scanning any one of these against another's headers would report every
declaration as missing, which is why the roots are not shared.

**Not covered by the scan**, and covered by `abidiff` instead — but for two
different reasons, only one of which is a real design boundary:

| library | exports | declared in a shipped header? |
|---|---|---|
| `libonedal_thread.so` | 290 | **no** — the `_daal_*` threading entry points are declared only in `cpp/daal/src/threading/threading.h`, which is not installed |
| `libonedal_parameters.so` | 75 | **yes** — `class ONEDAL_EXPORT system_parameters` in `include/oneapi/dal/detail/parameters/system_parameters.hpp`, plus 27 installed `include/oneapi/dal/algo/*/parameters/` headers |
| `libonedal_parameters_dpc.so` | 124 | **yes**, same headers under `-DONEDAL_DATA_PARALLEL` |

So `libonedal_thread.so` genuinely has no public header surface — it is the
internal C ABI between `libonedal_core.so` and its threading backend, and a
header-versus-exports comparison has nothing to compare. The two
`libonedal_parameters*` libraries are a different story: their exports *are*
header-declared, they are simply not scanned yet. That is a gap, not a
boundary. See [Known gaps](#known-gaps).

This matters more than the export counts suggest, because oneDAL is one header
tree implemented across six `.so` files that call into each other. Scanning
each library in isolation against the shared header tree has two consequences:

* every declaration the *other* libraries implement counts as
  "declared here but not exported here" (2787 such declarations on
  `libonedal.so`, 4974 on `libonedal_core.so`) — inherent to per-library
  scanning, which is why those cross-checks are advisory rather than gating;
* cross-library breakage is invisible. If `libonedal_thread.so` dropped
  `_daal_parallel_sort_int32`, `libonedal_core.so` would fail to load — and no
  per-library scan of either one reports it.

abicheck has a layer built for exactly this (ADR-023 bundle analysis, which
names oneDAL as its motivating case), and it is not what this integration
currently uses. See [Known gaps](#known-gaps).

No rebuild happens on the PR side: this job downloads the `__release_lnx`
artifact `LinuxMakeDPCPP` already uploaded (which contains the `daal`,
`oneapi_c` **and** `oneapi_dpc` outputs) and installs DPC++ only so that
`icpx` is available to parse headers.

## Why the header roots are an explicit list, not the directories

`new-header: <dir>` expands to every header in the tree and parses them as one
aggregate translation unit. Both directories fail that way outright:

* `cpp/daal/include` contains `arrow_numeric_table.h`, which includes
  `<arrow/table.h>`. Apache Arrow dev headers are not in the `ubuntu-24.04`
  apt archive, and oneDAL's own build never compiles that header either.
  abicheck's auto-exclusion only drops headers that raise a `#error`
  direct-inclusion guard — a missing include is a hard failure by design — so
  it cannot rescue this.
* `cpp/oneapi/dal` contains headers that only resolve relative to `cpp`, and
  several that additionally need `-I cpp/daal/include`.

So each side passes its umbrella header plus exactly those installed public
headers the umbrella does not reach. That list is load-bearing rather than
cosmetic: on the `oneapi::dal` side the six extra headers add 687 functions and
130 types to the snapshot (5289 → 5976 functions). `dbscan`,
`linear_regression`, `logistic_regression`, `correlation_distance`,
`cosine_distance` and `csr_accessor` all ship in `include/` but are unreachable
from `dal.hpp`, so the umbrella alone silently under-covers the public API.

**When a public header is added,** add it to the matching `new-header` list in
*both* `ci.yml` and `abicheck-baseline.yml` if it is not reachable from the
umbrella. The two lists must stay identical; a baseline dumped from a
different header set than the scan uses reports every difference as a spurious
add or remove.

## Baselines

### Why the release is rebuilt instead of downloaded

The published release binaries are a multi-ISA (`sse2 sse42 avx2 avx512`)
release-mode build. This repository's CI compiles a single-ISA `avx2` build.
Comparing the two would compare build configurations, not ABI changes. So
`.github/workflows/abicheck-baseline.yml` checks out the release **tag** and
builds it with the same targets the PR-side build uses (`daal`, `oneapi_c`,
`oneapi_dpc`) before dumping.

### How they are produced and stored

`.github/workflows/abicheck-baseline.yml` ("Publish Abicheck Baseline") runs on
`workflow_dispatch` (with a `baseline_tag` input) and on `release: published`,
so baselines refresh automatically for each new release. It:

1. builds the tag, three targets, no debug info;
2. dumps three snapshots (`mode: dump`, same `depth`, frontend and compile
   options as the matching scan);
3. checks a compression tripwire — a snapshot is a few MiB compressed, so
   anything orders of magnitude larger means the `.json.zst` envelope did not
   apply;
4. uploads the three `.json.zst` files as **release assets**;
5. commits only their `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`.

The snapshots themselves are deliberately **not** in git. Compressed they are a
few MiB each and uncompressed 115–152 MiB, over GitHub's 100 MiB per-file push
limit; and a binary blob refreshed every release is not something review can
meaningfully inspect. The digest file is a few hundred bytes, is reviewable,
and is the trust anchor: `LinuxAbicheckScan` downloads exactly the asset names
the digest file lists and then runs `sha256sum --check --strict`, so a
replaced, truncated or corrupted asset fails the job instead of silently
changing every PR's verdict.

Published assets are treated as **immutable**. A re-run that produces
byte-identical assets is a safe retry and is skipped; differing bytes stop the
job, because every already-merged PR's "compatible with `<tag>`" verdict was
computed against the published bytes.

### Bootstrap

`workflow_dispatch` only becomes available once the workflow file is on the
default branch, so the digest file cannot exist before this lands. Until it
does, `LinuxAbicheckScan` **skips** its scans and says so in a warning
annotation and in the job summary — inert and visibly inert, rather than red
(which would block the very PR that delivers the publishing workflow) or
quietly green. Once the digest file is on `main`, the guard passes on every
branch that merges it and the gate arms itself with no further edit.

### Rotating to a newer baseline

1. Dispatch **Publish Abicheck Baseline** for the new tag.
2. Update `ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml`.

### The abicheck pin and the baseline must move together

Every `uses: abicheck/abicheck@<sha>` in `ci.yml` and in
`abicheck-baseline.yml` must name the **same** commit. A snapshot records a
`schema_version`, and detectors whose evidence postdates that version decline
to run rather than trust stale facts — so a baseline dumped by an *older*
abicheck than the scanner does not fail, it silently **under-reports**.
abicheck warns on stderr naming each degraded fact, but a warning is not a
gate. The reverse direction (a snapshot newer than the reader) is a hard
reject.

So bumping the pin obliges re-dispatching the baseline workflow for the tags
already published — **unless** `schema_version` is unchanged across the bump,
in which case existing baselines stay valid and only the pin moves.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is printed with its symbol and source
  location. Nothing is filtered out of the report.
* *Did oneDAL break its ABI* — the severity model in `.abicheck.yml`
  (`exit_code_scheme: severity`), re-classified per finding by
  `.github/abicheck/policy.yaml`.

`fail-on-breaking` / `fail-on-api-break` are deliberately unused: they only
zero the *step's* exit status after the fact, while the CLI still exits 4 and
the report still says `verdict: BREAKING` — so the JSON, the SARIF upload and
the PR comment all keep claiming oneDAL broke its ABI. That is the wrong signal
to publish, and it also disables gating on genuine breaks.

A **suppression** file was tried first and rejected. A suppression rule removes
the matching change *before* the verdict and the severity counts are computed,
and in `scan --format text` a suppressed finding leaves no trace at all — not
the finding, not a count, not even a note that a suppression file was in
effect. The failure mode is silence, not noise: a synthesized STRONG-symbol
visibility regression produced a run whose visible finding set was identical to
a clean one's, exit 0. A policy re-classification changes only the severity, so
the finding stays in the list. That is the whole reason for the swap.

`.github/abicheck/policy.yaml` records each downgrade with its evidence and,
importantly, the class of break it gives up gating on. Four of the five are
**selector-scoped** — naming the specific type, namespace, header or constant —
so the gate stays live for everything the evidence does not cover: any other
type's vtable change, any experimental removal outside `preview`, and every
other constant all still gate. Only `func_visibility_changed` is kind-global,
because abicheck cannot yet narrow it (see below).

### Current state

Measured against the `2026.0.0` baselines, on a no-debug-info build of both
sides, with `.github/abicheck/policy.yaml` in effect:

| library | verdict | exit | risk | compatible | wall | peak RSS |
|---|---|---|---|---|---|---|
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 78 | 1 | 7m13s | 3.48 GB |
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 0 | 27 | 45 | 4m07s | 2.45 GB |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 47 | 49 | 11m24s | 6.56 GB |

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
printed report either way.

Until the `risk` bucket has had a burn-in period, treat a change in these counts
as something to read, not as a regression by itself.

## Known gaps

* **`libonedal_parameters.so` / `libonedal_parameters_dpc.so` are not
  scanned**, although their exports are declared in shipped headers. Adding
  two more `scan` steps is mechanical; the reason to think first is that it
  would be the fifth and sixth per-library scan of the *same* header tree, and
  the per-library model is the thing worth revisiting (next item).
* **Bundle analysis is not used.** oneDAL is one header tree implemented across
  six interdependent `.so` files, which is precisely the shape abicheck's
  ADR-023 bundle layer exists for — it names oneDAL as its motivating case. It
  catches what no per-library scan can: a sibling dropping a symbol another
  sibling imports (`bundle_intra_dep_removed`), `extern "C"` signature drift
  across a DSO boundary, cross-DSO type drift through template-instantiated
  symbols, and provider migration between libraries. It runs by default on
  `abicheck compare <old_dir> <new_dir> -H <headers>` and covers every library
  in one invocation, and
  [abicheck#829](https://github.com/abicheck/abicheck/pull/829) added a
  whole-product baseline format (`pack_product_baseline`/
  `unpack_product_baseline`, plus `compare_product_directories`) for storing the
  old side.

  It was tried end to end against this repository — 2026.0.0 versus `main`, both
  no-debug builds, all six libraries — and it is **not adoptable yet**. Three
  blockers, each measured rather than assumed:

  1. **Header analysis is refused outright in directory mode.** `compare`
     rejects `--ast-frontend`/`--compiler`/`--compiler-option` for
     directory/package operands: *"the per-library fan-out does not thread the
     L2 compile context to each pair's header dump. Compare the libraries
     individually to use them."* oneDAL cannot do without them —
     `cpp/oneapi/dal` includes `sycl/sycl.hpp`, which castxml/GCC cannot parse
     at all — so directory mode can only run headerless here.
  2. **Headerless means no public-surface scoping**, and the numbers show what
     that costs: `libonedal_core.so` reports 1414 breaking, `libonedal.so` 237,
     `libonedal_dpc.so` 272. Those are unscoped internal symbols, not a public
     ABI break. Not gateable.
  3. **The cross-library findings are dominated by externals.** All 379
     `bundle_intra_dep_removed` findings name symbols provided by libraries
     *outside* the compared directory — `_ZdlPvm`, libstdc++ vtables,
     `tbb::detail::r1::spawn`, MKL and SYCL entry points. Neither
     `--bundle-system-providers` (with all 24 sonames listed) nor
     `--search-path` (pointing at the real system and oneAPI library
     directories) changed the count: 379 before, 379 after, both times.

  Worth knowing for whoever picks this up: oneDAL's *actual* intra-bundle
  coupling is narrow. `DT_NEEDED` shows only two sibling edges —
  `libonedal_parameters.so → libonedal.so` and
  `libonedal_parameters_dpc.so → libonedal_dpc.so`. Everything else each library
  needs is external. `libonedal_core.so` does **not** link `libonedal_thread.so`
  at all. So the upside of bundle analysis here is real but smaller than the
  six-library count suggests, and it is gated on blocker 3 being fixed first.

  The storage side is the easy part and already measured: the six libraries are
  145.4 MB raw and **28.1 MB as a single `tar.zst`**, comfortably inside the
  2 GiB release-asset limit — one asset instead of three. Note also that
  `pack_product_baseline` is a library function with no CLI command, so a
  workflow would need a small Python step (or plain `tar --zstd`, giving up the
  manifest and per-library digests). Bundle analysis is ELF/Linux-only, which is
  fine for this gate.
* **`public-header-dir` cannot be passed.** `scan` has a provenance-only
  `--public-header-dir`; `dump` does not — its provenance comes from `-H`,
  where a *directory* entry is also a parse root, and for oneDAL that parse
  fails. Since the comparability hash is computed over exactly
  `("headers", "public_header_dirs")`, a baseline always carries an empty
  `public_header_dirs` while a scan passing the input carries one, the
  fingerprints differ, and the scan returns `NOT_COMPARABLE` instead of a
  verdict. The input is therefore omitted, at the cost of four **advisory**
  cross-checks reporting "skipped — no public-header provenance":
  `exported_not_public`, `public_not_exported`, `private_header_leak` and
  `rtti_for_internal_type`. None of them feeds the verdict or the exit code.
  Restoring them needs a provenance-only `--public-header-dir` on `dump`.
  Re-checked against the pinned commit: `SCOPE_FIELD_KEYS` still contains
  `public_header_dirs`, `dump` still has no such flag, and a scan that passes
  the input still returns `NOT_COMPARABLE` (exit 6). The "dump/scan
  comparability" fixes that landed recently (abicheck#812, #821, #824, #825)
  address a *different* mismatch — Bazel include-dir/`header_roots` parity — and
  do not affect this one. Re-test by adding the input back to one scan step; if
  it produces a verdict instead of `NOT_COMPARABLE`, this gap is closed and the
  input should go back into all three.
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
* **L4 source facts** (macro / `constexpr` / inline-body diffing) stay out of
  the PR path. The Clang plugin is 2.39× compile time; the `abicheck-cc`
  wrapper is a second full frontend per translation unit. Fact volume is the
  open risk — one daal TU emitted 33 MB of facts, so ~2200 TUs is tens of GB
  unless the public-root scoping and cross-TU dedup land first. Intel's oneAPI
  apt package ships no `LLVMConfig.cmake`/`ClangConfig.cmake`, so building the
  plugin against `icpx` is not possible with that package; a nightly job over
  the daal half using apt `clang`/`libclang-dev` is the sane way to get real
  numbers.

## Keeping this document honest

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the
rationale. If you change the header lists, the pinned commit, the baseline
storage or the policy, update this file in the same PR.
