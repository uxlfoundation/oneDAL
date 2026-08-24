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
| `L1_debug` | **present** for `libonedal_core.so`, *not collected* for the other two | `LinuxMakeDPCPP` builds without `--debug symbols`, but `icx` emits DWARF for the `daal` target anyway |
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
Equally, scanning one of these against another's headers would report every
declaration as missing — which is why the roots are not shared.

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
`libonedal_parameters*` libraries are different: their exports *are*
header-declared, just not scanned yet — a gap, not a boundary. See
[Known gaps](#known-gaps).

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

abicheck's ADR-023 bundle layer is built for exactly this and is not used here
yet; see [Known gaps](#known-gaps).

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

Published assets are treated as **immutable**: a run that would overwrite one
stops the job, because every already-merged PR's "compatible with `<tag>`"
verdict was computed against the published bytes. The guard deliberately does
not try to recognise a "harmless re-upload of the same snapshot", because there
is no such thing to recognise — a snapshot is **not byte-reproducible**. Dumping
the same tag twice with the same pin and the same build differs in `created_at`
and in the edge order of the L5 `source_graph` (measured: same serialized
length, same content, different order). Recovering from a partially-published
run therefore means deleting the asset deliberately, which is the point.

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
baselines still hold. The last two bumps both came out that way, and re-dumping
all three anyway differed only in `created_at` and L5 edge order. Any difference
means re-dispatching the baseline workflow for every published tag first. If a
bump ever produces a wave of findings in a single kind across an unchanged tree,
suspect this before suspecting oneDAL.

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
  six interdependent `.so` files, precisely the shape abicheck's ADR-023 bundle
  layer exists for — it names oneDAL as its motivating case. It catches what no
  per-library scan can: a sibling dropping a symbol another sibling imports
  (`bundle_intra_dep_removed`), `extern "C"` signature drift across a DSO
  boundary, cross-DSO type drift through template-instantiated symbols, and
  provider migration between libraries. It runs by default on `abicheck compare
  <old_dir> <new_dir> -H <headers>`, covering every library in one invocation,
  and [abicheck#829](https://github.com/abicheck/abicheck/pull/829) added a
  whole-product baseline format (`pack_product_baseline`/
  `unpack_product_baseline`, `compare_product_directories`) for the old side.
  `--bundle-facts-out` landed as the *producer* half of a stored-baseline bundle
  comparison; the consumer half (feeding a facts file back in as the old
  operand) is deliberately deferred upstream, so the snapshot-first pattern used
  everywhere else here is not yet reachable for bundles from the CLI.

  It was tried end to end against this repository — 2026.0.0 versus `main`, both
  no-debug builds, all six libraries — re-run on every pin bump since, and it is
  **still not adoptable**, though not for the reasons this file first gave. Two
  of the three original blockers were fixed upstream (`--ast-frontend`/
  `--compiler`/`--compiler-option` are accepted for directory and package
  operands as of [abicheck#831](https://github.com/abicheck/abicheck/pull/831),
  and `--bundle-system-providers` — inert while its matcher compared
  `libmkl_core` against the real `DT_NEEDED` string `libmkl_core.so.3` — now
  stem-matches, taking 379 `bundle_intra_dep_removed` findings to **0**). Two
  remain, and together they are a pincer:

  1. **Headerless means no public-surface scoping.** `libonedal_core.so` reports
     1414 breaking, `libonedal.so` 237, `libonedal_dpc.so` 272 — unscoped
     internal symbols, not a public ABI break. Not gateable. (Upstream tried
     deriving the surface from ELF visibility, found it non-functional and
     reverted it, so this is known-open rather than an oversight.)
  2. **Header-scoped directory compare costs more than a runner has.** `scan
     --against <snapshot>` parses one side only, since the baseline is already a
     snapshot — measured `candidate_snapshot=401.07s` against
     `baseline_compare=32.44s`. Directory mode has a snapshot on neither side, so
     it parses **both**: six library pairs × 2 = 12 full parses of the *union*
     header set, against 3 here. Library size is irrelevant — a directory holding
     only `libonedal_thread.so` (290 symbols) with **one** header takes 6m41s and
     4.13 GB, worse than the real `libonedal.so` scan; at 14 headers it ran past
     25 minutes. All six pairs with the real header list plateau at **~38 GB** and
     run for hours; a 16 GB runner cannot host it. Three causes compound: no
     snapshot input, so 2 parses per pair; one shared header list for every pair,
     because per-library header roots live only in #829's library API, not the
     CLI; and no cross-pair AST cache, so the identical union AST is rebuilt 12
     times and — bundle analysis being cross-library and ELF/Linux-only — all 12
     stay resident at once. The last two are #829's own deferred follow-ups; the
     opt-in streaming pruner (`ABICHECK_CLANG_PRUNE_DEPENDENCY_DECLS=1`) is
     upstream-documented as a negative result (~1% less peak RSS for 13–40%
     *more* wall time), and a later upstream fix that stopped retaining full
     snapshots for the whole release moved the ceiling ~4% on the ELF-only path
     and not at all here.

  So header-scoped whole-product compare is out: 1 says headers are needed for
  scoping, 2 says headers in directory mode cost 12 parses. What the
  soname-matching fix *did* unlock is the **headerless** path, which is cheap and
  no longer noise: 34s and 240 MB for all six libraries, and `bundle_verdict` is
  reported separately from the per-library aggregate `verdict`, so a step could
  consume only the former and ignore the unscoped per-library diffs entirely.
  That yields `bundle_verdict: COMPATIBLE_WITH_RISK` with 156 advisory
  `bundle_intra_dep_signature_unverified` findings — a C-boundary
  signature-evidence gate, which fires by construction in headerless mode
  because neither side has type evidence. It is classified
  `COMPATIBLE_WITH_RISK`, not breaking, so a gate on `bundle_verdict == BREAKING`
  is unaffected. Passing `--bundle-system-providers` is not optional on this
  path: without it the same run reports 255 extra `bundle_intra_dep_removed`
  against external providers and `bundle_verdict` becomes `BREAKING`.

  Also aimed squarely at this repository: `scan --artifact-set DIR` (ADR-056,
  whose motivating example is oneDAL by name) audits a set of libraries as one
  artifact with **no old side**, so it needs no baseline. All six libraries in
  **10.8s / 383 MB** — but only with `--depth binary`. Without it the default
  depth reads `libonedal_core.so`'s DWARF (that library does carry
  `.debug_info`) and the same audit takes **8m15s / 1.12 GB** for identical
  findings. It is not adoptable as-is either: it reports 862
  `bundle_unresolved_intra_dependency` findings, of which 804 disappear once
  `--bundle-system-providers` names the sonames `DEFAULT_SYSTEM_PROVIDERS` omits
  (`libtbbmalloc.so.2`, the MKL libraries, and the Intel runtime —
  `libimf`/`libsvml`/`libirng`/`libintlc`), because one unmatched `DT_NEEDED`
  edge disables the system-edge exemption for that whole library. Upstream has
  since recorded that gap as an open action item, unfixed. The remaining 58 are
  libstdc++/libgcc symbols on the two `parameters` libraries and are *by design*:
  audit mode deliberately has no symbol-name-shape fallback, and its suppression
  path requires the consumer to have zero intra-bundle `DT_NEEDED` edges — which
  both `parameters` libraries fail, since each links its sibling. Adopting this
  means either naming those symbols or gating on kinds rather than counts.

  Worth knowing for whoever picks this up: oneDAL's *actual* intra-bundle
  coupling is narrow. `DT_NEEDED` shows only two sibling edges —
  `libonedal_parameters.so → libonedal.so` and
  `libonedal_parameters_dpc.so → libonedal_dpc.so`; everything else is external.
  `libonedal_core.so` does **not** link `libonedal_thread.so`, and
  `libonedal_dpc.so` does not link `libonedal_core.so` either despite importing
  ~800 symbols from it — applications link both. So the upside is real but
  smaller than six libraries suggests, and any cross-DSO gate has to treat
  "consumer under-links its provider" as oneDAL's normal, not a finding. Storage
  is the easy part: the six libraries are 145.4 MB raw and **28.1 MB as a single
  `tar.zst`**, one asset instead of three — though `pack_product_baseline` is a
  library function with no CLI command, so a workflow needs a small Python step
  (or plain `tar --zstd`, giving up the manifest and per-library digests).
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
  comparability" fixes (abicheck#812, #821, #824, #825) address a *different*
  mismatch — Bazel include-dir/`header_roots` parity. Re-test by adding the input
  back to one scan step; a verdict instead of `NOT_COMPARABLE` means the gap
  closed and it should go back into all three.
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
  the PR path. The Clang plugin is 2.39× compile time; the `abicheck-cc` wrapper
  is a second full frontend per TU. Fact volume is the open risk — one daal TU
  emitted 33 MB, so ~2200 TUs is tens of GB unless public-root scoping and
  cross-TU dedup land first. Intel's oneAPI apt package ships no
  `LLVMConfig.cmake`/`ClangConfig.cmake`, so the plugin cannot be built against
  `icpx` from it; a nightly job over the daal half using apt
  `clang`/`libclang-dev` is the sane way to get real numbers.

## Keeping this document honest

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the
rationale. If you change the header lists, the pinned commit, the baseline
storage or the policy, update this file in the same PR.
