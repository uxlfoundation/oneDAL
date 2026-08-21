<!--
  Copyright contributors to the oneDAL project

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# ABI checks in oneDAL CI

oneDAL runs **two** ABI checks on every pull request. They ask different
questions about different inputs, and neither subsumes the other, so both run in
parallel off the same `LinuxMakeDPCPP` build.

| | `LinuxABICheck` | `LinuxAbicheckScan` |
|---|---|---|
| Tool | [libabigail](https://sourceware.org/libabigail/) `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck) `scan`, via its GitHub Action |
| Inputs compared | two **binaries**: this PR's `.so` files vs. the ones from the last completed `main` build | this PR's **public headers** (parsed with `icpx`) plus its **exported symbols**, diffed against a snapshot of the last **release** |
| Reference point | moving - latest green `main` | fixed - a published release tag (`2026.0.0` today) |
| Evidence | ELF `.dynsym`, plus DWARF where the build carries debug info | ELF `.dynsym` plus public-header AST (no DWARF required) |
| Catches | symbols added/removed, and - where DWARF exists - function/variable sub-type changes | the above **plus** header/binary mismatch: a declaration that changed or disappeared from the public headers, a header-visible entity that is no longer exported, vtable/RTTI changes, public-constant changes, alignment and import changes |
| Blind to | whatever the headers say (it never parses them); drift accumulated release-over-release, because the reference moves with `main` | a pure code-generation difference with an unchanged declaration and an unchanged symbol - it does not diff two binaries against each other |
| Verdict model | non-zero `abidiff` exit, with `4` (possibly-compatible) tolerated | abicheck's severity model, per finding kind, from [`.abicheck.yml`](../.abicheck.yml) and [`.abicheck-policy.yaml`](.abicheck-policy.yaml) |

The practical difference is the reference point. `abidiff` answers *"do these two
binaries agree?"*, so a break that landed in an earlier release is invisible once
`main` carries it on both sides. `abicheck scan` answers *"does the public API
this source tree declares still match the last release's, and does the library
actually export it?"* - a break stays reported until it is fixed or explicitly
accepted in the policy file, and a header edit that silently drops a declaration
is caught even though both binaries were built from that same edited header.

Neither check covers ISAs other than the one built here (`avx2`), and neither
covers Windows.

## What `LinuxAbicheckScan` runs

Three scans, one per library/header-tree pair, because `scan` analyses exactly
one artifact and oneDAL exports three independent surfaces:

| Library | Public headers | Extra compile flags |
|---|---|---|
| `libonedal_core.so` | `cpp/daal/include` (classic DAAL API) | `-std=c++17` |
| `libonedal.so` | `cpp/oneapi/dal` (oneapi::dal host API) | `-fsycl -std=c++17` |
| `libonedal_dpc.so` | `cpp/oneapi/dal` (oneapi::dal SYCL API) | `-fsycl -DONEDAL_DATA_PARALLEL -std=c++17` |

`depth: headers` (L2) is the level used: public-header AST plus the binary's
exported symbols. It needs **no debug info** - the AST comes from `icpx`, the
export set from `.dynsym`. That is what makes the DPC++ scan practical:
`LinuxMakeDPCPP` builds every library without `--debug symbols` (device DWARF
alone would blow up object size and peak compiler RSS on a standard runner), and
none of this check's evidence depends on DWARF being present. It also means the
baseline has to be built the same way - see "Rotation" below.

`-DONEDAL_DATA_PARALLEL` on the third scan is what makes the `sycl::queue`
overloads in `cpp/oneapi/dal` visible to the parser; they are exactly the part of
the public API only `libonedal_dpc.so` exports.

Building without debug info is not free, even though the comparison works: with
no DWARF there are no vtable/layout facts and no public-constant values, so
`type_vtable_changed`, `layout_unverifiable` and `constant_changed` are not
collected at all in this configuration. They *were* observed on a debug-info
build of the same libraries, which is why
[`.abicheck-policy.yaml`](.abicheck-policy.yaml) still carries (currently inert)
rules for them. If [#3725](https://github.com/uxlfoundation/oneDAL/pull/3725)
lands `--debug symbols` for the host libraries, this check gets that evidence
back for free - and the published baselines must then be re-dumped, because both
sides have to carry the same evidence.

`--ast-frontend clang` pointed at `icpx` (`gcc-path: icpx`) replaces abicheck's
default castxml/GCC frontend: `cpp/oneapi/dal/detail/common.hpp` includes
`sycl/sycl.hpp`, which castxml/GCC cannot parse at all. The Clang *plugin*
producer (L4: macro/`constexpr`/inline-body diffing) is not usable here - Intel's
oneAPI apt package ships no `LLVMConfig.cmake`/`ClangConfig.cmake` - so
`--ast-frontend clang`, a plain `icpx -Xclang -ast-dump=json` invocation, is the
correct L2 path for `icpx`.

Header roots are an umbrella header plus a short explicit list, not the
directories. `new-header: <dir>` parses every header in the tree as one aggregate
translation unit, and both trees fail that way:
`cpp/daal/include/arrow_numeric_table.h` includes `<arrow/table.h>`, which no
runner has and which oneDAL itself never compiles, and several `cpp/oneapi/dal`
headers only resolve relative to `cpp`. The explicit list is not cosmetic: on the
oneapi::dal side those 6 extra headers add 687 functions and 130 types that
`dal.hpp` alone does not reach.

### Known gap: public-header provenance vs. comparability

The scans deliberately do **not** pass `public-header-dir`, and that is a
trade-off, not an oversight.

abicheck fingerprints the declared surface of both sides (`scope_fingerprint`
over `headers` and `public_header_dirs`) and refuses to compare when they differ.
`scan --public-header-dir X` populates `public_header_dirs`; `dump` has no
equivalent - the Action forwards that input as one more `-H` root, which for a
*directory* triggers the whole-tree aggregate parse above and fails - and neither
`--config .abicheck.yml`'s `sources.public_headers` nor the `-H` file list
populates the field on a dump. A scan that sets it against a dump that cannot is
`NOT_COMPARABLE` (exit 6, *"old and new snapshots do not cover the same declared
surface"*), i.e. no gate at all. Verified on both host libraries.

Dropping it makes the two sides symmetric and the comparison real. The price is
that four provenance-dependent cross-checks report *"skipped: no public-header
provenance"*: `exported_not_public`, `public_not_exported`, `private_header_leak`
and `rtti_for_internal_type`. Those are advisory cross-source warnings, not
inputs to the baseline verdict, so the gate is unaffected. Restoring them needs a
provenance-only input on `dump` upstream.

## Baselines: published as release assets, verified by digest

The baseline is an `abicheck dump` snapshot of a release tag, and it has to be
built the way the PR-side build is - CI compiles a single-ISA (`avx2`) build
while the published release binaries are multi-ISA, so comparing against a
downloaded release *binary* would compare apples to oranges.
[`abicheck-baseline.yml`](workflows/abicheck-baseline.yml) therefore rebuilds the
tag with the same three targets and dumps the snapshots itself.

Those snapshots are **release assets**, not files in the repository:

```
release <tag>
  |-- <tag>-daal.abicheck.json.zst            ~2 MiB
  |-- <tag>-oneapi-dal.abicheck.json.zst      ~2 MiB
  \-- <tag>-oneapi-dal-dpc.abicheck.json.zst  ~5 MiB
```

Uncompressed, a snapshot is 115-152 MiB, over GitHub's hard 100 MiB per-file
push limit; `zstd` brings it to ~1.5% of that with identical findings.
Compression is a storage envelope only - abicheck infers it from the `.json.zst`
suffix on both the dump and the scan side.

What *is* committed is the digest file
`.github/abicheck-baselines/<tag>.abicheck.sha256`, a few hundred bytes in plain
`sha256sum` format. That file is the trust anchor: the PR-side job downloads the
assets named in it and then runs `sha256sum --check --strict`, so the bytes every
PR is gated on are pinned by a reviewed commit rather than by whatever the
release currently serves. A replaced, truncated or corrupted asset fails the job
with a digest mismatch instead of silently changing every PR's verdict.

Consequences worth knowing:

* **Bootstrap.** Until the digest file exists on `main`, the scans are skipped
  with a warning annotation and a job-summary note - inert and visibly inert,
  rather than red or quietly green. `abicheck-baseline.yml` can only be
  dispatched once it is *on* the default branch, so this state is unavoidable on
  the way in.
* **Rotation.** Moving to a newer baseline means dispatching
  `abicheck-baseline.yml` for that tag (it uploads the assets and commits the
  digest file) and updating `ABICHECK_BASELINE_TAG` in
  [`ci.yml`](workflows/ci.yml). The `release: published` trigger does the upload
  half automatically for every new release. Re-dispatch is also required if
  `LinuxMakeDPCPP`'s build flags change in an ABI-visible way - notably if it
  starts building with `--debug symbols`.
* **Immutability.** The publish step refuses to overwrite an existing asset whose
  bytes differ; a re-run producing identical bytes is treated as a safe retry.
  Deliberately replacing a published baseline means deleting the asset
  explicitly (`gh release delete-asset <tag> <name>`) and re-running.

## The abicheck pin in both workflows must move together

A snapshot records the `schema_version` it was produced with. Detectors whose
evidence postdates that version decline to run rather than trust stale facts, so
a baseline dumped by an *older* abicheck than the one scanning does not fail - it
silently **under-reports**. Measured across a `schema_version` 18 baseline read
by a `schema_version` 24 scanner, the scan missed three genuine findings the
matched baseline surfaces. abicheck warns on stderr, naming each degraded fact,
but a warning is not a gate. The reverse direction (snapshot newer than reader)
is a hard reject.

So the commit pinned by the three scan steps in `ci.yml` and the one pinned by
the three dump steps in `abicheck-baseline.yml` must be equal, and bumping either
obliges re-dispatching the baseline workflow for the tags already published.

## Gating: report everything, fail only on a real break

Two questions, two mechanisms:

* *What changed* - every finding is printed with its symbol and source location.
  Nothing is filtered out of the report.
* *Did oneDAL break its ABI* - the severity model in `.abicheck.yml`
  (`exit_code_scheme: severity`, `preset: default`, so `abi_breaking` stays
  `error`), reclassified per finding in `.abicheck-policy.yaml`.

Under `preset: default`, **only `abi_breaking` is error-level**. A source-level
API break is reported, counted, and named in the verdict (`API_BREAK`), but the
job still exits 0 - measured directly. That is deliberate for the burn-in
period; promoting `api_break` to error-level is a one-line `severity:` change in
`.abicheck.yml` once the risk bucket is trusted. It also means the exit code
alone is not a sufficient regression signal when reviewing changes to the policy
file: read the bucket counts.

Current state on the pinned commit, against the 2026.0.0 baselines:

| library | verdict | exit | risk | compatible | abi_breaking |
|---|---|---|---|---|---|
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 78 | 1 | 0 |
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 0 | 27 | 45 | 0 |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 47 | 49 | 0 |

`fail-on-breaking`/`fail-on-api-break` are deliberately unused: they only zero
the step's exit status after the fact, while the report, the JSON and the PR
comment all keep claiming a break. A suppression file was also rejected - a
suppressed finding is removed *before* the verdict and the counts are computed,
and in `scan --format text` it leaves no trace at all (measured: 17 findings
dropped with no count, among them a synthesized STRONG-symbol visibility
regression whose run looked byte-identical to a clean one). A policy
reclassification changes only the severity, so the finding stays in the report.

The per-rule evidence for every downgrade, and the class of break each one gives
up gating on, lives in [`.abicheck-policy.yaml`](.abicheck-policy.yaml). Four of
the five downgrades are selector-scoped to the specific type, namespace or
constant involved, so every *other* vtable change, experimental removal and
constant change still gates.
