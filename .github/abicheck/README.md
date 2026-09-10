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

oneDAL's `CI` workflow runs **three ABI gates**, in parallel, all depending only
on `LinuxMakeDPCPP` and never on each other:

| | `ABI Conformance(avx2)` | `Abicheck (multilib)` | `Abicheck L2 (<library>)` |
|---|---|---|---|
| job | `LinuxABICheck` | `LinuxAbicheckScan` — **one** job, one check step | `LinuxAbicheckL2Scan` — a **five-leg matrix**, one library each |
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck)'s own Action, `mode: compare` | the same Action, `mode: compare` |
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as one binary-depth **baseline-set** archive: six snapshots, six staged ELFs and a manifest ([Baselines](#baselines)) | the same tag, as a separate **header-depth** baseline-set archive: five snapshots, no staged ELFs |
| operands | two directories, compared library by library by the script | two directories — the baseline-set's `binaries/` and the PR build's `lib/intel64` — compared library by library by abicheck's release fan-out | one snapshot against one shared object, per leg |
| evidence | ELF exported symbols from the two binaries — `LinuxMakeDPCPP` ships no debug info, so `abidiff` has no type info to read either | ELF exported symbols and metadata, re-derived from both sides' real binaries | the same, **plus** the public-header AST (`evidence_tier: ['elf', 'header']`) |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release | the same, over the declared surface as well as the exported one |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) | the same `policy.yaml` |
| cost | 20 min timeout | 42 s, ~442 MiB | 7–10 min and ~4.7 GiB **per leg** |
| blocks a PR | yes | yes | no — `continue-on-error`, see [Header depth](#header-depth) |

They are complementary, not redundant. `abidiff`'s baseline is `main`, so drift
that accumulates one merged PR at a time across a release cycle never appears in
it; the two abicheck gates answer the question users actually ask — "can I drop
this build in for the release I have installed?" — and answer it per library, with
a verdict that distinguishes *break* from *risk* under a policy whose every
downgrade is evidence-backed.

The two abicheck gates differ in exactly one thing: the evidence they are allowed
to use, and therefore the operand shape they can use it with. abicheck refuses
`--depth headers` for a *directory* operand, so the one-run release comparison is
binary depth by contract; a *single-pair* comparison accepts it and can be held to
it. Hence one cheap blocking job over all six libraries at binary depth, and five
expensive advisory legs at header depth
([why five](#why-five-libraries-and-not-six)). Both read the same released tag,
through two separate baseline asset families ([Baselines](#baselines)).

## Where everything lives

| path | what it is |
|---|---|
| `.github/workflows/ci.yml`, job `LinuxAbicheckScan` | the blocking check: a single `mode: compare` step with two directory operands, binary depth |
| `.github/workflows/ci.yml`, job `LinuxAbicheckL2Scan` | the advisory check: one `mode: compare` step per library, header depth |
| `.github/workflows/abicheck-baseline.yml` | the baseline publisher: two `actions/baseline` calls (one per depth) plus two `actions/stage-baseline` calls that package them |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as both check jobs' `policy-file` |
| `.github/abicheck/abicheck.yml` | the project settings that have no usable Action input — the bundle allow-list and the completeness gate — passed as `build-config` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | the digest of `<tag>`'s **binary-depth** baseline-set archive — the trust anchor, and the only baseline artifact in git ([Baselines](#baselines) for why the header family has none) |
| `.github/.abignore` | libabigail suppressions, used by the *other* tool |

There is no oneDAL-owned Python and no oneDAL-owned driver script: all three
abicheck call sites use `abicheck/abicheck` at a pinned SHA and pass everything as
Action inputs — except the settings abicheck either demoted off its CLI and its
Action, or refuses as inputs on a release operand, which is the whole content of
`abicheck.yml`. Nothing that *has* a usable input is repeated there.

## What the multilib run compares

In `LinuxAbicheckScan`, both operands are directories, which is abicheck's own
multi-library ("release") comparison: one process fans out over every shared object
in `__release_lnx/daal/latest/lib/intel64` and matches each to the baseline library
of the same name. The old side is the `binaries/` directory `actions/resolve-baseline`
extracts out of the baseline-set archive — the tag's **real shared objects**, staged
into the set and digest-verified against its manifest, not the set's snapshots
([why the binaries and not the snapshots](#baselines)). Six libraries, one step, one
verdict, plus a per-library verdict table:

| library | verdict | breaking | risk | additions | dominant kinds |
|---|---|---|---|---|---|
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 0 | 305 | 108 | 237 `func_removed_elf_only`, 102 `func_added`, 51 `symbol_leaked_from_dependency_changed`, 16 `imported_symbol_added` |
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 1417 | 60 | 1414 `func_removed_elf_only`, 36 `var_added`, 24 `func_added`, 2 `symbol_leaked_from_dependency_changed` |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 420 | 120 | 272 `func_removed_elf_only`, 115 `symbol_leaked_from_dependency_changed`, 114 `func_added`, 27 `imported_symbol_added` |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 0 | 23 | 0 | 12 `func_removed_elf_only`, 10 `symbol_leaked_from_dependency_changed` |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 28 | 0 | 17 `func_removed_elf_only`, 10 `symbol_leaked_from_dependency_changed` |
| `libonedal_thread.so` | `COMPATIBLE` | 0 | 0 | 0 | 1 `visibility_leak` |

Measured on `main` against the `2026.0.0` baseline-set, through the same root Action
the job invokes, off a `resolve-baseline`-staged `binaries/` directory, with
`policy.yaml` and `abicheck.yml` in effect: **exit 0**, verdict
`COMPATIBLE_WITH_RISK`, `scope: complete`, 2502 per-library findings plus 156 bundle
ones (2658 detected, 0 effective, 1952 reclassified), nothing removed from the
report; 41.6 s and 442 MiB peak RSS.

**Comparing against staged binaries reports strictly more than comparing against
snapshots did, and the delta is exactly two kinds.** Every risk count above is
higher than the snapshot-operand shape's (254/1415/305/13/18) by precisely its
`symbol_leaked_from_dependency_changed` count — 51, 2, 115, 10, 10 — and
`exported_object_alignment_reduced` appears for the first time (1 on
`libonedal_core.so`, 6 on `libonedal_dpc.so`, both quality issues). Both kinds need
the *old* side's real ELF: which of a library's exports were leaked in from a
dependency, and what alignment its exported objects had, are facts a binary-depth
snapshot does not carry forward. So the migration to baseline-sets is not
cost-neutral in either direction — it costs 41.6 s against 28 s and 29.5 MB of
release asset against 0.49 MB, and it buys 188 findings the previous shape could
not see, plus the six phantom `bundle_library_added` disappearing (below).

There is no `jobs:` input any more — abicheck removed both the input and the
CLI's `--jobs`, and always auto-detects with a memory clamp. The earlier `jobs: 1`
was not a tuning choice but a fit-the-runner one (205 MiB against tens of GiB),
and giving it up is most of the cost above; the clamp is now what has to hold.

**The bundle allow-list is load-bearing, and for two separate reasons.** Bundle
analysis is unconditional since abicheck removed `--no-bundle-analysis` with no
replacement, and the bundle it analyses here is oneDAL's own six libraries — not
the MKL and DPC++ runtime they link. Without the `bundle.system_providers` list in
`abicheck.yml`, every `DT_NEEDED` edge into those runtimes reads as an
intra-bundle dependency that no library in the new bundle provides: 212
`bundle_intra_dep_removed`, bundle verdict `BREAKING`, exit 4, on an unchanged
tree. That count is about the *new* side's unresolved imports and does not depend
on what the old side is — measured twice with the list omitted, once with both
sides real binaries and once with the old side a stored bundle-facts document:
both report the same 212, `BREAKING`, exit 4. The list therefore stays load-bearing
under any old-side representation; it is not an artefact of comparing against
snapshots. Naming MKL's four SYCL interface libraries as system-provided takes the
bundle verdict to `COMPATIBLE`.

The second reason was the six phantom `bundle_library_added` findings, and **they
are gone**: they came from an old side made of six JSON snapshots, which bundle
analysis skips as non-ELF, so every library read as newly added to the bundle. The
old side is now the baseline-set's staged binaries, and the bundle half is real —
`COMPATIBLE_WITH_RISK` over 156 `bundle_intra_dep_signature_unverified`, reported,
not gating. Measured both ways against this exact set: pointing the old operand at
the set's extracted *snapshot* directory reproduces the old behaviour exactly
(bundle verdict `COMPATIBLE`, six `bundle_library_added` for libraries that have
existed all along), which is why `ci.yml` uses `binaries-dir` and not
`snapshot-path`. The 156 unverified-signature findings are the price: a staged ELF
carries no record of the signatures its intra-bundle dependencies were built
against, so abicheck says so rather than assuming.

Three facts worth keeping for whoever revisits cross-library checking. `DT_NEEDED`
shows only two sibling edges (`libonedal_parameters*.so → libonedal*.so`) —
`libonedal_dpc.so` does not link `libonedal_core.so` despite importing ~800
symbols from it, because applications link both. **One** unmatched `DT_NEEDED`
edge disables the system-edge exemption for a whole library, which is why four
entries move the verdict at all (measured on an earlier pin: the difference
between 862 and 58 `bundle_unresolved_intra_dependency` findings, and between
`BREAKING` and `COMPATIBLE_WITH_RISK`). And the entries are soname *stems*:
abicheck matches a `DT_NEEDED` soname against its provider list exactly first,
then falls back to a stem match, and the fallback only applies against a
version-generic entry — so `libmkl_sycl_blas` covers `libmkl_sycl_blas.so.2`
while `libmkl_sycl_blas.so.2` would cover nothing else. abicheck's own 43-entry
`DEFAULT_SYSTEM_PROVIDERS` covers everything else oneDAL links, including the
DPC++ runtime and TBB; the four MKL SYCL interface libraries are the whole
delta, computed with abicheck's own matcher rather than guessed.

## Binary depth

The release fan-out is **binary depth by contract**. abicheck's own
`check-target` reference says a bundle's baseline "is always raw binaries with no
historical header/build/source evidence staged per member", and `--depth headers`
and up is rejected for a directory operand (exit 64, "the per-library fan-out does
not enforce a per-library evidence floor"). `depth: binary` *is* accepted and the
Action forwards it, so `ci.yml` states it rather than inheriting it; it changes
nothing today — `LinuxMakeDPCPP` emits no DWARF at all (`readelf -SW` finds no
`.debug_info` in `libonedal_core.so`) and a run with no depth pinned produces an
identical library table — but it pins the rung the baseline snapshots are dumped
at instead of leaving both sides to a default.

What that gives up, in the report's own words — one such line per library:

> Binary-only analysis without debug info; many ABI changes cannot be detected
> (struct layout, enum values, type changes)

Concretely, within this job — everything in this list is what
`LinuxAbicheckL2Scan` exists to supply, at its own cost:

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
  (`extra-args: --write json=…`) and archives it: 720 KB, one annotation entry
  per finding, all 2314 of them, naming every symbol, plus the
  `comparison_scope` record the completeness gate points at. Note the json's own
  `findings` array is capped at 10 per library (`findings_truncated: true`); the
  `annotations` array is not.

**Header depth inside this job was measured and rejected**, twice over — which is
why it is a separate job rather than a `depth:` change here.

*Asking this job for header depth is refused.* `--depth headers` on a directory
operand is exit 64 ("the per-library fan-out does not enforce a per-library
evidence floor"), and supplying the compile context as Action inputs alongside a
release operand is `::error::` + exit 1 ("the per-library fan-out never threads the
L2 compile context to each pair's header dump"). Passing `-H`/`-I` through
`extra-args`, past that guard, does produce a header-aware whole-release run — and
costs **29 min and 20.1 GiB peak RSS**, on a 16 GB runner, per PR. It is not a
tuning problem: one process holds every library's parsed AST at once.

*Pointing this job at the header-depth baselines is worse than useless.* `depth:
binary` is a projection cap, not a filter — abicheck loads each snapshot in full
and projects it down — so the whole-release directory run measured **13 min 27 s
and 14.6 GiB** and then reported the projected-away header types as removals, on
an unchanged tree. Re-measured per library in the shape that gives the projection
its best case — single-pair operand (so nothing is truncated), the `.l2`
snapshots this repo actually publishes, and this directory's `abicheck.yml` and
`policy.yaml` applied: **every leg BREAKING, exit 4, with 903 breaking
`typedef_removed` on each of the five libraries** and up to **1455 breaking** on
`libonedal_core.so` (which adds 409 `func_virtual_removed` and 143
`func_static_changed`); `source_breaks: 0` throughout; 117–133 s and ~3.2 GiB per
library. The policy this PR ships suppresses about half of the raw count — the
first, policy-less measurement of the same projection reported 1959 on
`libonedal_parameters.so` — but it does not change the outcome, because a
projected-away typedef is not a kind any policy can reclassify to non-breaking.
Hence two baseline families rather than one, keyed by asset name.

What is left is a single-pair comparison per library, which is the one shape
abicheck will *hold* to header depth. That is `LinuxAbicheckL2Scan`.

## Header depth

`LinuxAbicheckL2Scan` is the same comparison against the same released tag, with
the public-header AST added to the evidence. It is a five-leg matrix, one library
per leg, `mode: compare` with a single snapshot against a single shared object —
the only operand shape abicheck will *hold* to `--depth headers`.

`require-complete-analysis: true` is what makes "held to" mean something. Without
it a header set that failed to parse degrades to symbols-only and the run still
exits 0 under a report that says `headers`; with it, an effective depth below the
requested one is exit 1 on abicheck's own assurance axis. Measured on every leg:
`requested_depth: headers`, `effective_depth: headers`, `depth_satisfied: true`,
`header_context_status: clean`, `evidence_tier: header_aware`,
`evidence_tiers: ['elf', 'header']`. A green leg therefore means *compared at
header depth*, not *asked for header depth*.

Measured on `main` against the `2026.0.0` header-depth baselines, with
`policy.yaml` and `abicheck.yml` in effect (so `frontend: castxml`, the frontend
this job actually ships), one library per process, the five run concurrently:

| library | verdict | exit | breaking | API break | risk | additions | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|
| `libonedal.so` | `API_BREAK` | 2 | 0 | 4 | 2407 | 45 | 471 s | 4.70 GiB |
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 3007 | 19 | 572 s | 4.71 GiB |
| `libonedal_dpc.so` | `API_BREAK` | 2 | 0 | 4 | 3271 | 47 | 473 s | 4.71 GiB |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 146 | 19 | 434 s | 4.71 GiB |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 195 | 19 | 440 s | 4.71 GiB |

The same five legs under the clang JSON-AST frontend give the same five verdicts,
the same five exit codes and the same four API breaks, with 3–66 fewer `risk`
findings per library and 237–438 s wall — i.e. the frontend choice moves counts and
runtime, not the gate's answer, on today's tree.

Exit 2 is an API break, and the Action's `fail-on-api-break` defaults to false —
the same setting the binary gate runs with — so those two legs report it without
failing the step. **Zero breaking findings on any leg**, which is the load-bearing
result: header depth does not turn this tree red.

### What it finds that binary depth cannot

Four findings, and they are not noise:

```
experimental_removed_without_replacement ×4 (libonedal.so, and the same 4 in libonedal_dpc.so)
  oneapi::dal::preview::spmd::v1::communicator<...device_memory_access::v1::none>::allgatherv
  …::allreduce
  …::bcast
  …::sendrecv_replace
```

Four `oneapi::dal::preview` SPMD communicator members that `2026.0.0` declared and
`main` no longer declares, with no replacement declaration — a source-level break
in an experimental namespace, `recommended_action: recompile_required`. Nothing in
the export table changed, so `LinuxAbicheckScan` cannot see them at all, and
neither can `abidiff` against `main` (the removal predates the current `main`).
This is the whole argument for paying for header depth.

An earlier, header-scoped revision of this gate carried a policy rule demoting
exactly this kind in exactly this namespace. It was deleted as unfireable at binary
depth, and it is now fireable again — so restoring it is a live decision, not an
oversight. It is left out: four named `preview` removals reported once, in an
advisory job, is more useful than the same four silently reclassified. Revisit if
`preview` churn makes the API_BREAK verdict permanent noise.

The rest of the added volume is intra-version observation rather than drift, and it
is large: `exported_not_public` (2081 on `libonedal.so`, 2894 on
`libonedal_dpc.so`, 1363 on `libonedal_core.so`) — symbols the binary exports that
no public header declares — plus 66 `private_header_leak` on every leg, 140
`rtti_for_internal_type` on `libonedal_core.so`, and `public_not_exported`. All
`risk`, none gating. They describe oneDAL's surface,
not this PR's change to it, so read them as a backlog and not as a per-PR signal
([Known gaps](#known-gaps)).

Header depth also confirms the `-fvisibility-inlines-hidden` demotion arrives under
*both* of `policy.yaml`'s kinds once a declaration is available: on
`libonedal_core.so`, 1201 `func_removed_elf_only` **and** 230
`func_visibility_changed`, all `symbol_binding: weak`, 1857 findings in total
stamped `reclassified_by: inlines-hidden-demotion` (243 and 278 on `libonedal.so`
and `libonedal_dpc.so`, 12 and 17 on the two `parameters` libraries). Unlike the
release fan-out's json, a single-pair json carries per-finding `severity`,
`symbol_binding` and `finding_id`, so *which* linkage a demoted finding had is now
answerable from CI output instead of a local rerun.

### Why five libraries and not six

`libonedal_thread.so` installs no public header of its own. Parsing oneDAL's
umbrellas against it produces 2692 findings — 2190 `public_not_exported`, 334
`private_header_leak`, 151 `exported_not_public` — every one of them saying it does
not export what *other* libraries' headers declare, and none about drift since the
baseline. Its `analysis_assurance.status` is `partial` for the same reason, which
`require-complete-analysis` correctly turns into exit 1 on every run. A permanently
red advisory leg teaches reviewers to ignore the job, so the library stays covered
by the binary-depth gate — where it is `COMPATIBLE` with one finding — and out of
this one, and no `.l2` baseline is published for it.

### Why it does not block

Nothing above argues for advisory-only; the environment does. No leg has ever run
on a GitHub runner, the castxml parse is new to CI, and each leg holds ~4.7 GiB
against a 16 GB runner. So `continue-on-error: true` for a burn-in period, with the
verdict visible in the job summary and the full json archived per leg. Drop the
line once the job has been green across a release cycle.

### Rebuilding an old tag on a runner: pin the oneAPI components

Two separate failures, both found by actually running the publisher on a GitHub
runner rather than locally, both the same shape: **the baseline tag pins oneAPI
component versions, `setvars.sh` prefers the newest installed, and the newest
installed is a runtime-only dependency of something else.** The tag's own
`.ci/env/apt.sh` is what installs them, so the versions are a property of the tag
being rebuilt and are read out of that file rather than written into the workflow.

* `apt.sh dpcpp` installs `intel-oneapi-compiler-dpcpp-cpp=2025.3.3-30` **and**
  unpinned `intel-oneapi-runtime-libs`, which pulls a newer compiler's runtime in.
  `compiler/latest` then points at that runtime-only tree, whose `bin/` has no
  compiler driver: `/bin/sh: 1: icx: not found`, make Error 127, on a tree that
  builds fine locally.
* `apt.sh dpcpp` also drags in `intel-oneapi-mkl-core/-sycl 2026.1.0-236`
  (runtime only) while `apt.sh mkl` pins `intel-oneapi-mkl-devel=2025.3.1-8`.
  `MKLROOT` resolves to the newer one, which ships no static libraries, and the
  tag's static MKL link fails on a file that install never wrote:
  `No rule to make target '/opt/intel/oneapi/mkl/2026.1/lib/libmkl_intel_ilp64.a'`.

The publisher therefore sources the two pinned components' own
`env/vars.sh` after `setvars.sh`, and asserts what it needs before starting a
40-minute build: a resolvable `icx` under the pinned compiler prefix, and
`libmkl_intel_ilp64.a` under the pinned `MKLROOT`. Both assertions print the
versions actually installed, because that is the information the next person needs.
`ci.yml`'s `LinuxMakeDPCPP` has the same latent skew and does not hit it only
because on `main` the pinned versions *are* the newest ones installed — which is
exactly why a publisher that rebuilds an older tag has to be explicit.

### The compile context, and why there is no `compile:` block

A header-depth snapshot records the compile context it was extracted under as a
**profile fingerprint**, and abicheck refuses to compare across a mismatch. So the
publisher and the L2 job have to agree on the frontend, the language standard and
everything else in that fingerprint — and the way they agree is by *neither* of
them asserting it.

`actions/baseline/run.sh` forwards only `-H`/`-I`/`--build-info`/`--depth`/
`--version`/`--compression`/`-o` to its dumps. It declares a `build-config` input
and never reads it, so **a `compile:` block cannot reach a baseline-set dump at
all**: whatever compile context abicheck resolves on its own is what the published
snapshots record. Asserting one on the consumer side only is therefore not
harmless, it is a guaranteed mismatch — measured on `libonedal_parameters.so` with
`compile: {frontend: castxml, std: c++17}` still in `abicheck.yml`:

> Error: 'libonedal_parameters.so' old='2026.0.0' new='probe' are not comparable:
> old and new snapshots were extracted under different compile contexts
> (profile_fingerprint mismatch; differing fields: language_standard) — the
> comparison is not comparable.

Exit 16, no verdict, on an otherwise identical tree. With the block removed from
both sides the same leg is exit 0, `effective_depth: headers`,
`analysis_assurance: complete`. That is why `abicheck.yml` now carries only
`bundle:` and `scope:`, and why the L2 job passes no `ast-frontend` and no
`gcc-options`.

The toolchain is still pinned; it is just pinned by the *installer* rather than by
a config key. `dependency-source: system` on the L2 job makes it provision
dependencies through abicheck's own `install-deps.sh` — the checksum-pinned CastXML
Superbuild, **castxml 0.6.20260105-g9864b1e with clang 21.1.8** — which is the only
castxml `actions/baseline` can provision, and therefore the only one both sides can
share. That is a change from the previous revision's `dependency-source:
conda-forge` (castxml 0.7.0), and the reason the header-depth numbers below were
re-measured rather than carried over. abicheck accepts castxml
**>=0.6.11,<0.8.0** and never falls back to another frontend silently, so the
alternatives remain unavailable rather than merely unchosen: Ubuntu 24.04's apt
castxml is 0.6.3 (refused as too old) and the PyPI `castxml` distribution is 0.4.5
and explicitly "not a supported default scanner setup".

castxml is also the frontend with layout evidence *in principle*: the clang
JSON-AST backend carries no record size/alignment/offset layout, so it cannot see a
struct-layout break, and this build ships no DWARF for L1 to see it instead. It
buys nothing today — measured, both frontends report
`layout_unverified_detectors: ['dwarf', 'advanced_dwarf', 'layout_descriptor']` on
every leg ([Known gaps](#known-gaps)).

**Anything that moves the fingerprint invalidates every published header-depth
set, loudly.** That now includes bumping the abicheck pin far enough to change the
installed CastXML Superbuild, since nothing in this repository states the frontend
any more. The failure mode is exit 16 and no verdict rather than a wrong answer,
which is the right way round; the fix is re-capturing, which means bumping
`ABICHECK_BASELINE_GENERATION` (preferred) or deleting the tag's asset
(`gh release delete-asset`) before re-dispatching, because a published asset is
immutable and the publisher keeps it.

The upstream ask that would give this back as configuration rather than
coincidence: **have `actions/baseline` forward its declared `build-config` input to
the dump as `--config`.** Then both sides could state the compile context in one
file again, and a mismatch would be a config error rather than an install-path
difference.

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
2. calls **`actions/baseline`** twice, once per depth, and lets it dump, validate
   and manifest each family: six libraries at `depth: binary` with
   `stage_binary: true` (35.4 s, 327 MiB peak on the largest, `content-digest`
   `97dab5f9…`), and five at `depth: headers` with no staged binaries
   (~23 min, ~2.9 GiB peak). `validation: strict` round-trips every freshly
   dumped snapshot through a self-compare before it is packaged, which replaces
   the hand-written per-family size ceiling and asset count an earlier revision
   of this workflow carried;
3. calls **`actions/stage-baseline`** twice to package each set as one
   `abicheck-baseline-{profile}-gen{generation}.tar.zst`;
4. uploads the **two archives** as release assets;
5. commits the `sha256sum` of the *binary* archive to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`.

**A baseline-set is not a bag of snapshots.** Each archive carries a
`manifest.json` recording `manifest_version`, `project_ref` (the tag),
`profile`, `snapshot_schema`, `fact_set`, `baseline_generation`, a `generator`
block, and per artifact a `sha256` plus — for the binary family — the staged
`binary` and its `binary_sha256`; the binary family also carries a `binaries/`
directory holding the tag's real shared objects. That is what makes a baseline
*checkable* rather than merely present, and `actions/resolve-baseline` on the
consumer side is what checks it, with a typed outcome per failure mode rather
than a silent wrong comparison. All of these were exercised against this exact
archive:

| case | outcome | exit |
|---|---|---|
| the archive as published | `resolved` | 0 |
| asking for `project_ref` 2025.9.0 | `wrong_project_ref` | 1 |
| asking for generation 2 | `stale_generation` | 1 |
| asking for the `-headers` profile | `wrong_profile` | 1 |
| a path that does not exist, `required: true` | `not_found` | 1 |
| the same path, `required: false` | `not_found`, `bootstrap=true` | 0 |
| one staged ELF altered inside the archive | `ambiguous` | 1 |

The last one is the interesting one: the manifest's `binary_sha256` is verified per
member before the comparison starts, and a one-byte edit to
`binaries/libonedal_parameters.so` fails the job with the expected and actual
digests named — "the baseline-set is corrupt, was tampered with, or was
truncated/replaced after the manifest was written". `wrong_project_ref` is not
hypothetical hardening either: it is what catches a cache or asset-name mix-up
landing on a set built from a different commit than the check requires.

**Two families, because neither job can use the other's set.** The header-depth
one costs ~44× the wall time to capture, so putting it on the blocking job's bill
is not free; and a header-depth snapshot read at `depth: binary` is projected
down, which reports its header-derived types as removed (903 breaking
`typedef_removed` per library on an unchanged tree, up to 1455 on
`libonedal_core.so`, measured single-pair with this directory's config and policy;
see "Header depth inside this job was measured and rejected" above). The
**`profile`** is what keeps them apart, and it is enforced rather than documented:
`resolve-baseline` refuses a profile mismatch outright (`wrong_profile`, exit 1),
where the previous revision's `.l2` filename infix was a convention nothing
checked. It is also why the profile names carry the build shape they were captured
from (`linux-x86_64-icx-avx2-nodbg-…`): a version number alone would let an
avx2/no-debug-info baseline be compared against an avx512 or debug build.

**Generations are the second half of that identity.** `baseline_generation` is
bumped when an abicheck upgrade invalidates already-published sets — a fixed or
newly-extracted fact, a changed normalization or hash recipe — and *not* for
report-format, policy or detector-only changes. It appears in the manifest and in
both archive names, so a bump publishes side by side with the old assets instead of
colliding with immutable ones, and a consumer still pinned to the previous
generation fails as `stale_generation` rather than comparing against facts the new
scanner cannot read. Both jobs' `ABICHECK_BASELINE_GENERATION` must equal the
publisher's.

The archives themselves are deliberately **not** in git; the binary family's digest
is. It is one line, reviewable, and it is the cross-repository trust anchor:
`LinuxAbicheckScan` downloads the asset the digest file names, runs `sha256sum
--check --strict`, and only then hands the archive to `resolve-baseline`. The
manifest inside the archive digests every artifact, but it travels inside the very
bytes an attacker with release-write access would replace, so it cannot anchor
itself. `LinuxAbicheckL2Scan` has no such anchor — it fetches through the Action's
own `abi-baseline`/`baseline-asset-name-template` inputs, which offer no
interposition point for a repo-committed digest — and that is recorded as a gap
rather than papered over with a digest nothing checks (upstream ask: an
expected-archive-digest input). It is the advisory job.

Publishing the baseline **binaries** was rejected in an earlier revision as a
144 MB per-release product decision; that decision is now reversed, for a measured
reason. Bundle analysis skips non-ELF inputs, so without staged binaries the bundle
half of the blocking job is vacuous (six phantom `bundle_library_added`, above),
and the two kinds in the per-library table that need the old side's real ELF —
`symbol_leaked_from_dependency_changed` and `exported_object_alignment_reduced` —
are invisible. The cost is what the archive weighs: **~29.5 MB with the staged
binaries against 491,816 bytes without them**, one asset per release, zstd-compressed
against the 144 MB the raw binaries take.

Published assets are **immutable**: a run that would overwrite one stops, because
every already-merged PR's "compatible with `<tag>`" verdict was computed against
the published bytes. Existence, not a digest compare, is the rule — and has to be,
since a snapshot is content-reproducible but not byte-reproducible (two dumps of
the same `libonedal.so` by the same pin: 87,584 bytes each, different sha256,
differing only in the recorded `created_at`). A re-run keeps every asset already on
the release and re-hashes the published bytes, so it converges instead of
deadlocking.

### Bootstrap is explicit, and absence fails

**A required baseline that is missing fails the job.** An earlier revision of these
jobs skipped every comparison step when the digest file was absent, which produced
six `skipped` steps under six green jobs: a gate that cannot prove it ran is not a
gate, and "green because nothing was compared" is indistinguishable from "green
because nothing broke". Both jobs now fail closed:

* `LinuxAbicheckScan` errors and exits 1 if the digest file for
  `ABICHECK_BASELINE_TAG` is absent, and errors separately if the file exists but
  lists no line for the profile/generation asset it is about to fetch — naming both
  and printing the file, since that is what a generation bump without a re-publish
  looks like. `resolve-baseline` then runs with `required: true`, so an archive that
  downloads but does not resolve is also exit 1.
* `LinuxAbicheckL2Scan` gets the same guarantee from inside the Action: the
  `abi-baseline` path resolves with `required=True`, so a missing or
  wrong-generation asset is `::error::` + exit 1 rather than a skip. (The job is
  `continue-on-error`, so that failure is visible without blocking.)

The one legitimate exception is the bootstrap window — the PR that *delivers* this
workflow cannot have a published baseline, because `workflow_dispatch` only becomes
available once the workflow file is on the default branch. That window is an
explicit opt-in, not an inferred state: set `ABICHECK_BOOTSTRAP: "true"` in
`LinuxAbicheckScan`'s `env:` and the missing digest file downgrades to a
`::notice::` and a skipped comparison. It ships as `"false"`. Flipping it is a
one-line, reviewable change with a name that says what it does, and the review
question it raises — "why is this gate allowed not to run?" — is the right one.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update
`ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml` — in **both**
`LinuxAbicheckScan` and `LinuxAbicheckL2Scan`. The two must name the same tag: a
reviewer reading "compatible with 2026.0.0" from one gate and a different tag from
the other has no way to reconcile them. Rotating the *generation* is the same edit
in three places instead of two: `ABICHECK_BASELINE_GENERATION` in the publisher and
in both jobs. Nothing enforces that they agree — what enforces it is that a
disagreement fails loudly and specifically, as `stale_generation` or as the
"asset not listed in the digest file" error above, rather than comparing against
the wrong facts.

### The abicheck pin and the baseline must move together

The `uses: abicheck/abicheck@<sha>` pin in `ci.yml` (twice) and in
`abicheck-baseline.yml` (eleven times) must name the **same** commit. `uses:`
accepts no expression, so the SHA cannot be shared through an env var and is
written out at each call site; a bump has to touch all of them. The `compile:`
block in `abicheck.yml` carries the same obligation for the `.l2` family, and
enforces it itself — see [The compile
context](#the-compile-context-and-why-it-lives-in-abicheckyml). A snapshot
records a `schema_version`, and detectors whose evidence postdates it decline to
run rather than trust stale facts, so a baseline dumped by an *older* abicheck
than the reader does not fail — it silently **under-reports**. The reverse
direction, a snapshot newer than the reader, is a hard reject.

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
against the old pin's. All three bumps done so far came out identical
line-for-line apart from timings, so none needed a re-capture — the current pin
reads the `schema_version` 25 snapshots the previous one published and writes 44
itself, and still reports the same per-library table, the same 2314 findings and
the same verdict. Any difference means re-capturing every published tag's
baseline first — deleting its assets, then dispatching the baseline workflow. If a
bump ever produces a wave of findings in one kind across an unchanged tree, suspect
this before suspecting oneDAL.

**Rolling a pin back is not symmetric with bumping it.** The previous pin cannot
read a snapshot this one writes at all — measured, it fails per library with
`Cannot detect format` (its own classifier gave up on a `.json.zst` whose first
frame decodes short, fixed upstream since) and behind that fix it would still hit
the hard `schema_version` rejection, since 44 is far past what it understands. A
bump is therefore one-way for as long as the baselines it will read were written
by the older pin: safe now, because the published `2026.0.0` assets predate this
pin, but the moment the baseline workflow publishes at this pin, any rollback
means re-capturing every published tag *first* — and the immutability rule makes
that a deliberate `gh release delete-asset` per library before the dispatch.

## Gating

Two separate questions, answered by two separate mechanisms:

* *What changed* — every finding is in the report. Nothing is filtered out.
* *Did oneDAL break its ABI* — the verdict, computed after `policy.yaml`
  re-classifies findings by kind and ELF linkage.

`LinuxAbicheckScan` gates on the folded release verdict with the Action's defaults:
a binary ABI break (`BREAKING`, exit 4) fails it; a source-level API break (exit 2)
does not, since `fail-on-api-break` stays off. Everything in this section describes
that job unless it says otherwise; `LinuxAbicheckL2Scan` runs the same policy over
richer evidence but is `continue-on-error` and gates nothing yet — see
[Header depth](#header-depth).

A library disappearing from the release also fails it, but **not** through
`fail-on-removed-library: true`. With no proof that the new side's inventory is
complete, abicheck classifies a baseline snapshot with no counterpart as
`not_supplied` — "NEW's inventory is not proven complete, so this is unmatched,
not removed" (ADR-065 D2) — which never reaches the removed-library gate.
Measured by deleting `libonedal_thread.so` from the new side: exit 0 with that
input set either way. `scope.on_incomplete: block` in `abicheck.yml` is what makes
it red, on abicheck's separate *completeness* axis: the same run then exits 1 with
`verdict=SCOPE_INCOMPLETE`, an `::error::` annotation, and the unchecked member
named. A full six-library release is a complete scope, so this does not touch a
normal run (measured: exit 0, unchanged verdict and table). The
`fail-on-removed-library` input stays set for the case abicheck *can* prove the
inventory complete.

The bundle axis gates too, independently of the per-library verdicts: the run that
made the allow-list necessary reported `BREAKING` with all six libraries
individually `COMPATIBLE_WITH_RISK`. It does not mask a per-library break in the
other direction either — the negative control below still exits 4 with the bundle
verdict `COMPATIBLE`.

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
identical to the seven-rule one's, and an identical negative-control failure.
Header depth is back in CI now, in `LinuxAbicheckL2Scan`, and *none of those five
kinds appears in its findings* on the current tree, so they still have nothing to
re-classify. Restoring any of them means measuring it against that job's report
first, not recovering it from git history.

The `func_visibility_changed` rule, by contrast, was written for exactly this and
now earns its place: the same `-fvisibility-inlines-hidden` demotion arrives under
`func_removed_elf_only` where only the export table is available and under
`func_visibility_changed` where a declaration is too, and `LinuxAbicheckL2Scan`
reports both (measured on `libonedal_core.so`: 1201 and 225 respectively, before
policy). Both rules are still linkage-scoped, so a STRONG symbol losing visibility
gates on either job.

Two abicheck behaviours are worth knowing before reading a job summary. The
Action's `verdict` output is one word for the whole run — at this pin it reports
the real one (`COMPATIBLE_WITH_RISK`, and `SCOPE_INCOMPLETE` for the missing-library
case; the pin before it folded both down to `COMPATIBLE`), but the per-library
verdicts and the bundle row exist only in the report the job pastes into the
summary. And the per-library rows are keyed by the *snapshot* filename
(`libonedal_core.so.abicheck.json.zst`), not the shared object's.

`require-complete-analysis` is deliberately **not** set on `LinuxAbicheckScan` and
deliberately **is** set on every leg of `LinuxAbicheckL2Scan`. The Action gates on
it only for a single-pair compare (or `scan --against`), which is the shape the L2
job runs and not the shape the release job runs; and on the L2 job it is the point,
since it is what turns "asked for header depth" into "compared at header depth".
The one library it would fail for the wrong reason, `libonedal_thread.so`, is not
in that matrix at all — see [Header depth](#header-depth).

Until the `risk` bucket has had a burn-in period, treat a change in these counts as
something to read rather than as a regression in itself.

## Why not abicheck's declarative project configuration

`abicheck.yml` here is not that configuration: it carries three keys — two
settings abicheck demoted off its CLI and its Action, and one it refuses as inputs
on a release operand — and no description of oneDAL's libraries, baselines or
profiles.

abicheck's paved road for a multi-library project (G30/ADR-047) is a
`targets:`/`bundles:`/`profiles:`/`baseline:` block in `.abicheck.yml`, consumed by
the reusable `check-project.yml` and `publish-baseline.yml` workflows. Two of the
four obstacles an earlier, header-scoped revision of this gate documented are moot
for the way header depth is reached here: `public_headers:` never reaching a
run-plan cell and `bundles:` being restricted to `depth: binary` are both about the
multi-library shape, and `LinuxAbicheckL2Scan` is a single-pair job that names its
headers as Action inputs and takes part in no bundle. The third —
`-fsycl`/`-DONEDAL_DATA_PARALLEL` being inexpressible — is not solved but
side-stepped: that surface is out of scope for either job (see
[Known gaps](#known-gaps)).

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
* **The SYCL surface is parsed by nothing.** Everything behind
  `ONEDAL_DATA_PARALLEL` needs a DPC++ frontend; castxml is the only frontend that
  carries record layout, and it is not one. So `libonedal_dpc.so` and
  `libonedal_parameters_dpc.so` are compared at header depth on their non-SYCL
  declarations plus their full export table, and the SYCL-only public API is
  covered by neither abicheck gate. `abidiff` sees its symbols but not its
  declarations. What it would take is measured below ("The SYCL gap is an absent
  compiler, not an absent abicheck feature").
* **Header depth is advisory, and the risk volume is why it has to earn its way to
  blocking.** 146 to 3271 risk findings per library, dominated by
  `exported_not_public` (up to 2894) — intra-version observations about oneDAL's
  surface, not drift since the baseline. Zero are breaking, so the counts are a
  backlog to work down rather than a gate to satisfy; treating them as a per-PR
  signal will produce false alarms. Adopting them as one means a policy pass over
  the reason buckets first.
* **Layout breaks are outside both abicheck gates.** Every L2 leg reports
  `analysis_assurance.layout_unverified_detectors: ['dwarf', 'advanced_dwarf',
  'layout_descriptor']` — measured, under castxml as much as under clang — because
  this build ships no DWARF and neither depth reconstructs record layout here. A
  struct that changed size, alignment or member offsets without changing a symbol
  name or a declaration is seen by nothing in this repository's CI — `abidiff`
  would read it out of the two binaries' type info, but `LinuxABICheck` consumes
  the same `LinuxMakeDPCPP` artifact, which carries no debug info, so libabigail
  falls back to comparing exported symbols and covers no layout either. Closing it
  means either debug info in the scanned
  build or L3 evidence (`--sources`/`--build-info`), which is also what the
  preprocessor axis wants: both sides report `ran: false`, `skipped_reason: "no L3
  build evidence"`.
* **castxml on a real runner is unverified.** Every header-depth number here was
  measured locally with conda-forge castxml 0.7.0 on a 224-core host, five legs
  concurrently (434–572 s each). The install path
  (`dependency-source: conda-forge`), the parse's behaviour under a runner's
  narrower CPU and its 16 GB ceiling against a ~4.7 GiB working set, and the
  per-step install cost are all first exercised by this PR's own CI.
* **Duplicate type and mangled-symbol names are resolved first-wins.** Each L2
  parse emits `WARNING: Duplicate type names skipped (first-wins)` and a
  `Duplicate mangled symbols skipped` list running to hundreds of entries
  (`operator()` ×160, `operator=` ×155, `SharedPtr<T>` ×11 and so on) — abicheck
  keys its header-derived surface by name, and oneDAL's `interface1`/`v1`
  versioned namespaces plus template instantiations collide heavily. A change
  confined to a shadowed duplicate is invisible to this gate.
* **`scan --artifact-set DIR` remains the cheap fallback** if the baseline assets
  are ever unavailable: all six libraries, no old side, in 10.8 s with
  `--depth binary`. Its residual findings are by design, so gate on kinds rather
  than counts if it is ever adopted.
* **The release path's policy provenance is now recorded, but not per-symbol
  linkage.** This was a full gap one pin ago: the fan-out's json carried
  `{bucket, kind, symbol, description, source_location}` and reported
  `policy.base: ""` / `policy.reclassify: "[]"` with `--policy` demonstrably
  applied, so the 1952 demoted findings were indistinguishable from findings the
  base policy calls risk on its own. At this pin the json stamps each finding with
  `reclassified_by` (`inlines-hidden-demotion`), carries a `disposition_audit` per
  library and for the run — `reclassified_total: 1952`,
  `reclassifications: [{rule_id: inlines-hidden-demotion, matched_count: 1952}]`,
  1414 of them on `libonedal_core.so` — and fills `effective_config_fields` with
  `policy.base: strict_abi@1:<digest>` plus the serialized rule list, alongside an
  `effective_config_digest` that now describes the config that produced the
  verdict. What a single-pair comparison still has and this does not: per-finding
  `severity`, `symbol_binding` and `finding_id`. Since the two policy rules are
  linkage-scoped, checking *which* linkage a demoted finding had still needs a
  local single-library rerun. The `findings` array is also still capped at 10 per
  library (`findings_truncated: true`); the `annotations` array is not. Both of these
  are properties of the release fan-out, not of comparing a set of libraries — the
  stored-bundle-facts shape in the next-but-one gap emits full per-library reports
  with all three fields and no cap.
* **The bundle axis cannot be silenced or re-classified, only satisfied.** Bundle
  analysis is unconditional (no `--no-bundle-analysis` any more, and no config key
  for it), `policy.yaml` cannot reach it — a `reclassify:` entry on
  `bundle_intra_dep_removed` or `bundle_library_added` is accepted only with
  `to:` in `break, warn, risk, ignore`, and `ignore` on the removal kind is a mute
  rather than an answer — and the six `bundle_library_added` findings are
  structural: the old side is snapshots, so every library reads as newly added to
  the bundle, in every PR's report, for as long as the old side stays snapshots (the
  stored-bundle-facts gap below is what removes them). They do not gate. The one
  supported lever is `bundle.system_providers`, which is why the allow-list exists
  rather than a policy rule. Keeping it current is real maintenance: a new
  `DT_NEEDED` edge to a library outside both abicheck's 43-entry default list and
  this one turns the whole job red, and the message will name a dependency
  removal, not a missing provider.
* **The old side is six snapshots, not a bundle — the stored-bundle-facts shape
  that fixes that is measured and waiting on Action surface.** abicheck can persist
  the old side of a live release compare as one `BundleFacts` document
  (`--bundle-facts-out`, member identity + SONAMEs + provider/consumer entries +
  `variant_fingerprint` + `inventory_complete`) and then take that document as the
  OLD operand. There is no standalone capture command — the flag is a side-output of
  a compare — but the baseline workflow's shape works: comparing the released tree
  against *itself* captures it (exit 0, 66 s) and the bytes are **identical**
  (sha256) to the document captured as a side-output of the PR-side comparison, so
  what gets published does not depend on what it was captured next to. Measured end
  to end on this release pair, binary depth: capture
  43.5 s / 655 MiB peak, **469,852 bytes** as `.json.zst` — slightly *smaller* than
  the 487,979 bytes the six separate snapshots take — and the stored-vs-live compare
  **exit 0 in 24.0 s / 311 MiB**, against 40.4 s / 645 MiB for today's
  directory-of-snapshots shape on the same host. Per-library verdicts are identical
  (five `COMPATIBLE_WITH_RISK`, `libonedal_thread.so` `COMPATIBLE`), the six phantom
  `bundle_library_added` are gone, and `comparison_scope.old_inventory` becomes
  `completeness: "proven"` ("stored bundle-facts capture asserting a complete
  inventory") — which is what would make `fail-on-removed-library: true` mean
  something and let `scope.on_incomplete: block` stop standing in for it. It is also
  a *reporting upgrade*, not a regression: each library's entry in the stored/live
  report is a full single-pair-grade report — all 1477 of `libonedal_core.so`'s
  changes, untruncated, each with `severity`, `symbol_binding`, `finding_id` and
  `reclassified_by`, plus a per-library `disposition_audit`, `effective_config_digest`
  and `effective_config_fields` — which is exactly what the two gaps above ("no
  per-symbol detail in the markdown", "findings capped at 10 per library", "checking
  which linkage a demoted finding had needs a local rerun") ask for. What moves is
  the *root* of the document: `changed_libraries`, `unmatched_old`, `unmatched_new`,
  `warnings`, `old_dir` and the root-level `exit`/`scope`/`disposition_audit`/
  `effective_config_*` blocks are not there, replaced by `per_library_verdict`,
  `not_comparable_members`, `old_bundle_facts` and `extraction_failures`. The Action
  already anticipates the missing root `exit` block and reads
  `comparison_scope`'s `*_exit_contribution` names instead, so the scope axis still
  gates; anything else this directory's notes read off the root has to be re-pointed.
  Three things hold it back, none of them fatal but all of them real: the Action at
  the pinned commit exposes **no bundle-facts input at all**, so both the capture flag
  and the stored operand would have to ride through `extra-args` (upstream ask
  filed); loading the document needs `resource_limits.max_bundle_facts_decode_nodes`
  raised to at least **2,908,138** (measured as the minimum this document passes at)
  against a default of 1,000,000 — and the refusal message says "contains more than
  1000000 JSON containers", which under-describes what it counts: the budget also
  charges one node per scalar leaf, and this document holds only 427,934 real
  containers, so sizing the override from the message alone lands 7× low. It applies
  to the `.json.zst` envelope too, which is otherwise accepted as the stored operand;
  and `--version old=` is rejected against a stored-facts OLD operand, so the
  Action's `old-version` input has to be dropped on that path. It also *adds* 156
  `bundle_intra_dep_signature_unverified` findings, moving the bundle verdict from
  `COMPATIBLE` to `COMPATIBLE_WITH_RISK` — reported, not gating. What it does **not**
  fix is `bundle.system_providers`: with the list omitted, the stored shape reports
  the same 212 `bundle_intra_dep_removed` and exits 4.
* **The same stored shape reaches header depth but cannot be *held* to it, so the
  five L2 legs stay five legs.** Unlike a live directory operand, a stored-facts OLD
  side does accept `--depth headers`, and the NEW side takes a per-library compile
  context through `--bundle-facts-library-manifest` (flat keys — `headers`,
  `includes`, `frontend`, `frontend_context`, `gcc_options`, `gcc_path`,
  `gcc_prefix`, `sysroot`, `nostdinc`; there is no nested `compile:` block and no
  `defines`/`std` key, so `-DONEDAL_DATA_PARALLEL` and `-std=c++17` ride in
  `gcc_options`). Manifest keys do match: measured by giving only
  `libonedal_thread.so` a one-struct throwaway header — that leg came back
  `evidence_tier: header_aware` with the struct reported as `type_added`, and the
  other five came back `elf_only`. That is also the problem. The five libraries with
  no manifest entry were silently dropped to symbols-only *under `--depth headers`*,
  and `--require-complete-analysis` is refused on this operand ("not supported for
  directory/package (release) comparisons yet (P0.4): the per-library fan-out has no
  single analysis_assurance result to gate on"), so nothing enforces the floor —
  exactly the assurance the five single-pair legs exist to provide. The Action would
  refuse it one step earlier anyway: `action/validate-inputs.sh`'s
  `_is_release_style_operand()` is true when *either* operand is a directory, so both
  the compile-context inputs and `require-complete-analysis` are rejected on a
  stored-facts-vs-directory compare even though the CLI accepts the former (measured
  by running that script directly: the plain stored shape with
  `fail-on-removed-library` passes, adding either `ast-frontend`/`gcc-options` or
  `require-complete-analysis` fails it).

  Run end to end anyway, it is close enough to be tempting and the trap is worth
  recording. Capturing the header-depth old side costs **32 min 52 s and 14.5 GiB**
  (headers scoped to `old=`, so the capture's own new side stays symbols-only) and
  produces 26.5 MB of `.json.zst` against the five `.l2` assets' 22.4 MB — but that
  document decompresses to **1.81 GiB**, over a 1 GiB decoded-snapshot ceiling whose
  only overrides are the private `_ABICHECK_SNAPSHOT_MAX_DECODED_BYTES` env var and
  the not-yet-shipped `format="archive"` container, so this path is not reachable
  through supported configuration at all today. Past that, the compare takes **17 min
  52 s and 10.0 GiB** and the five manifested legs reproduce the L2 matrix exactly:
  `libonedal.so` and `libonedal_dpc.so` `API_BREAK` with the same 4 `source_breaks`
  each, the other three `COMPATIBLE_WITH_RISK`, zero breaking findings anywhere. The
  run still exits 4, because `libonedal_thread.so` — the one library with no manifest
  entry, since the L2 matrix deliberately skips it — was compared header-depth OLD
  against symbols-only NEW and reported **2065 breaking findings on an unchanged
  tree**. Its leg records `analysis_assurance: partial`, and nothing acts on that,
  which is the whole point: on this operand the floor is recorded but not enforced.
  Collapsing `LinuxAbicheckL2Scan` onto this path therefore trades a held floor for a
  silent one; it waits on upstream P0.6 (run-plan-aware aggregation).
* **The SYCL gap is an absent compiler, not an absent abicheck feature.** The
  per-library manifest does let a `_dpc` leg ask for a device-context parse, and
  abicheck refuses it for a concrete, fixable reason each time: with
  `frontend: castxml`, "`--frontend-context 'device'` requires the clang header
  backend (`--ast-frontend clang`); castxml has no SYCL/DPC++ host/device context
  concept"; with `frontend: clang` and a plain `clang++`, "`--frontend-context
  'device'` requires a DPC++-capable compiler (icx/icpx/dpcpp/dpcpp-cl); 'clang++' is
  a plain clang/gcc invocation with no device AST context to select". Both were
  measured on the stored path; the two `_dpc` libraries land in
  `extraction_failures` and the run's verdict becomes `ERROR`. So covering
  everything behind `ONEDAL_DATA_PARALLEL` needs oneAPI DPC++ (`icpx`) installed in
  the job and pointed at through `gcc_path`/`frontend`, which is a cost and
  install-surface decision, not a missing capability — and it is the one thing that
  would make the two `_dpc` legs mean what their names suggest.
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
