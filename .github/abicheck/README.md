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

`CI` runs **three ABI gates** in parallel, all depending only on `LinuxMakeDPCPP`:

| | `ABI Conformance(avx2)` | `Abicheck (multilib)` | `Abicheck L2 (<library>)` |
|---|---|---|---|
| job | `LinuxABICheck` | `LinuxAbicheckScan` — one job, one check step | `LinuxAbicheckL2Scan` — five-leg matrix, one library each |
| tool | libabigail `abidiff`, via `.ci/scripts/abi_check.sh` | [abicheck](https://github.com/abicheck/abicheck)'s own Action, `mode: compare` | the same Action, `mode: compare` |
| baseline | last successful **`main`** build, from the Actions cache | last **release tag**, as one binary-depth **baseline-set**: six snapshots, six staged ELFs, a manifest ([Baselines](#baselines)) | the same tag, separate **header-depth** baseline-set: five snapshots, no staged ELFs |
| operands | two directories, compared library by library | the baseline-set's `binaries/` and the PR build's `lib/intel64`, through abicheck's release fan-out | one snapshot against one shared object, per leg |
| evidence | ELF exports only (the build ships no debug info, so `abidiff` has no type info either) | ELF exports and metadata, re-derived from both sides' real binaries | the same **plus** the public-header AST (`evidence_tier: ['elf', 'header']`) |
| measures | did this PR change the ABI vs `main` | has the ABI drifted since the last release | the same, over the declared surface as well as the exported one |
| filtering | `.github/.abignore` (suppressions) | `.github/abicheck/policy.yaml` ([Gating](#gating)) | the same `policy.yaml` |
| cost | 20 min timeout | 42 s, ~442 MiB | 7–10 min, ~4.7 GiB **per leg** |
| blocks a PR | yes | yes | no — `continue-on-error`, see [Header depth](#header-depth) |

Complementary, not redundant: `abidiff`'s baseline is `main`, so drift accumulating one
merged PR at a time across a release cycle never shows up in it. The abicheck gates
answer "can I drop this build in for the release I have installed?" per library, with a
verdict separating *break* from *risk* under a policy whose every downgrade is
evidence-backed. The two abicheck jobs differ in the evidence they may use, hence the
operand shape: the Action drops `--depth headers` for a *directory* operand and no
assurance floor can be applied to one, so the one-run release comparison is binary depth in
practice, while a *single-pair* comparison both takes the rung and can be held to it — one
cheap blocking job over all six libraries, five
expensive advisory legs ([why five](#why-five-libraries-and-not-six)), one tag read
through two baseline families.

## Where everything lives

| path | what it is |
|---|---|
| `ci.yml`, job `LinuxAbicheckScan` | blocking check: one `mode: compare` step, two directory operands, binary depth |
| `ci.yml`, job `LinuxAbicheckL2Scan` | advisory check: one `mode: compare` step per library, header depth |
| `.github/workflows/abicheck-baseline.yml` | publisher: two `actions/baseline` calls (one per depth) plus two `actions/stage-baseline` calls that package them |
| `.github/abicheck/policy.yaml` | severity re-classification, both jobs' `policy-file` |
| `.github/abicheck/abicheck.yml` | the settings with no usable Action input — bundle allow-list, completeness gate — passed as `build-config` |
| `.github/.abignore` | libabigail suppressions, used by the *other* tool |

Nothing about a baseline is stored in git — no archive, and no digest of one. The
baseline is whatever the selected release carries as an asset, verified against the
digest that release itself records ([Baselines](#baselines)).

No oneDAL-owned Python and no driver script: all three call sites use `abicheck/abicheck`
at a pinned SHA and pass everything as Action inputs, except the settings abicheck
demoted off its CLI and Action or refuses as inputs on a release operand — which is all
`abicheck.yml` contains.

## What the multilib run compares

Both operands are directories, i.e. abicheck's release fan-out: one process over every
shared object in `__release_lnx/daal/latest/lib/intel64`, each matched to the baseline
library of the same name. The old side is the `binaries/` directory
`actions/resolve-baseline` extracts — the tag's **real shared objects**, digest-verified
against the manifest, not the set's snapshots. Six libraries, one step, one verdict, plus
a per-library table:

| library | verdict | breaking | risk | additions | dominant kinds |
|---|---|---|---|---|---|
| `libonedal.so` | `COMPATIBLE_WITH_RISK` | 0 | 305 | 108 | 237 `func_removed_elf_only`, 102 `func_added`, 51 `symbol_leaked_from_dependency_changed` |
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 1423 | 60 | 1414 `func_removed_elf_only`, 36 `var_added`, 8 `exported_object_alignment_reduced` |
| `libonedal_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 436 | 120 | 270 `func_removed_elf_only`, 115 `symbol_leaked_from_dependency_changed`, 47 `imported_symbol_added` |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 0 | 23 | 0 | 12 `func_removed_elf_only`, 10 `symbol_leaked_from_dependency_changed` |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 29 | 0 | 18 `func_removed_elf_only`, 10 `symbol_leaked_from_dependency_changed`, 8 `needed_added` |
| `libonedal_thread.so` | `COMPATIBLE` | 0 | 0 | 0 | 1 `visibility_leak` |

Measured on `main` against the **published** `2026.0.0` baseline-set bytes — the release
asset the job fetches, not a local rebuild ([why that distinction cost a debugging
session](#when-the-baseline-operand-is-not-the-published-one)) — with `policy.yaml` and
`abicheck.yml` in effect: **exit 0**, `COMPATIBLE_WITH_RISK`, `scope: complete`, 2592
per-library plus 156 bundle findings (2748 detected, 0 effective, 1951 reclassified by one
rule, `inlines-hidden-demotion` ×1951), nothing removed from the report; 41.3 s, 428 MiB
peak RSS.

Staged binaries report strictly more than snapshots did, and the delta is exactly two
kinds — 186 `symbol_leaked_from_dependency_changed` (51 / 1 / 115 / 10 / 10) and 13
`exported_object_alignment_reduced` (1 / 8 / 4), both zero against snapshots, because
both need the *old* side's real ELF. Cost of that: 41.3 s against 28 s, and 29.5 MB of
release asset against 0.49 MB. Most of the added time is the loss of `jobs: 1`, which
abicheck replaced with auto-detection plus a memory clamp; the clamp is now what has to
hold on a 16 GB runner.

**The bundle allow-list is load-bearing, twice over.** Bundle analysis is unconditional
(`--no-bundle-analysis` removed, no replacement) and the bundle is oneDAL's own six
libraries, not the MKL/DPC++ runtime they link.

* Without `bundle.system_providers`, every `DT_NEEDED` edge into those runtimes reads as
  an intra-bundle dependency nothing provides: 212 `bundle_intra_dep_removed`,
  `BREAKING`, exit 4, on an unchanged tree — a count about the *new* side's unresolved
  imports, so identical against real binaries and against a stored bundle-facts document.
  Naming MKL's four SYCL interface libraries takes the bundle verdict to `COMPATIBLE`.
* The six phantom `bundle_library_added` an earlier revision reported are **gone**: they
  came from an old side of six JSON snapshots, which bundle analysis skips as non-ELF, so
  every library read as newly added. Reproducible by pointing the old operand at the
  extracted snapshot directory — which is why `ci.yml` uses `binaries-dir`, not
  `snapshot-path`.
* The price of staged ELFs is 156 `bundle_intra_dep_signature_unverified` — reported, not
  gating — because a staged ELF carries no record of the signatures its intra-bundle
  dependencies were built against.
* Entries are soname *stems*: exact match first, then a stem fallback that applies only
  against a version-generic entry, so `libmkl_sycl_blas` covers `libmkl_sycl_blas.so.2`
  while `libmkl_sycl_blas.so.2` covers nothing else. abicheck's 43-entry
  `DEFAULT_SYSTEM_PROVIDERS` covers everything else oneDAL links, including the DPC++
  runtime and TBB; the four MKL SYCL libraries are the whole delta, computed with
  abicheck's matcher rather than guessed.
* **One** unmatched edge disables the system-edge exemption for a whole library, which is
  why four entries move the verdict at all (measured on an earlier pin: 862 vs 58
  `bundle_unresolved_intra_dependency`, `BREAKING` vs `COMPATIBLE_WITH_RISK`).
* `DT_NEEDED` shows only two sibling edges (`libonedal_parameters*.so →
  libonedal*.so`); `libonedal_dpc.so` does not link `libonedal_core.so` despite importing
  ~800 symbols from it, because applications link both.

## Binary depth

The fan-out is **binary depth in practice, not by CLI contract**, and the earlier wording
here ("`--depth headers` is rejected for a directory operand, exit 64") was wrong at this
very pin. Corrected: the CLI's `_resolve_depth_for_set_inputs` forwards every rung to the
release fan-out — its rejection was deleted once every member pair started routing through
`service.run_compare`, which enforces the floor per member — while **the Action this gate
calls still drops the rung** for a directory operand, as a `::notice::` rather than an
error (`action/run.sh:2906`, verified at the pin and at abicheck `main`). Independently,
nothing can hold such a run to the rung: `assurance.require_complete: true` exits 64 on a
directory/package operand, and on a stored-bundle-facts operand too. What remains true from
the older reasoning is the evidence itself: a bundle's baseline "is always raw binaries with
no historical header/build/source evidence staged per member". `depth: binary` is accepted and
forwarded, so `ci.yml` states it rather than inheriting it; it changes nothing today (no
`.debug_info` in the build) but it pins the rung both sides are read at. What it gives up,
in the report's own words — "Binary-only analysis without debug info; many ABI changes
cannot be detected (struct layout, enum values, type changes)":

* **no public-header AST**: no `exported_not_public`/`public_not_exported` provenance, no
  declaration-vs-export mismatch, no struct layout, enum value or type comparison. The
  detectors needing those facts announce themselves as disabled (`vtable_layout`,
  `dwarf_layout_coherence`, `sycl`, plus the kABI/PE/Mach-O/Python ones that never
  applied) — nine coverage warnings per library.
* **no SARIF, so no code scanning** — rejected for a directory operand ("sarif/html/review
  require a single-pair comparison"); `json`, `junit`, `markdown` remain.
* **no per-symbol detail in the markdown** (7.3 KB of per-library counts, no symbol
  names), so the gate also writes and archives the json via `extra-args: --write json=…`:
  720 KB, one annotation per finding, all 2314, naming every symbol, plus the
  `comparison_scope` record the completeness gate points at. The json's `findings` array
  is capped at 10 per library (`findings_truncated: true`); `annotations` is not.

`LinuxAbicheckL2Scan` supplies that list, at its own cost. **Header depth inside this job
was measured and rejected twice**: asking for it does not survive this Action (the rung is
dropped with a `::notice::` as above, and the L2 compile context is a hard `::error::` +
exit 1, "the per-library fan-out never threads the L2 compile context to each pair's header
dump"), and forcing `-H`/`-I` through
`extra-args` past that guard produces a header-aware whole-release run at **29 min and
20.1 GiB peak RSS** per PR, untunable because one process holds every library's AST at
once. Pointing this job at the header-depth baselines is worse: `depth: binary` is a
projection cap, not a filter, so the directory run took **13 min 27 s / 14.6 GiB** and
reported header-side facts the binary side cannot see as changes on an unchanged tree.

**The mechanism has changed and the numbers were re-measured at this pin; the conclusion
has not.** The earlier claim here — 903–1455 breaking `typedef_removed` per library — no
longer reproduces: the pinned `compare/typedefs.py` clears `typedefs`,
`typedefs_qualified` *and* `typedef_entity_ids` when it projects, and the re-run reports
**zero** `typedef_removed`. What replaced it is attribute asymmetry, and it is easiest to
read off a pair that cannot have an ABI change at all — the 2026.0.0 header-depth snapshot
against **the very ELF it was dumped from** (`--depth binary`, 36.3 s / 1.03 GiB):

| finding on a self-identical pair | count | why it appears |
|---|---|---|
| `func_added` / `var_added` | 3975 / 1605 | exports not declared in the public headers; the old side is header-scoped, the new side is the whole export table |
| `func_static_changed` | 148 | `static` is an AST fact; the binary side has none to compare |
| `func_lost_inline` | 49 | same, for `inline` |
| `func_virtual_removed` | 42 | same, for `virtual` — e.g. `~KernelErrorCollection` "is no longer virtual" |
| `func_noexcept_removed` | 16 | same, for `noexcept` |

**190 breaking, verdict BREAKING, exit 4, on two views of one build.** The candidate
(`main`) side of the same shape is 5,325 findings including 439 `func_removed_elf_only`.

It is asymmetry, not projection: the same snapshot compared **against itself** at
`--depth binary` is `COMPATIBLE`, exit 0, one finding (64.5 s / 1.39 GiB). So the
projection is self-consistent, and what breaks is a header-parsed side meeting a
live-extracted side under one requested rung.
So the two baseline families are still required, for a reason now stated correctly: a
header-depth snapshot read at binary depth loses declaration attributes rather than
typedefs, and no policy can reclassify a difference that is an artifact of one side being
pre-parsed and the other extracted live. Upstream ask, sharper than before: a `binary`-depth
comparison should drop the declaration-attribute facts on the header-parsed side the way it
already drops the typedefs, so a pair that is two views of one build cannot come out
BREAKING.

## Header depth

`LinuxAbicheckL2Scan` is the same comparison against the same tag with the public-header
AST added — five legs, one library each, a single snapshot against a single shared object,
the one shape abicheck will *hold* to `--depth headers`. `require-complete-analysis: true`
is what makes "held to" mean something: without it a header set that failed to parse
degrades to symbols-only and still exits 0 under a report saying `headers`. Measured on
every leg: `effective_depth: headers`, `depth_satisfied: true`, `header_context_status:
clean`, `evidence_tier: header_aware`. A green leg means *compared at* header depth, not
*asked for* it.

Measured on `main` against the **published** `2026.0.0` header-depth snapshots, with
`policy.yaml` and `abicheck.yml` in effect, five processes concurrent, under the pinned
CastXML Superbuild (0.6.20260105-g9864b1e) the job provisions:

| library | verdict | exit | breaking | API break | risk | detected | reclassified | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| `libonedal.so` | `API_BREAK` | 2 | 0 | 4 | 2770 | 5205 | 6 | 459 s | 4.55 GiB |
| `libonedal_core.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 3642 | 4470 | 443 | 571 s | 4.55 GiB |
| `libonedal_dpc.so` | `API_BREAK` | 2 | 0 | 4 | 3600 | 6085 | 6 | 471 s | 4.55 GiB |
| `libonedal_parameters.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 89 | 2641 | 0 | 435 s | 4.54 GiB |
| `libonedal_parameters_dpc.so` | `COMPATIBLE_WITH_RISK` | 0 | 0 | 0 | 136 | 2704 | 0 | 422 s | 4.54 GiB |

Both operands resolve out of one header-depth baseline-set (five snapshots, 21,885,624
bytes as `.tar.zst`; capturing it takes 4576 s and 3.6 GiB, most of it `validation:
strict`'s per-library self-compare). Exit 2 is an API break and `fail-on-api-break`
defaults to false, so those legs report without failing the step. **Zero breaking findings
on any leg** is the load-bearing result: header depth does not turn this tree red.

Frontend and operand move counts, not answers. Against a locally dumped `2026.0.0` under
conda-forge castxml 0.7.0 the verdicts, exit codes and four API breaks were identical with
2407 / 3007 / 3271 / 146 / 195 risk at 434–572 s; the clang JSON-AST frontend likewise,
with 3–66 fewer risk findings each and 237–438 s wall.

### What it finds that binary depth cannot

```
experimental_removed_without_replacement ×4 (libonedal.so, and the same 4 in libonedal_dpc.so)
  oneapi::dal::preview::spmd::v1::communicator<...device_memory_access::v1::none>::allgatherv
  …::allreduce
  …::bcast
  …::sendrecv_replace
```

Four `oneapi::dal::preview` SPMD communicator members `2026.0.0` declared and `main` no
longer declares, with no replacement — a source-level break in an experimental namespace,
`recommended_action: recompile_required`. No export table change, so `LinuxAbicheckScan`
cannot see them, and neither can `abidiff` against `main` (the removal predates the
current `main`). This is the whole argument for paying for header depth. Demoting the kind
by policy is deliberately **not** done: four named `preview` removals reported once in an
advisory job beats four silently reclassified. Revisit if `preview` churn makes `API_BREAK`
permanent noise.

The rest of the added volume is intra-version observation: `exported_not_public` (2081 /
2894 / 1363 on `libonedal.so` / `_dpc` / `_core`), 66 `private_header_leak` per leg, 140
`rtti_for_internal_type` on `libonedal_core.so`, plus `public_not_exported`. All `risk`,
none gating — they describe oneDAL's surface, not this PR's change to it, so read them as
backlog ([Known gaps](#known-gaps)).

Header depth is also where the `-fvisibility-inlines-hidden` demotion arrives under a
*second* kind: with a declaration in hand abicheck says `func_visibility_changed` where the
fan-out says `func_removed_elf_only`. That is the one place `policy.yaml` is load-bearing
here rather than inherited. Measured:

* `libonedal_core.so` reports 443 `func_visibility_changed`, all WEAK, and no
  `func_removed_elf_only` at all. Remove the four `func_visibility_changed` rules and this
  leg is `BREAKING`, exit 4 with those 443 breaking, and `libonedal.so` is `BREAKING`, exit
  4 with 6 — which is why the rules exist despite firing zero times on the blocking gate.
* Scoping them to `daal::` alone left `libonedal.so` and `libonedal_dpc.so` `BREAKING` with
  6 breaking each, all WEAK, all `oneapi::`. With the `oneapi` rule both are back to
  `API_BREAK`, zero breaking, and the four `experimental_removed_without_replacement`
  findings still reported — the point of demoting the *linkage* finding and not the *source*
  one. No `sycl` rule for this kind, because no leg reports one.
* `namespace:` rules only: `libonedal_core.so` still `BREAKING` on one finding of 443,
  `_ZNK4daal…NumericTable8getValueIiEET_mm` (`NumericTable::getValue<int>`), whose demangled
  form starts with the printed return type `int`; the mangled counterpart `_ZN[KVR]*4daal.*`
  demotes it. Mangled rules only: both legs identical to the shipped file. So for *this*
  kind the `namespace:` halves match nothing the mangled halves miss today; they stay
  because the removal kind proves the shape is real here (108 `_ZZN4daal…` block-scope
  entities only a demangled walk reaches).

Unlike the fan-out's json, a single-pair json carries per-finding `severity`,
`symbol_binding` and `finding_id` — how these rules were scoped, and how the one
unreachable symbol was found.

### Why five libraries and not six

`libonedal_thread.so` installs no public header of its own. Parsing oneDAL's umbrellas
against it produces 2692 findings — 2190 `public_not_exported`, 334 `private_header_leak`,
151 `exported_not_public` — all saying it does not export what *other* libraries' headers
declare, none about drift since the baseline. Its `analysis_assurance.status` is `partial`,
which `require-complete-analysis` correctly turns into exit 1 every run. A permanently red
advisory leg teaches reviewers to ignore the job, so the library stays covered by the
binary gate (`COMPATIBLE`, one finding) and out of this one, and no `.l2` baseline is
published for it.

### Why it does not block

The environment, not the findings: no leg has ever run on a GitHub runner, the castxml
parse is new to CI, and each leg holds ~4.7 GiB against a 16 GB runner. So
`continue-on-error: true` for a burn-in period, verdict in the job summary and full json
archived per leg. Drop the line once the job has been green across a release cycle.

### Rebuilding an old tag on a runner: pin the oneAPI components

Three failures found by running the publisher on a real runner, the first two the same
shape: **the baseline tag pins oneAPI component versions, `setvars.sh` prefers the newest
installed, and the newest installed is a runtime-only dependency of something else.** The
tag's own `.ci/env/apt.sh` installs them, so the versions are read out of that file rather
than written into the workflow.

* `apt.sh dpcpp` installs `intel-oneapi-compiler-dpcpp-cpp=2025.3.3-30` **and** unpinned
  `intel-oneapi-runtime-libs`, so `compiler/latest` points at a runtime-only tree with no
  driver in `bin/`: `icx: not found`, make Error 127.
* it also drags in runtime-only `intel-oneapi-mkl-core/-sycl 2026.1.0-236` while `apt.sh
  mkl` pins `intel-oneapi-mkl-devel=2025.3.1-8`, so `MKLROOT` resolves to a prefix with no
  static libraries: `No rule to make target
  '/opt/intel/oneapi/mkl/2026.1/lib/libmkl_intel_ilp64.a'`.
* oneTBB is a *provisioning* problem: `.ci/scripts/build.sh` installs it only for the `ref`
  compiler and the arm/riscv64 cross targets, never for 32e with MKL, because on `main`'s
  CI the DPC++ install already carries `tbb/tbb.h`. Elsewhere:
  `threading.cpp:33:10: fatal error: 'tbb/tbb.h' file not found`, 25 minutes in. No pin to
  derive either — `apt.sh` asks for `2023.1.0-151` while the install prefix is
  `/opt/intel/oneapi/tbb/2021.9.0`.

So the publisher sources the two pinned components' own `env/vars.sh` after `setvars.sh`,
selects `TBBROOT` by *capability* (newest prefix actually carrying `include/tbb/tbb.h`),
and asserts before starting a 40-minute build: a resolvable `icx` under the pinned compiler
prefix, `libmkl_intel_ilp64.a` under the pinned `MKLROOT`, and both `include/tbb/tbb.h` and
`lib/libtbb.so*` under the selected `TBBROOT` — the last because `makefile:235/251` derive
the include path from `$TBBROOT` and then filter `LD_LIBRARY_PATH` on that prefix, so a
header-only match still fails at link time. Every assertion prints the versions actually
installed. `LinuxMakeDPCPP` has the same latent skew and misses it only because on `main`
the pinned versions *are* the newest installed.

### The compile context, and why there is no `compile:` block

A header-depth snapshot records its compile context as a **profile fingerprint** and
abicheck refuses to compare across a mismatch, so publisher and consumer must agree on
frontend, language standard and everything else in it — and they agree by *neither*
asserting it. `actions/baseline/run.sh` forwards only
`-H`/`-I`/`--build-info`/`--depth`/`--version`/`--compression`/`-o` and its `action.yml`
has no `build-config` input, so **a `compile:` block cannot reach a baseline-set dump at
all**. Asserting one consumer-side only is a guaranteed mismatch — measured on
`libonedal_parameters.so` with `compile: {frontend: castxml, std: c++17}` still present:

> Error: 'libonedal_parameters.so' old='2026.0.0' new='probe' are not comparable:
> old and new snapshots were extracted under different compile contexts
> (profile_fingerprint mismatch; differing fields: language_standard)

Exit 16, no verdict; with the block removed from both sides, exit 0 and
`analysis_assurance: complete`. Hence `abicheck.yml` carries only `bundle:` and `scope:`,
and the L2 job passes no `ast-frontend` and no `gcc-options`. The toolchain is still
pinned, by the *installer*: `dependency-source: system` provisions through abicheck's own
`install-deps.sh` — the checksum-pinned CastXML Superbuild, **castxml 0.6.20260105-g9864b1e
with clang 21.1.8** — the only castxml `actions/baseline` can provision and therefore the
only one both sides can share. abicheck accepts castxml **>=0.6.11,<0.8.0** and never falls
back silently, so the alternatives are unavailable rather than unchosen (Ubuntu 24.04 apt:
0.6.3, too old; PyPI `castxml`: 0.4.5, "not a supported default scanner setup"). castxml is
also the frontend with layout evidence *in principle* — the clang JSON-AST backend records
no size/alignment/offset — but it buys nothing today: both report
`layout_unverified_detectors: ['dwarf', 'advanced_dwarf', 'layout_descriptor']` on every leg
([Known gaps](#known-gaps)).

**Anything that moves the fingerprint invalidates every published header-depth set,
loudly** — including bumping the abicheck pin far enough to change the installed Superbuild,
since nothing here states the frontend any more. The failure is exit 16 and no verdict
rather than a wrong answer; the fix is re-capturing, i.e. bumping
`ABICHECK_BASELINE_GENERATION` (preferred) or deleting the asset before re-dispatching.
Upstream ask that would make this configuration rather than coincidence: **give
`actions/baseline` a `build-config` input and forward it as `--config`**, the way
`actions/check-target` already does.

## Baselines

The published release binaries are a multi-ISA (`sse2 sse42 avx2 avx512`) release-mode
build while CI compiles single-ISA `avx2`, so comparing against them would compare build
configurations rather than ABI changes. Hence `abicheck-baseline.yml` ("Publish Abicheck
Baseline") checks out the release **tag** and rebuilds it with the same three targets, ISA
and debug setting the PR build uses. It runs on `workflow_dispatch` (`baseline_tag` input)
and `release: published`, and it: builds the tag (`daal`, `oneapi_c`, `oneapi_dpc`); calls
**`actions/baseline`** twice, once per depth, letting it dump, validate and manifest each
family — six libraries at `depth: binary` with `stage_binary: true` (35.4 s, 327 MiB peak
on the largest), five at `depth: headers` with no staged binaries (~23 min, ~2.9 GiB peak),
`validation: strict` round-tripping every fresh snapshot through a self-compare before
packaging; calls **`actions/stage-baseline`** twice, packaging each set as one
`abicheck-baseline-{profile}-gen{generation}.tar.zst`; and uploads the **two archives** as
release assets. It commits nothing and pushes to no branch: publication ends at the
release.

**A baseline-set is not a bag of snapshots.** Each archive carries a `manifest.json`
recording `manifest_version`, `project_ref`, `profile`, `snapshot_schema`, `fact_set`,
`baseline_generation`, a `generator` block, and per artifact a `sha256` plus — for the
binary family — the staged `binary` and its `binary_sha256`; that family also ships a
`binaries/` directory of the tag's real shared objects. `actions/resolve-baseline` checks
all of it, with a typed outcome per failure mode rather than a silent wrong comparison. All
exercised against this exact archive:

| case | outcome | exit |
|---|---|---|
| the archive as published | `resolved` | 0 |
| asking for `project_ref` 2025.9.0 | `wrong_project_ref` | 1 |
| asking for generation 2 | `stale_generation` | 1 |
| asking for the `-headers` profile | `wrong_profile` | 1 |
| a path that does not exist, `required: true` | `not_found` | 1 |
| the same path, `required: false` | `not_found`, `bootstrap=true` | 0 |*
| one staged ELF altered inside the archive | `ambiguous` | 1 |

\* exercised for completeness; neither job uses `required: false`, so this outcome cannot occur
here ([A missing baseline fails](#a-missing-baseline-fails-with-no-opt-out)).

`binary_sha256` is verified per member before the comparison starts, so a one-byte edit to
`binaries/libonedal_parameters.so` fails the job with both digests named; `wrong_project_ref`
catches a cache or asset-name mix-up landing on a set built from a different commit.

**Two families, because neither job can use the other's set.** The header-depth one costs
~44× the wall time to capture, and read at `depth: binary` it is projected down and reports
its header-derived types as removed (see [Binary depth](#binary-depth)). The **`profile`**
keeps them apart and is enforced, not documented: `resolve-baseline` refuses a mismatch
outright (`wrong_profile`, exit 1), where the previous revision's `.l2` filename infix was a
convention nothing checked. Profile names carry the build shape
(`linux-x86_64-icx-avx2-nodbg-…`) because a version number alone would let an avx2/no-debug
baseline be compared against an avx512 or debug build.

**Generations are the second half of that identity.** `baseline_generation` is bumped when
an abicheck upgrade invalidates published sets — a fixed or newly-extracted fact, a changed
normalization or hash recipe — and *not* for report-format, policy or detector-only changes.
It appears in the manifest and both archive names, so a bump publishes beside the old assets
instead of colliding with immutable ones. **What the generation check does and does not
catch**, corrected from an earlier overstatement here: `stale_generation` fires when the
archive a consumer *loaded* declares a generation other than the one it asked for. It does
not discover that a newer generation exists — a consumer pinned to `1` that finds the
retained gen-1 asset resolves cleanly and keeps comparing against gen-1 facts. So publishing
gen 2 does not fail gen-1 consumers; it leaves them behind silently, and rotating both jobs'
`ABICHECK_BASELINE_GENERATION` to match the publisher's is the step that moves them. What
`stale_generation` does protect against is the mismatch inside one selection: an asset name
promising one generation and a manifest declaring another.

**The baseline is a release artifact, and only a release artifact.** Nothing generated by
the publisher is committed here. `LinuxAbicheckScan` asks the release API for the asset it
needs, reads that asset's own `digest` (`sha256:<hex>`, recorded by GitHub at upload), and
verifies the download against it before handing the archive to `resolve-baseline`. The
failure it can report is "the required release baseline asset is unavailable" — a statement
about the release, not about which files a branch happens to carry. An earlier revision
committed the binary archive's `sha256sum` to `.github/abicheck/baselines/<tag>.abicheck.sha256`
and had the publisher `git push origin HEAD:main` to place it; that is removed. What it bought
was an anchor outside the bytes an attacker with release-write access would replace — the
manifest digests every artifact but travels inside those same bytes, so it cannot anchor
itself, and the release-recorded digest is rewritten with the asset. What it cost was a
generated file in the repository, push access to the default branch from a release-triggered
workflow, and consumer state whose only purpose was getting that file onto `main`. The
weaker trust model is accepted deliberately; the stronger form is a release property, not a
repository one — **immutable releases** plus **artifact attestations** — and immutable
releases need a draft → attach assets → publish shape, which this publisher's
attach-after-publish flow does not yet have.
`LinuxAbicheckL2Scan` cannot even verify that much: it fetches inside the Action through
`abi-baseline`/`baseline-asset-name-template`, which offer no interposition point (upstream
ask: an `expected-archive-digest` input on `abi-baseline` — still absent at upstream `main`,
checked).

Publishing the baseline **binaries** was rejected in an earlier revision as a 144 MB
per-release product decision; reversed, for a measured reason — without staged binaries the
bundle half is vacuous and the two kinds needing the old side's real ELF are invisible.
Cost: **~29.5 MB against 491,816 bytes**, one asset per release, zstd-compressed against the
raw 144 MB.

Published assets are **immutable**: a run that would overwrite one stops, because every
merged PR's "compatible with `<tag>`" verdict was computed against the published bytes.
Existence, not a digest compare, is the rule — and has to be, since a snapshot is
content-reproducible but not byte-reproducible (two dumps of the same `libonedal.so` by the
same pin: 87,584 bytes each, different sha256, differing only in `created_at`). A re-run
keeps every asset already on the release, so it converges instead of deadlocking, and reports
the digests the release records for what is now attached.

### A missing baseline fails, with no opt-out

**A required baseline that is missing fails the job.** An earlier revision skipped every
comparison step when the baseline could not be found, producing six `skipped` steps under six
green jobs: a gate that cannot prove it ran is not a gate. Both jobs fail closed, and there is
no bootstrap switch — the earlier `ABICHECK_BOOTSTRAP` opt-in existed to tolerate a missing
*committed digest file* during the window before this workflow reached `main`, and that file no
longer exists. `LinuxAbicheckScan` errors and exits 1 when the release named by
`ABICHECK_BASELINE_TAG` carries no asset for the profile/generation it is about to fetch,
printing the assets that release does carry, since that is what a generation bump without a
re-publish looks like; `resolve-baseline` then runs with `required: true`, so an archive that
downloads but does not resolve is also exit 1. `LinuxAbicheckL2Scan` gets the same guarantee
inside the Action, where `abi-baseline` resolves with `required=True`, so a missing or
wrong-generation asset is `::error::` + exit 1 rather than a skip — visible without blocking.

The consequence is load-bearing and intended: until **Publish Abicheck Baseline** has been
dispatched for `ABICHECK_BASELINE_TAG`, both jobs are red for a reason that names the missing
release asset. Upstream `2026.0.0` currently has **zero** assets, so that one-time publication
is a prerequisite for this gate, not an optional follow-up.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update `ABICHECK_BASELINE_TAG`
in `ci.yml` — in **both** abicheck jobs, which must name the same tag, since a reviewer
reading "compatible with 2026.0.0" from one gate and a different tag from the other cannot
reconcile them. Rotating the *generation* is the same edit in three places:
`ABICHECK_BASELINE_GENERATION` in the publisher and both jobs. Nothing enforces agreement —
what enforces it is that a disagreement fails loudly and specifically, as `stale_generation`
or as the "asset not listed" error above.

### When the baseline operand is not the published one

Every local measurement here is only as good as the bytes it compared against, and that trap
sprang once. The blocking gate was exit 0 locally and exit 4 on its first end-to-end CI run,
on the same policy file, the same pin (checked by re-running both pins locally) and the same
new-side binaries (checked against the runner's `__release_lnx` artifact). What differed was
the *old* operand: a stale local rebuild of `2026.0.0`, all six libraries differing from the
released set's staged binaries. The published `libonedal_dpc.so` exports the two
`sycl::_V1::handler::getRoundedRange<N>` symbols and the stale copy did not, so no local run
could have produced the finding that turned the gate red. Two lessons:

- **Fetch the published asset for local work.** `gh release download <tag> -R <repo> -p
  'abicheck-baseline-<profile>-gen<N>.tar.zst'`, unpack, compare against `binaries/` (binary
  depth) or `<library>.abicheck.json.zst` (header depth) — what `stage-baseline` hands the
  gate, so a local run reproduces CI exactly (verified: exit 4, `libonedal_dpc.so` `BREAKING`
  with 2 breaking, 1949 reclassified, every per-library verdict and count equal to the
  runner's report).
- **Do not diff a policy change against a report taken with a different policy.** The first
  suspicion here was a pin regression, because two L2 legs differed from stored reports by 6
  findings moving breaking → risk; those reports predated a policy rule, not the pin. Pin
  parity is only meaningful with the policy held fixed, and a policy check only with the pin
  and the baseline bytes fixed.

### The abicheck pin and the baseline must move together

All `abicheck` pins must name the **same** commit: `ci.yml` uses the root Action twice and
`actions/resolve-baseline` once; `abicheck-baseline.yml` uses `actions/baseline` twice and
`actions/stage-baseline` twice. `uses:` accepts no expression, so the SHA is written at each
call site and a bump touches all nine — those seven plus the two `generator-git-sha:` inputs
recording which abicheck wrote each family, which a search for `uses: abicheck` misses. The
header-depth family carries the same obligation through its fingerprint, and the pin is now
the only thing stating the frontend.

A snapshot records a `schema_version`, and detectors whose evidence postdates it decline to
run rather than trust stale facts, so a baseline dumped by an *older* abicheck than the reader
silently **under-reports**; the reverse is a hard reject. The loud failure mode is worse and
`schema_version` does not protect against it: a fix in the **dumper** changes recorded facts
without changing the schema — measured while validating an earlier bump, where an
enumerator-value fix made baselines dumped by the previous pin report **280** breaking
`enum_member_value_changed` on an unchanged tree, `schema_version` 25 both sides, and
re-dumping took all 280 to zero. Not that kind at binary depth, which collects no enum facts,
but the mechanism holds for any fact this depth does collect.

So a pin bump obliges re-*verifying* the published baselines: compare the unchanged tree with
the new pin against the current baselines and diff against the old pin's report. All four
bumps so far came out identical line-for-line apart from timings. The fourth, `555905fbe` →
`0e9fdd008` (36 commits), was verified on **both** shapes this gate runs: the blocking gate
exit 0, `COMPATIBLE_WITH_RISK`, 41.54 s, every per-library verdict, reclassified total and
per-(severity, kind) count identical to the old pin; the five L2 legs exits 0 / 0 / 2 / 2 / 0
with 243 / 1857 / 278 / 12 / 17 reclassified over 2441 / 2999 / 3295 / 165 / 214 changes, also
identical per (severity, kind) — on the locally dumped header baseline both pins ran against,
deliberately not the published-set table above, because a parity claim needs identical
operands on both sides. No Action *input* changed either: `--acknowledgments`,
`expected-archive-digest` and `actions/baseline`'s `build-config` are all still absent, so
every gap below stays true.

The baseline-set migration is not such a bump and was not expected to be neutral: the operand
changed from six snapshots to six staged binaries and the report gained findings for a [named
reason](#what-the-multilib-run-compares). Any *unexplained* difference means re-capturing
every published tag's baseline first, under a bumped generation rather than a deleted asset.
If a bump ever produces a wave of findings in one kind across an unchanged tree, suspect this
before suspecting oneDAL. **And rolling a pin back is not symmetric with bumping it**: the
previous pin cannot read a snapshot this one writes (measured — `Cannot detect format`, and
behind that fix the hard `schema_version` rejection, 44 being far past what it understands),
and every baseline-set is written by *this* pin.

## Gating

Two questions, two mechanisms: *what changed* — every finding is in the report, nothing is
filtered out; and *did oneDAL break its ABI* — the verdict, computed after `policy.yaml`
re-classifies findings by kind and ELF linkage. `LinuxAbicheckScan` gates on the folded
release verdict: a binary ABI break (`BREAKING`, exit 4) fails it; a source-level API break
(exit 2) does not, since `fail-on-api-break` stays off. This section describes that job unless
it says otherwise; `LinuxAbicheckL2Scan` runs the same policy over richer evidence but gates
nothing yet. Every number here is measured against the **published** `2026.0.0` baseline-set
bytes with the new side being `LinuxMakeDPCPP(avx2)`'s uploaded artifact — not pedantry, see
[when the operand is not the published
one](#when-the-baseline-operand-is-not-the-published-one).

### The `API/ABI breaking change` label relaxes gating; it does not delete the report

Neither abicheck job is skipped by the label any more. A label in the `if:` is a *skip*, and a
skipped job produces no report, no table and no JSON artifact — on the one PR where the report
matters most. The label feeds the gating inputs instead:

```yaml
fail-on-breaking:        ${{ ! contains(toJson(github.event.pull_request.labels.*.name), '"API/ABI breaking change"') }}
fail-on-removed-library: ${{ ! contains(toJson(github.event.pull_request.labels.*.name), '"API/ABI breaking change"') }}
```

so a labelled PR still runs the full comparison, still publishes the table, summary and
artifact, and merges anyway. `LinuxAbicheckL2Scan` needs no equivalent: it is
`continue-on-error`, so a label has nothing to relax.

**`scope.on_incomplete: block` is deliberately *not* relaxed by the label.** A labelled PR that
also *drops a library* stays red — exit 1 on the completeness axis, not the compatibility one,
because the run cannot prove it compared what it was asked to. The escape hatch is
re-publishing the baseline for a tag that no longer contains that library, i.e. changing what
"the release" means, which is the decision such a PR is making. A library disappearing does
**not** reach `fail-on-removed-library: true`: with no proof that the new side's inventory is
complete, abicheck classifies a baseline snapshot with no counterpart as `not_supplied`
("NEW's inventory is not proven complete, so this is unmatched, not removed", ADR-065 D2) —
measured by deleting `libonedal_thread.so` from the new side: exit 0 with that input set either
way. `scope.on_incomplete: block` is what makes it red, on abicheck's separate *completeness*
axis: the same run exits 1 with `verdict=SCOPE_INCOMPLETE`, an `::error::` annotation and the
unchecked member named. A full six-library release is a complete scope, so a normal run is
untouched, and `fail-on-removed-library` stays set for the case abicheck *can* prove the
inventory complete. The bundle axis gates independently of the per-library verdicts (the run
that made the allow-list necessary was `BREAKING` with all six libraries individually
`COMPATIBLE_WITH_RISK`) and masks nothing in the other direction (the negative control below
still exits 4 with the bundle verdict `COMPATIBLE`).

**`policy.yaml` is load-bearing, and measured to be.** The same comparison with no `--policy`
is `BREAKING` on five of six libraries — 1951 breaking findings (237 + 1414 + 270 + 12 + 18),
exit 4. With it: 0 breaking, exit 0, all 1951 still printed as risk. Every one is
`func_removed_elf_only` on a WEAK symbol: the `2026.0.0` baseline predates `makefile` gaining
`-fvisibility-inlines-hidden`, so `main` stopped *exporting* a large set of COMDAT inline and
template symbols still defined as LOCAL FUNC in the new binaries' `.symtab`. The evidence for
tolerating that, and the exact scope of what the rules give up, is in `policy.yaml` itself.
They should be **deleted** once a post-`-fvisibility-inlines-hidden` release becomes the
baseline, and that is not left to good intentions: every rule carries `expires: 2027-03-01`, an
expired reclassify rule never matches, so on that date the 1951 findings return to `break` and
this gate goes red until someone re-captures the baseline or re-dates the rules with fresh
justification.

**The demotion is bounded by a namespace, not by `.*`.** An earlier revision spelled the linkage
rules as `symbol_pattern: ".*"` plus `binding: weak`, which bounds nothing: it demotes every
weak function removal oneDAL will ever make. (The selector grammar is conjunctive-only and
refuses `binding:` as a rule's sole scope, so some identity selector is mandatory.) It is now
ten rules, two forms of the same three namespaces: `namespace:` rules on the demangled name
(`daal`, `oneapi`, `sycl` for `func_removed_elf_only`; `daal`, `oneapi` for
`func_visibility_changed`) and a mangled-name `symbol_pattern` counterpart for each. Both forms
are load-bearing:

* `namespace:` rules only — two findings stay breaking, exit 4:
  `sycl::_V1::handler::getRoundedRange<1>` and `<2>`, whose demangled spelling begins with a
  printed return type (`std::tuple<sycl::_V1::range<1>, bool> sycl::…`), so the first
  `::`-segment is `std::tuple<…> sycl` and no `namespace: sycl` rule reaches them. Adding the
  mangled counterparts takes 1949 demoted → 1951, exit 4 → 0.
* mangled rules only — `libonedal_core.so` comes out `BREAKING` with 108 breaking findings,
  every one a `_ZZN4daal…` symbol: a block-scope static or lambda inside a function body, whose
  mangling starts `_ZZ` rather than `_ZN`, which the demangled walk resolves and a
  `_ZN`-anchored prefix cannot.
* the mangled form's bound is the namespace and nothing wider, checked against real data: of
  the **13,485** mangled symbols the published baseline's six libraries define, the three
  patterns fullmatch 6163 / 4099 / 17, and every one really is in the namespace its rule names.
  (`symbol_pattern` is a regex `fullmatch` against the *mangled* symbol —
  `abicheck/policy/selectors.py`, `_symbol_matches` — which is why each pattern ends in `.*`,
  and why a demangled-looking pattern such as `'^[^(]* sycl::'` matches nothing.)
* injecting a *new* weak function removal into the baseline snapshot — into the declarations
  inventory as well as `.dynsym`, so the detector sees it — gates outside the three namespaces
  (`otherproj::foo::bar()`: breaking, exit 4) and is demoted inside one
  (`daal::brandnew::foo()`: risk). That is the honest limit of the grammar. Closing it needs
  per-finding acknowledgments, which abicheck implements engine-side and exposes to neither CLI
  nor Action ("no `--acknowledgments` CLI flag yet", ADR-067); sized for the upstream ask, 1951
  findings over ~1689 unique symbols, so the useful form is a bounded acknowledgment (component
  + kind + cause + release range).

**Two kinds, because the kind depends on the depth.** A revision of this file deleted the
`func_visibility_changed` rules on the grounds that they fired zero times — true of the blocking
gate, false of the advisory one, where the same cause arrives as *visibility changed* rather
than *export removed* (443 on `libonedal_core.so`, 6 on `libonedal.so`, both `BREAKING`/exit 4
without the rules, [details](#header-depth)). A rule's blast radius has to be measured at
**every depth the file is passed to**, and this file is passed to both jobs.

**The matching defect behind the mangled rules.** `namespace:` walks the *demangled* name's
ancestor chain, and a function template's demangled form starts with a printed return type —
`auto& oneapi::dal::…`, `int daal::…`, and at least one shape where the return type wraps the
name entirely (`bool (*daal::data_management::internal::getVector<int>())(…)`). The first
`::`-segment is then `auto& oneapi`, not `oneapi`: abicheck strips template arguments before
that walk but not a return type. It fails *closed* — the finding stays breaking — so it is a
usability defect, not a hole. An earlier revision named six such symbols individually and
recorded that "a seventh would turn the gate red"; a seventh and eighth did, on the first
end-to-end CI run (the two `getRoundedRange` symbols, `BREAKING`, exit 4, 2 breaking on
`libonedal_dpc.so`, every other sycl removal in the same report demoted). An enumeration that
grows whenever the compiler emits one more inline is a maintenance queue, not a bound; the
mangled-prefix rules state the same bound in a spelling the return type cannot displace.

**The policy is accountable rather than a blanket mute, and the linkage scoping is what makes it
so.** Negative control on the shipped shape: take one tolerated symbol in the baseline snapshot
— `oneapi::dal::v1::exception::~exception()`, WEAK, no longer exported — and flip only its
recorded linkage to `global`. `libonedal_parameters.so` moves to `BREAKING` with exactly one
breaking finding (risk 13 → 12, so that finding changed bucket and nothing else did) and the run
exits 4. A STRONG export disappearing gates; a WEAK one is reported and does not. A
**suppression** file was tried first and rejected: it removes the change *before* the verdict and
counts are computed and leaves no trace in text output — not the finding, not a count, not even a
note that a suppression file was in effect. The failure mode is silence, not noise: a synthesized
STRONG-symbol visibility regression produced a run whose visible finding set was identical to a
clean one's, exit 0.

Five rules earlier revisions carried are gone with the header-scoped shape they were measured
against (three named types' `type_vtable_changed`, `oneapi::dal::preview`'s
`experimental_removed_without_replacement`, one internal `constant_changed`): at binary depth no
type, layout, enum or constant facts reach the comparison, so they were config that could never
fire, and *none of those five kinds appears* in the header-depth job's findings on the current
tree either. Restoring any means measuring it against that job's report first, not recovering it
from git history.

Two abicheck behaviours worth knowing before reading a job summary. The Action's `verdict` output
is one word for the whole run — at this pin the real one (`COMPATIBLE_WITH_RISK`, and
`SCOPE_INCOMPLETE` for the missing-library case; the pin before folded both to `COMPATIBLE`) —
but the per-library verdicts and the bundle row exist only in the report the job pastes into the
summary. And per-library rows are keyed by the *snapshot* filename
(`libonedal_core.so.abicheck.json.zst`), not the shared object's.

`require-complete-analysis` is deliberately **not** set on `LinuxAbicheckScan` and deliberately
**is** set on every L2 leg: the Action gates on it only for a single-pair compare (or `scan
--against`), which is the L2 shape and not the release shape, and on L2 it is the point. The one
library it would fail for the wrong reason is not in that matrix
([why](#why-five-libraries-and-not-six)). Until the `risk` bucket has had a burn-in period, treat
a change in these counts as something to read rather than a regression in itself.

## Why not abicheck's declarative project configuration

`abicheck.yml` here is not that configuration: three keys — two settings abicheck demoted off its
CLI and Action, one it refuses as inputs on a release operand — and no description of oneDAL's
libraries, baselines or profiles. abicheck's paved road for a multi-library project
(G30/ADR-047) is a `targets:`/`bundles:`/`profiles:`/`baseline:` block in `.abicheck.yml`,
consumed by the reusable `check-project.yml` and `publish-baseline.yml`. Two of the four
obstacles an earlier revision documented are moot for the way header depth is reached here
(`public_headers:` never reaching a run-plan cell, and `bundles:` being restricted to `depth:
binary`, are both about the multi-library shape; the L2 job is single-pair, names its headers as
inputs and joins no bundle), and the third, `-fsycl`/`-DONEDAL_DATA_PARALLEL` being
inexpressible, is side-stepped rather than solved — that surface is out of scope for either job
([Known gaps](#known-gaps)).

What remains is the build contract. Both reusable workflows are "build once, scan many": each
expects one `<prefix><profile-id>` artifact per contract profile containing a
`build-output.json` plus the binaries it references (G30 P1.1), and oneDAL's makefile build emits
no such manifest. Producing one means oneDAL-owned build glue — the class of thing this revision
exists to remove — so both workflows keep calling the root Action directly. Everything the paved
road expects to be portable is already here: release-asset baselines verified against the
release's own recorded digest, a release-triggered (never `pull_request`) publisher, SHA-pinned
Actions, abicheck's own
Action rather than a project driver, and policy over suppression. If oneDAL's build ever emits
`build-output.json`, this collapses into two `workflow_call` jobs.

## Known gaps

* **What the baseline-set saves is the *build*, not the parse.** Both sides of the binary job
  *are* re-derived from real ELFs, deliberately. What the published baseline buys is not having
  to *rebuild* the tag with the pinned oneAPI toolchain inside every PR job, the expensive
  (~20 min) and fragile part. Do not drop it in favour of building the tag in the PR job.
* **The SYCL surface is parsed by nothing.** Everything behind `ONEDAL_DATA_PARALLEL` needs a
  DPC++ frontend; castxml is the only frontend carrying record layout and is not one. So the two
  `_dpc` libraries are compared at header depth on their non-SYCL declarations plus their full
  export table, and the SYCL-only public API is covered by neither gate (`abidiff` sees its
  symbols, not its declarations). It is an absent compiler, not an absent abicheck feature: a
  `_dpc` leg *can* ask for a device-context parse and is refused for a concrete reason each time
  — with castxml, "requires the clang header backend … castxml has no SYCL/DPC++ host/device
  context concept"; with clang and a plain `clang++`, "requires a DPC++-capable compiler
  (icx/icpx/dpcpp/dpcpp-cl)". Covering it needs `icpx` installed in the job and pointed at
  through `gcc_path`/`frontend` — a cost and install-surface decision.
* **Header depth is advisory, and the risk volume is why it has to earn blocking.** 89 to 3642
  risk findings per library, dominated by `exported_not_public` (up to 2894) — intra-version
  observations, not drift. Zero breaking, so they are a backlog to work down rather than a gate
  to satisfy; adopting them as a gate means a policy pass over the reason buckets first.
* **Layout breaks are outside both abicheck gates.** Every L2 leg reports
  `layout_unverified_detectors: ['dwarf', 'advanced_dwarf', 'layout_descriptor']`, under castxml
  as much as clang, because this build ships no DWARF and neither depth reconstructs record
  layout. A struct that changed size, alignment or member offsets without changing a symbol name
  or declaration is seen by nothing in this repository's CI — `abidiff` consumes the same
  debug-info-less artifact and also falls back to exported symbols. Closing it needs debug info
  in the scanned build or L3 evidence (`--sources`/`--build-info`), which is also what the
  preprocessor axis wants (`ran: false`, `skipped_reason: "no L3 build evidence"`).
* **castxml on a real runner is verified on the publisher side only.** The header-depth numbers
  here were measured locally on a 224-core host, five legs concurrently, so the timings do not
  transfer. Proven on `ubuntu-24.04` is the publisher's five dumps — pinned castxml provisioning,
  the parse, and a ~3.6 GiB working set against 16 GB, in a 2h24m job that published a
  21,827,548-byte set. The unknown is the *consumer* side, which is why that job is
  `continue-on-error: true`.
* **Duplicate type and mangled-symbol names are resolved first-wins.** Each L2 parse emits
  `Duplicate type names skipped (first-wins)` and a `Duplicate mangled symbols skipped` list
  running to hundreds of entries (`operator()` ×160, `operator=` ×155, `SharedPtr<T>` ×11 …):
  abicheck keys its header-derived surface by name, and oneDAL's `interface1`/`v1` versioned
  namespaces plus template instantiations collide heavily. A change confined to a shadowed
  duplicate is invisible to this gate.
* **`scan --artifact-set DIR` remains the cheap fallback** if the baseline assets are ever
  unavailable: all six libraries, no old side, 10.8 s at `--depth binary`. Its residual findings
  are by design, so gate on kinds rather than counts if adopted.
* **The release path records policy provenance, but not per-symbol linkage.** The json stamps each
  finding with `reclassified_by`, carries a `disposition_audit` per library and for the run
  (`reclassified_total: 1951`, `reclassifications: [{rule_id: inlines-hidden-demotion,
  matched_count: 1951}]`) and fills `effective_config_fields` with `policy.base:
  strict_abi@1:<digest>` plus the serialized rule list — a full gap one pin ago, when it reported
  `policy.base: ""` with `--policy` demonstrably applied. What a single-pair comparison still has
  and this does not: per-finding `severity`, `symbol_binding`, `finding_id`. Since every rule is
  linkage-scoped, checking *which* linkage a demoted finding had still needs a local
  single-library rerun, and it is why the acknowledgment ask needs `finding_id` in this shape.
  `findings` is also still capped at 10 per library; `annotations` is not.
* **The bundle axis cannot be silenced or re-classified, only satisfied.** Bundle analysis is
  unconditional (no `--no-bundle-analysis`, no config key) and `policy.yaml` cannot reach it — a
  `reclassify:` entry on `bundle_intra_dep_removed` or `bundle_library_added` is accepted only
  with `to:` in `break, warn, risk, ignore`, and `ignore` on the removal kind is a mute rather
  than an answer. The one supported lever is `bundle.system_providers`, which is why the
  allow-list exists rather than a policy rule, and keeping it current is real maintenance: a new
  `DT_NEEDED` edge outside both abicheck's 43-entry default and this list turns the whole job red,
  and the message will name a dependency removal, not a missing provider.
* **The stored-bundle-facts shape is measured and waiting on Action surface.** abicheck can
  persist the old side of a live release compare as one `BundleFacts` document
  (`--bundle-facts-out`) and take that document as the OLD operand. Measured on this release pair
  at binary depth: capture 43.5 s / 655 MiB, **469,852 bytes** as `.json.zst` (smaller than the
  six snapshots' 487,979), stored-vs-live compare **exit 0 in 24.0 s / 311 MiB** against 40.4 s /
  645 MiB, identical per-library verdicts, and bytes that are sha256-identical whether captured
  against the release tree or as a side-output of the PR-side comparison. It buys
  `comparison_scope.old_inventory` becoming `completeness: "proven"` — which would make
  `fail-on-removed-library: true` mean something and let `scope.on_incomplete: block` stop
  standing in for it — plus per-library single-pair-grade reports (untruncated changes with
  `severity`, `symbol_binding`, `finding_id`, `reclassified_by`, `disposition_audit`,
  `effective_config_*`), i.e. the two gaps above. In exchange the document's *root* changes shape
  (`per_library_verdict`, `not_comparable_members`, `old_bundle_facts`, `extraction_failures`
  replacing `changed_libraries`, `unmatched_*`, `old_dir`, and the root
  `exit`/`scope`/`disposition_audit`/`effective_config_*`); the Action already reads
  `comparison_scope`'s `*_exit_contribution` instead of the root `exit`, so the scope axis still
  gates, but anything else read off the root has to be re-pointed. Three things hold it back: the
  Action at this pin exposes **no bundle-facts input at all**, so both the capture flag and the
  stored operand would ride through `extra-args` (upstream ask filed); loading the document needs
  `resource_limits.max_bundle_facts_decode_nodes` raised to at least **2,908,138** against a
  default of 1,000,000 — and the refusal message ("contains more than 1000000 JSON containers")
  under-describes what it counts, since the budget also charges one node per scalar leaf and this
  document holds only 427,934 real containers, so sizing from the message lands 7× low; and
  `--version old=` is rejected against a stored-facts OLD operand, so `old-version` has to be
  dropped on that path. It also adds the 156 `bundle_intra_dep_signature_unverified` and does
  **not** fix `bundle.system_providers`.
* **The same stored shape reaches header depth but cannot be *held* to it, so the five L2 legs
  stay five legs.** A stored-facts OLD side accepts `--depth headers`, and the NEW side takes a
  per-library compile context through `--bundle-facts-library-manifest` (flat keys — `headers`,
  `includes`, `frontend`, `frontend_context`, `gcc_options`, `gcc_path`, `gcc_prefix`, `sysroot`,
  `nostdinc`; no nested `compile:` block and no `defines`/`std`, so `-DONEDAL_DATA_PARALLEL` and
  `-std=c++17` ride in `gcc_options`). The keys work — giving only `libonedal_thread.so` a
  one-struct throwaway header brought that leg back `header_aware`, the other five `elf_only` —
  and that is the problem: libraries with no manifest entry are silently dropped to symbols-only
  *under `--depth headers`*, and `--require-complete-analysis` is refused on this operand ("not
  supported for directory/package (release) comparisons yet (P0.4)"), so nothing enforces the
  floor the single-pair legs exist to provide. The Action refuses it a step earlier anyway:
  `validate-inputs.sh`'s `_is_release_style_operand()` is true when *either* operand is a
  directory. Run end to end anyway: capturing the header-depth old side costs **32 min 52 s /
  14.5 GiB** and produces 26.5 MB of `.json.zst` that decompresses to **1.81 GiB**, over a 1 GiB
  decoded-snapshot ceiling whose only overrides are a private env var and a not-yet-shipped
  container format. Past that, the compare takes **17 min 52 s / 10.0 GiB** and the five
  manifested legs reproduce the L2 matrix exactly, but the run still exits 4 because
  `libonedal_thread.so` — the one library with no manifest entry — was compared header-depth OLD
  against symbols-only NEW and reported **2065 breaking findings on an unchanged tree**, with
  `analysis_assurance: partial` recorded and nothing acting on it. Collapsing the L2 matrix onto
  this path trades a held floor for a silent one; it waits on upstream P0.6.
* **`binding:` is not accepted as a rule's only scope, and there is no per-finding acknowledgment
  loader.** A `reclassify:` entry must name at least one of `symbol`, `symbol_pattern`,
  `type_pattern`, `member_name`, `source_location`, `namespace` or `finding_id`; the grammar
  (`policy_file.py`) has no release or expiry *selector*, only the rule-level `expires:`.
  `Change.symbol_binding` *is* stamped on both the removal and the visibility branch, so `binding:
  weak` works as a conjunct — it just cannot stand alone. The consequence is the residual hole
  measured in Gating: a rule scoped by `namespace: daal` demotes a *new* weak removal in `daal::`
  as readily as a known one. Until acknowledgments ship, `expires: 2027-03-01` is the only
  backstop, and it is a deadline rather than a bound.
* **`actions/baseline` has no `build-config` input, so a baseline-set dump cannot be given a
  compile context** — checked at both the previous pin and current upstream `main`, and stated the
  other way round ("declares it and never reads it") in an earlier revision of this file, which
  was wrong. Hence the header-depth job carries no `compile:` block at all ([compile
  context](#the-compile-context-and-why-there-is-no-compile-block)). Upstream ask: add the input
  and forward it as `--config`, which `actions/check-target` already does with an input of that
  exact name. A per-library options input would additionally collapse the publisher's dump loop
  into one composite call; as it stands each dump step enumerates its libraries and the pin is
  repeated per step, since `uses:` takes no expression.
* **An explicit `baseline-profile`/`baseline-target`/`baseline-generation` is not authoritative
  inside the root Action.** `LinuxAbicheckL2Scan` fetches through the Action's own `abi-baseline`
  path, which downloads `*.abicheck.json[.gz|.zst]` assets **first** and only treats the
  profile-named baseline-set archive as a *fallback* ("Baseline-set fallback: when no single
  `*.abicheck.json` asset was found", `action/run.sh`; same at abicheck `main`). So one legacy
  snapshot asset on the release would be used instead of the explicitly selected set, and two
  would fail the job as ambiguous rather than resolving the set that was named. Second half of the
  same gap: that fallback calls `resolve_target(..., target, profile,
  expected_baseline_generation=...)` and passes **no** `expected_project_ref`, so the L2 path
  cannot enforce the baseline identity the blocking job's own `resolve-baseline` call does
  (`expected-project-ref: 2026.0.0`). oneDAL is not exposed today only because `2026.0.0` carries
  no assets at all. Upstream ask: when profile and target are given, resolve that selection
  directly, validate release ref and generation, and use legacy discovery only when no explicit
  selection was supplied. Not to be worked around here by curating release assets.
* **An import with no provider on *either* side is reported as a removal.**
  `bundle_detectors._detect_intra_dep_removed` computes `ever_provided_in_bundle` — did *this*
  consumer previously reach a version-compatible in-bundle provider — but only consults it inside
  the allow-list suppression branch. A consumer whose remaining `DT_NEEDED` edges are neither
  declared in `bundle.system_providers` nor `_looks_system()` falls through to
  `BUNDLE_INTRA_DEP_REMOVED` with the message "Runtime load of … will fail with undefined symbol",
  even when `ever_provided_in_bundle` is false, i.e. when nothing was ever removed. The detector's
  own docstring concedes the underlying limitation ("absence of a *bundle* regression is not proof
  of a system export"). The correctly-weakened kind already exists —
  `BUNDLE_UNRESOLVED_INTRA_DEPENDENCY` at `COMPATIBLE_WITH_RISK`, used by the audit-mode sibling
  `_detect_unresolved_intra_dependency` precisely because "an audit has no old side to confirm the
  symbol ever resolved". Unchanged at abicheck `main`. This is what makes the four MKL entries in
  `abicheck.yml` load-bearing rather than merely informative: without them a never-in-bundle
  dependency is manufactured into a breaking removal. Upstream ask: when
  `ever_provided_in_bundle` is false, emit the unresolved kind, not the removal — weaker evidence
  should narrow the conclusion, not create a finding. Keep the declarations either way; they are
  true.
* **`fail-on-removed-library` has no effect on a single-pair comparison.** The Action synthesizes
  it into `gate.fail_on_removed_library`, and the only consumer of the resolved value is the
  directory/release dispatch, so on the L2 legs it reached the effective-config digest and nothing
  else. Removed from those legs rather than left as a decorative assertion; it stays on the
  blocking job, whose operands are directories. Upstream ask: reject or warn on an input that
  cannot apply to the operand shape, the way the depth and evidence inputs already do.

`ci.yml` and `abicheck-baseline.yml` both point here rather than repeating the rationale. If you
change the pinned commit, the baseline storage, the policy or the shape of the comparison, update
this file in the same PR.
