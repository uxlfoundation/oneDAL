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
| baseline | the last successful **`main`** build, restored from the Actions cache | the last **release tag**, as one binary-depth snapshot asset per library | the same tag, as one **header-depth** snapshot asset per library |
| operands | two directories, compared library by library by the script | two directories, compared library by library by abicheck's release fan-out | one snapshot against one shared object, per leg |
| evidence | ELF symbols and type info from the two binaries | ELF exported symbols and metadata from the snapshots and the new binaries | the same, **plus** the public-header AST (`evidence_tier: ['elf', 'header']`) |
| what it measures | did this PR change the ABI relative to `main` | has the ABI drifted since the last release | the same, over the declared surface as well as the exported one |
| filtering | `.github/.abignore` (libabigail suppressions) | `.github/abicheck/policy.yaml` (re-classification, see [Gating](#gating)) | the same `policy.yaml` |
| cost | 20 min timeout | 28 s, ~446 MiB | 7–10 min and ~4.7 GiB **per leg** |
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
expensive advisory legs at header depth ([why five](#why-five-libraries-and-not-six)). Both read the same released tag, through
two separate baseline asset families ([Baselines](#baselines)).

## Where everything lives

| path | what it is |
|---|---|
| `.github/workflows/ci.yml`, job `LinuxAbicheckScan` | the blocking check: a single `mode: compare` step with two directory operands, binary depth |
| `.github/workflows/ci.yml`, job `LinuxAbicheckL2Scan` | the advisory check: one `mode: compare` step per library, header depth |
| `.github/workflows/abicheck-baseline.yml` | the baseline publisher: one `mode: dump` step per library **per depth**, eleven in all |
| `.github/abicheck/policy.yaml` | severity re-classification, passed as both check jobs' `policy-file` |
| `.github/abicheck/abicheck.yml` | the project settings that have no usable Action input — the bundle allow-list, the completeness gate, and the L2 compile context — passed as `build-config` |
| `.github/abicheck/baselines/<tag>.abicheck.sha256` | digests of the release assets for `<tag>`, both families — the trust anchor, and the only baseline artifact in git |
| `.github/.abignore` | libabigail suppressions, used by the *other* tool |

There is no oneDAL-owned Python and no oneDAL-owned driver script: all three
abicheck call sites use `abicheck/abicheck` at a pinned SHA and pass everything as
Action inputs — except the settings abicheck either demoted off its CLI and its
Action, or refuses as inputs on a release operand, which is the whole content of
`abicheck.yml`. Nothing that *has* a usable input is repeated there.

## What the multilib run compares

In `LinuxAbicheckScan`, both operands are directories, which is abicheck's own multi-library ("release")
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
the job invokes, with `policy.yaml` and `abicheck.yml` in effect: **exit 0**,
verdict `COMPATIBLE_WITH_RISK`, 2314 findings in total, nothing removed from the
report. Three runs of the same shape: 28.2–29.1 s, 439–446 MiB peak RSS. The table
above is byte-identical to the one the previous pin produced, which is how the
pin bump was validated (see
[The abicheck pin and the baseline must move together](#the-abicheck-pin-and-the-baseline-must-move-together)).

There is no `jobs:` input any more — abicheck removed both the input and the
CLI's `--jobs`, and always auto-detects with a memory clamp. The earlier `jobs: 1`
was not a tuning choice but a fit-the-runner one (205 MiB against tens of GiB),
and giving it up is most of the cost above; the clamp is now what has to hold.

**The bundle allow-list is load-bearing.** The cross-library bundle graph is built
from real ELF binaries, the old side here is six JSON snapshots, and the analysis
is unconditional since abicheck removed `--no-bundle-analysis` with no
replacement. So abicheck has no old bundle to diff against and, without the
`bundle.system_providers` list in `abicheck.yml`, reports 212
`bundle_intra_dep_removed` (an unresolvable `DT_NEEDED` soname reads as an
intra-bundle dependency removal) on top of six `bundle_library_added` — bundle
verdict `BREAKING`, exit 4, on an unchanged tree. Naming MKL's four SYCL
interface libraries as system-provided takes that to bundle verdict `COMPATIBLE`
with the six `bundle_library_added` left, which are inherent to comparing
snapshots against a real bundle and do not gate.

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
and projects it down — so the run measured **13 min 27 s and 14.6 GiB** and then
reported the projected-away header types as removals: **1959 breaking
`typedef_removed` findings on `libonedal_parameters.so` alone**, on an unchanged
tree. Hence two baseline families rather than one, keyed by asset name.

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
and `libonedal_dpc.so`, 12 and 17 on the two `parameters` libraries). Unlike the release fan-out's json, a
single-pair json carries per-finding `severity`, `symbol_binding` and `finding_id`,
so *which* linkage a demoted finding had is now answerable from CI output instead
of a local rerun.

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

### The compile context, and why it lives in `abicheck.yml`

The Action refuses a compile context as *inputs* alongside a release operand
(`::error::` + exit 1) and offers no separate spelling for a single-pair one, so
`compile:` in `.github/abicheck/abicheck.yml` is where the frontend and the
language standard live — read by both workflows, so a baseline and a PR-side parse
cannot drift apart on either. `frontend: castxml` is abicheck's own default,
pinned rather than inherited because a frontend default that moved would invalidate
every published baseline (below), and because it is the frontend with layout
evidence in principle: the clang JSON-AST backend carries no record
size/alignment/offset layout, so it cannot see a struct-layout break, and this build
ships no DWARF for L1 to see it instead. It buys nothing *today* — measured, both
frontends report `layout_unverified_detectors: ['dwarf', 'advanced_dwarf',
'layout_descriptor']` on every leg, so this gate's layout detectors have no evidence
either way and a struct-layout break would go unseen ([Known
gaps](#known-gaps)). `std: c++17` is what oneAPI's public headers require.

castxml is also why `dependency-source: conda-forge` is not optional on any step
that parses a header. abicheck accepts **castxml >=0.6.11,<0.8.0** and never falls
back to another frontend silently: Ubuntu 24.04's apt castxml is 0.6.3 (refused as
too old) and the PyPI `castxml` distribution is 0.4.5 and explicitly "not a
supported default scanner setup". conda-forge's is 0.7.0.

**Editing `compile:` invalidates every published `.l2` baseline, loudly.** A
header-depth snapshot records the compile context it was extracted under as a
profile fingerprint, and abicheck refuses to compare across a mismatch. Measured by
dumping baselines without `std:` and comparing them against a run that sets it:

> Error: 'libonedal.so' old='2026.0.0' new='probe' are not comparable: old and new
> snapshots were extracted under different compile contexts (profile_fingerprint
> mismatch; differing fields: language_standard) — the comparison is not
> comparable.

Exit 16, no verdict, on an otherwise identical tree. That is the *good* failure
mode — the alternative is two sides silently describing different surfaces — and it
is the reason `compile:` lives in one file both workflows read rather than as
per-workflow inputs. Changing `frontend`, `std` or anything else under `compile:`
means re-dispatching **Publish Abicheck Baseline** for every published tag in the
same change, exactly like bumping the abicheck pin
([below](#the-abicheck-pin-and-the-baseline-must-move-together)).

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
2. runs the abicheck Action in `mode: dump` once per library **per depth** (`dump`
   takes one library at a time; only `compare` fans a directory out) — eleven
   steps, measured 0.84 s to 15.5 s each at binary depth (35.4 s together, peak
   327 MiB on the largest) and 262.6–288.7 s each at header depth (~23 min for the
   five run in sequence, ~2.9 GiB peak RSS each);
3. checks a compression tripwire and an asset count, per family: a binary-depth
   snapshot written by this pin is 6.7 KiB to 200 KiB, 477 KiB for the six (the
   already-published `2026.0.0` assets, written by the previous pin, are 5.8 KiB
   to 148 KiB and 372 KB — the sectioned snapshot envelope this pin writes carries
   more per library), and a header-depth one is 4.45–4.53 MB, 22.4 MB for the five
   (a 1.3% zstd ratio against a ~340 MB uncompressed AST snapshot). Hence a 5 MiB
   ceiling on the first family and 32 MiB on the second: one number for both would
   either fail every L2 snapshot or stop catching a binary-depth one written
   uncompressed. The per-family count is asserted too — six and five — so a dump
   step that wrote its `output-file` somewhere unexpected fails here rather than
   going unnoticed;
4. uploads each `<library>.abicheck.json.zst` and each
   `<library>.abicheck.l2.json.zst` as a **release asset**;
5. commits only its `sha256sum` output to
   `.github/abicheck/baselines/<tag>.abicheck.sha256`, one line per asset, both
   families in one file.

**Two families, because neither job can use the other's snapshots.** The
header-depth ones cost 55× the bytes and ~44× the wall time, so putting them on
the blocking job's bill is not free; and a header-depth snapshot read at `depth:
binary` is projected down, which reports its header-derived types as removed
(1959 breaking `typedef_removed` findings on `libonedal_parameters.so`, measured).
The `.l2` infix is what separates them: `LinuxAbicheckScan` hands its baseline
directory to abicheck as a *directory operand*, so anything extra in there is read
as another library to compare, and its fetch step filters those lines out of the
digest list before downloading. `LinuxAbicheckL2Scan` downloads its one snapshot
by explicit name, so the infix costs it nothing.

**The asset name is the matcher, for the binary family.** abicheck keys a snapshot
to a shared object by the filename up to `.so`, so
`libonedal_core.so.abicheck.json.zst` pairs with `libonedal_core.so.4.0` — and a
decorated name (a `<tag>-` prefix, say) pairs with nothing, silently. The
`.json.zst` suffix is load-bearing too: abicheck infers the compression envelope
from it when writing and from the bytes when reading.

The snapshots themselves are deliberately **not** in git. The digest file is under
a kilobyte, is reviewable, and is the trust anchor: both check jobs download
exactly the asset names it lists — all of one family, or one line of the other —
and then run `sha256sum --check --strict`, so a replaced or corrupted asset fails
the job instead of silently changing every PR's verdict. Publishing the baseline
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
both check jobs **skip** and say so in a warning annotation and the job summary —
inert and visibly inert, rather than red (which would block the very PR that
delivers the publishing workflow) or quietly green. Once the digest file is on
`main`, they arm themselves with no further edit.

`LinuxAbicheckL2Scan` treats one more state the same way: a digest file that exists
but lists no `<library>.abicheck.l2.json.zst` line, which is what a tag published
before the header-depth family looks like. Re-dispatching **Publish Abicheck
Baseline** for that tag adds the five `.l2` assets, leaves the six already-published
binary-depth ones untouched (assets are immutable, and existence is the test), and
commits a digest file covering all eleven. No merged PR's verdict changes.

### Rotating to a newer baseline

Dispatch **Publish Abicheck Baseline** for the new tag, then update
`ABICHECK_BASELINE_TAG` in `.github/workflows/ci.yml` — in **both**
`LinuxAbicheckScan` and `LinuxAbicheckL2Scan`. The two must name the same tag: a
reviewer reading "compatible with 2026.0.0" from one gate and a different tag from
the other has no way to reconcile them.

### The abicheck pin and the baseline must move together

The `uses: abicheck/abicheck@<sha>` pin in `ci.yml` (twice) and in
`abicheck-baseline.yml` (eleven times) must name the **same** commit. `uses:`
accepts no expression, so the SHA cannot be shared through an env var and is
written out at each call site; a bump has to touch all of them. The `compile:`
block in `abicheck.yml` carries the same obligation for the `.l2` family, and
enforces it itself — see
[The compile context](#the-compile-context-and-why-it-lives-in-abicheckyml). A snapshot records a
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
against the old pin's. All three bumps done so far came out identical
line-for-line apart from timings, so none needed a re-capture — the current pin
reads the `schema_version` 25 snapshots the previous one published and writes 44
itself, and still reports the same per-library table, the same 2314 findings and
the same verdict. Any difference means re-dispatching the baseline workflow for
every published tag first. If a bump ever produces a wave of findings in one kind
across an unchanged tree, suspect this before suspecting oneDAL.

**Rolling a pin back is not symmetric with bumping it.** The previous pin cannot
read a snapshot this one writes at all — measured, it fails per library with
`Cannot detect format` (its own classifier gave up on a `.json.zst` whose first
frame decodes short, fixed upstream since) and behind that fix it would still hit
the hard `schema_version` rejection, since 44 is far past what it understands. A
bump is therefore one-way for as long as the baselines it will read were written
by the older pin: safe now, because the published `2026.0.0` assets predate this
pin, but the moment the baseline workflow is re-dispatched at this pin, any
rollback means re-dispatching for every published tag *first*, and the
immutability rule makes that a deliberate `gh release delete-asset` per library.

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
  declarations.
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
  name or a declaration is seen by `abidiff` (which reads the two binaries' type
  info) and by nothing else. Closing it means either debug info in the scanned
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
  library (`findings_truncated: true`); the `annotations` array is not.
* **The bundle axis cannot be silenced or re-classified, only satisfied.** Bundle
  analysis is unconditional (no `--no-bundle-analysis` any more, and no config key
  for it), `policy.yaml` cannot reach it — a `reclassify:` entry on
  `bundle_intra_dep_removed` or `bundle_library_added` is accepted only with
  `to:` in `break, warn, risk, ignore`, and `ignore` on the removal kind is a mute
  rather than an answer — and the six `bundle_library_added` findings are
  structural: the old side is snapshots, so every library reads as newly added to
  the bundle, in every PR's report, permanently. They do not gate. The one
  supported lever is `bundle.system_providers`, which is why the allow-list exists
  rather than a policy rule. Keeping it current is real maintenance: a new
  `DT_NEEDED` edge to a library outside both abicheck's 43-entry default list and
  this one turns the whole job red, and the message will name a dependency
  removal, not a missing provider.
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
