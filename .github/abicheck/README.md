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

# ABI checks

Two abicheck jobs in `../workflows/ci.yml` compare a PR's Linux build with the
published baseline for the last release (`2026.0.0`). abicheck is pinned to
`44d86bc112ebe606d0653fb7e9c9e1bd63f37e23` in both jobs and in the baseline workflow.

| File | Role |
|---|---|
| `abicheck.yml` | build config for both jobs (`bundle.system_providers`, `scope.on_incomplete: block`) |
| `policy.yaml` | `strict_abi`, `unused_rules: warn`, and the reclassify rules below |
| `../workflows/abicheck-baseline.yml` | builds a release tag and publishes the baseline assets |

## What the gate does

* **`LinuxAbicheckScan` (blocking).** Compares all six release libraries as a bundle at
  binary depth (ELF exports). A BREAKING verdict (exit 4) fails the job. API_BREAK
  (exit 2) does not. The `API/ABI breaking change` label relaxes the gate, but the
  comparison still runs.
* **`LinuxAbicheckL2Scan` (advisory, `continue-on-error`).** Five single-library legs at
  header depth (castxml over `dal.hpp`/`daal.h`). `libonedal_thread` has no public header,
  so it is covered by the blocking job only.

A partial analysis fails closed. An incomplete bundle is blocked by `on_incomplete`. An
L2 leg sets `analysis-assurance-complete: 'true'`, so incomplete header evidence fails it.
Both jobs pin the archive sha256 (`ABICHECK_BASELINE_DIGEST`, `expected-archive-digest`).

## Publish and rotate

Baseline assets are attached to the release they describe:

* `abicheck-baseline-linux-x86_64-icx-avx2-nodbg-binary-gen<N>.tar.zst`
* `abicheck-baseline-linux-x86_64-icx-avx2-nodbg-headers-gen<N>.tar.zst`

The publisher writes and the consumers read generation **3**.

* **Publish.** The workflow runs on `release: published`, or by hand:
  `gh workflow run abicheck-baseline.yml -f baseline_tag=<tag>`. It also runs on a fork
  (dispatch only), which uploads to the fork's release of the same tag. Assets are never
  overwritten (no `--clobber`). To re-publish, delete the asset first or bump the generation.
* **Rotate.** Publish for the new tag. Then, in one PR, set both `ABICHECK_BASELINE_TAG` and
  `ABICHECK_BASELINE_DIGEST` pairs in `ci.yml` (the publish job summary lists the digests) and
  delete every rule the audit reports `unused` on all runs.
* **Change the dump** (pin, toolchain or flags). Bump `ABICHECK_BASELINE_GENERATION` in
  `abicheck-baseline.yml`, publish, then move `ci.yml` to it with the new digests. A pin
  bump needs this when abicheck changes what it extracts, even if the old sets still load.

The build sources the compiler, MKL and TBB versions pinned in the tag's `.ci/env/apt.sh`.
Plain `setvars.sh` picks the newest installed versions, and with those MKL 2026.2 lacks
`libmkl_intel_ilp64.a`.

Measured cost of the gen2 publish (`ubuntu-24.04`, run 39380873673 on napetrov/oneDAL, 1 h 03 min; the gen3 run 39906849058 took 45 min 52 s in total):

| Step | Time |
|---|---|
| Build daal + oneapi_c + oneapi_dpc | 48 min |
| Binary dump | 57 s |
| Header dump | 10 min 45 s |

Compares against those assets at 44d86bc11 (local, 5 L2 legs concurrently, cold cache as in CI):

| Step | Time | Memory |
|---|---|---|
| Blocking compare | 12 s | 270 MiB |
| L2 leg (each) | 1:17-1:24 | 1.07-1.18 GiB |

Gen3 assets on the 2026.0.0 release (git_commit c2d1c4e19, icx 2025.3.3, generator 44d86bc11, schema 63):

| Asset | Bytes | sha256 |
|---|---|---|
| binary | 27,630,706 | `d3b5a2095817287551e9e0d0ba3106baaacdc42d3570e73d54da47ed57a24ecf` |
| headers | 10,304,115 | `98573c29e2703366cb4f442fbda29dbd5de72243287a4a70e3d2a700fd4ef505` |

## Rules

Every rule demotes findings to `risk` and expires on 2027-04-01. The json's
`disposition_audit.reclassifications` gives each rule's `id`, `status` and `matched_count`.
`unused_rules` is `warn`, not `block`: all runs share this file, and every rule except
`weak-func-hidden` matches nothing on at least one of them.

| id | Matches 2026.0.0 → main | Evidence |
|---|---|---|
| `weak-func-hidden` | 2001 bin, ≤659 L2 | #3604 `-fvisibility-inlines-hidden` (867c643b9, from 2026.1.0) hides WEAK inline/template symbols. GLOBAL removals still break. |
| `weak-var-hidden`, `weak-var-hidden-elf` | 24 | same; WEAK template statics |
| `weak-inst-hidden` | 42 | same; WEAK instantiations |
| `weak-template-leak` | 102 | same; restates hidden WEAK templates |
| `weak-visibility-hidden` | ≤442 L2 | same; default→hidden on WEAK |
| `csr-setarrays-still-declared` | 1 per L2 leg | abicheck false positive: `CSRNumericTable::setArrays<T>` is still inline in `csr_numeric_table.h`; `func_removed` on core, `inline_function_removed` elsewhere, so no kind |
| `threader-internal-3471` | 16 bin, 3 L2 | `^_daal_(static_)?threader_for.*`: internal C-linkage entry points of `libonedal_thread`, no installed header; same deselection as `.github/.abignore`. #3471 (fdd4e9528) moved them to int64; accepted. |

All `weak-*` rules retire once the baseline is 2026.1.0 or later.

Controls run against the final policy (patched ELFs):

* a WEAK→GLOBAL binding flip of a removed symbol exits 4 on both gates;
* removing a STRONG export exits 4 on both gates.

## Known gaps

* **Binary-only API findings are demoted, not checked.** The blocking job has no headers,
  so abicheck reports 10 `mandatory_template_param_added` on internal `radix_sort_indices_inplace`
  as risk (`binary_gated_api_break`); only the L2 legs see source breaks.
* **The SYCL surface is parsed by nothing.** castxml is not a DPC++ frontend. Declarations
  behind `ONEDAL_DATA_PARALLEL` are seen only as exported symbols.
* **No layout evidence.** The build ships no DWARF, so a struct size or offset change with
  no symbol or declaration change is unseen (`abidiff` has the same limit).
* **Duplicate names are first-wins** in header parses (`interface1`/`v1` namespaces,
  templates). A change confined to a shadowed duplicate is invisible.
* **Overloaded member templates.** abicheck keeps a still-declared template instance only
  when one primary template has that name; `CSRNumericTable` has two `setArrays<T>`
  overloads, so their instance reads removed. At `--depth headers` the check does not run
  at all, because that depth drops the declaration graph it reads. Hence the csr rule.
* **Snapshots carry no extractor generation.** abicheck does not notice when a stored set
  comes from an older extractor. gen2 header sets spelled an undeduced `auto&` return
  `AutoType&`; read at 44d86bc11, 8 graph `set_*` setters showed false
  `func_return_changed` (BREAKING). gen3 fixes it for us. After an abicheck bump that
  changes the extractor, bump the generation and republish.
* **Bundle findings.** With `system_providers` the bundle check gives 156
  `bundle_intra_dep_signature_unverified` + 37 `bundle_unresolved_intra_dependency`.
  Without it, about 200 more unresolved findings. Both are risk only, so neither affects
  the verdict.
* **Compiler skew between the sides.** The baseline is built with the tag's `apt.sh` pins
  (icx 2025.3.3 for 2026.0.0); the PR side uses `main`'s `apt.sh`. WEAK template
  instantiations differ with the compiler, so `weak-func-hidden` counts move when either pin
  changes.
