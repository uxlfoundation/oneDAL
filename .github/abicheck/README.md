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
`026a6109e809f266c303e908e11c7285f17b5a13` in both jobs and in the baseline workflow.

| File | Role |
|---|---|
| `abicheck.yml` | build config for both jobs (`bundle.system_providers`, `scope.on_incomplete: block`) |
| `policy.yaml` | `strict_abi` plus the labelled reclassify rules below |
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
unparsable header set or a profile mismatch makes the L2 leg exit 1 or 16, never 0.
For that reason there is no `require_complete`.

## Publish and rotate

Baseline assets are attached to the release they describe:

* `abicheck-baseline-linux-x86_64-icx-avx2-nodbg-binary-gen<N>.tar.zst`
* `abicheck-baseline-linux-x86_64-icx-avx2-nodbg-headers-gen<N>.tar.zst`

The current generation is **2**.

* **Publish.** The workflow runs on `release: published`, or by hand:
  `gh workflow run abicheck-baseline.yml -f baseline_tag=<tag>`. It also runs on a fork
  (dispatch only), which uploads to the fork's release of the same tag. Assets are never
  overwritten (no `--clobber`). To re-publish, delete the asset first or bump the generation.
* **Rotate.** Publish for the new tag. Then, in one PR, set both `ABICHECK_BASELINE_TAG` lines
  in `ci.yml` and re-check every rule below; most should match nothing and be deleted.
* **Change the dump** (pin, toolchain or flags). Bump `ABICHECK_BASELINE_GENERATION` in
  `ci.yml` and in `abicheck-baseline.yml` together, then re-publish.

The build sources the compiler, MKL and TBB versions pinned in the tag's `.ci/env/apt.sh`.
Plain `setvars.sh` picks the newest installed versions, and with those MKL 2026.2 lacks
`libmkl_intel_ilp64.a`.

Measured cost of the gen2 publish (`ubuntu-24.04`, run 39380873673 on napetrov/oneDAL, 1 h 03 min):

| Step | Time |
|---|---|
| Build daal + oneapi_c + oneapi_dpc | 48 min |
| Binary dump | 57 s |
| Header dump | 10 min 45 s |

Compares against those assets (local, 5 L2 legs concurrently):

| Step | Time | Memory |
|---|---|---|
| Blocking compare | 32 s | 486 MiB |
| L2 leg (each) | 1:00-1:26 | 0.7-0.9 GiB |

Gen2 assets on the 2026.0.0 release (git_commit c2d1c4e19, icx 2025.3.3):

| Asset | Bytes | sha256 |
|---|---|---|
| binary | 27,630,663 | `bcd881da6543d70504a955e2c4a270c3c4c9d1f2994410ebdc6579a58c26dfa6` |
| headers | 10,335,721 | `844399777aa961184fd3ade577929f95abf5987a3d9d9c8310f6436148b4e739` |

## Rules

Every rule demotes findings to `risk` and expires on 2027-04-01. abicheck reports matches
per `label` in the json's `disposition_audit`.

| label | Matches 2026.0.0 → main | Evidence |
|---|---|---|
| `weak-func-hidden` | 2001 bin, ≤659 L2 | #3604 `-fvisibility-inlines-hidden` (867c643b9, from 2026.1.0) hides WEAK inline/template symbols. GLOBAL removals still break. |
| `weak-var-hidden`, `weak-var-hidden-elf` | 24 | same; WEAK template statics |
| `weak-inst-hidden` | 42 | same; WEAK instantiations |
| `weak-template-leak` | 102 | same; restates hidden WEAK templates |
| `weak-visibility-hidden` | ≤442 L2 | same; default→hidden on WEAK |
| `csr-setarrays-still-declared` | 1 per L2 leg | abicheck false positive: `CSRNumericTable::setArrays<T>` is still inline in `csr_numeric_table.h` |
| `threader-int64-3471`, `-rename` | 3 | #3471 (fdd4e9528) changed `_daal_threader_for*` to int64 in `libonedal_thread`. Its only clients are libonedal_core/libonedal of the same release. **Accepted break**; it is in no tag yet. |

All `weak-*` rules retire once the baseline is 2026.1.0 or later.

Controls run against the final policy (patched ELFs):

* a WEAK→GLOBAL binding flip of a removed symbol exits 4 on both gates;
* removing a STRONG export exits 4 on both gates.

## Known gaps

* **A rule that matches nothing is silent.** It appears only in
  `effective_config_fields`, not in `disposition_audit`. Check stale rules by hand at rotation.
* **Residual API_BREAK on the blocking job.** 10 `mandatory_template_param_added` on
  internal `backend::primitives::radix_sort_indices_inplace` in `libonedal_dpc`.
  The job is not blocked because API_BREAK does not fail the job.
* **The SYCL surface is parsed by nothing.** castxml is not a DPC++ frontend. Declarations
  behind `ONEDAL_DATA_PARALLEL` are seen only as exported symbols.
* **No layout evidence.** The build ships no DWARF, so a struct size or offset change with
  no symbol or declaration change is unseen (`abidiff` has the same limit).
* **Duplicate names are first-wins** in header parses (`interface1`/`v1` namespaces,
  templates). A change confined to a shadowed duplicate is invisible.
* **The L2 baseline path does not check identity.** The Action's `abi-baseline` fallback
  prefers any `*.abicheck.json` release asset over the named set, and it passes no
  `expected_project_ref`. Do not attach such assets to releases. Upstream ask filed.
* **Bundle findings.** With `system_providers` the bundle check gives 156
  `bundle_intra_dep_signature_unverified` + 37 `bundle_unresolved_intra_dependency`.
  Without it, about 200 more unresolved findings. Both are risk only, so neither affects
  the verdict.
* **Compiler skew between the sides.** The baseline is built with the tag's `apt.sh` pins
  (icx 2025.3.3 for 2026.0.0); the PR side uses `main`'s `apt.sh`. WEAK template
  instantiations differ with the compiler, so `weak-func-hidden` counts move when either pin
  changes.
