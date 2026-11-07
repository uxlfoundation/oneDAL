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

# ABI gate

`ci.yml` job `LinuxAbicheckScan` compares the public headers and exports of the six shipped
libraries (header depth, castxml) with the `2026.0.0` baseline. It blocks the PR; the
`API/ABI breaking change` label relaxes gating only. abicheck is pinned to v0.8.0
(`597e47ae7cf9657f4ea987175bf9463d70dba594`) in both workflows. Config: `abicheck.yml`, `policy.yaml`.

## Rotate

* **Publish:** `gh workflow run abicheck-baseline.yml -f baseline_tag=<tag>` (or on `release: published`).
  Assets are never overwritten; to republish, bump the generation.
* **New tag:** one PR sets `ABICHECK_BASELINE_TAG` and `ABICHECK_BASELINE_DIGEST` in `ci.yml`
  (digest from the publish summary); delete rules `unused_rules: warn` reports.
* **Dump change** (abicheck, pin, toolchain or flags): bump `ABICHECK_BASELINE_GENERATION` in both
  workflows, publish, then move `ci.yml` to the new digest.

## Rules

Each rule demotes one WEAK finding kind to `risk`. GLOBAL removals still block.

| kind | binding | to | Evidence |
|---|---|---|---|
| `func_removed_elf_only` | weak | risk | #3604 `-fvisibility-inlines-hidden` hides WEAK inline and template functions (2026.1.0+). |
| `func_visibility_changed` | weak | risk | Same: default to hidden on WEAK inline functions. |
| `var_removed_elf_only` | weak | risk | Same: WEAK template static data members. |

* `symbol_pattern: '.*'` is required (abicheck refuses a weak-binding rule with no selector).
* No `expires`: the rules retire when the baseline reaches 2026.1.0, and rotation deletes them.
  The three rules give exit 0 with 0 breaking on the 5-library header compare.

## Known gaps

* **No binary-depth pass.** Symbol-only changes that have no header (`libonedal_thread`) are not checked.
* **No layout evidence.** No DWARF is shipped; struct size or offset changes are unseen.
* **SYCL is not parsed.** castxml is not a DPC++ frontend; `ONEDAL_DATA_PARALLEL` declarations are seen only as exports.
* **First-wins duplicates.** Same-named declarations in `interface1`/`v1` namespaces resolve to the first; 10 warnings remain. A change to a shadowed duplicate is invisible.
* **Overloaded member templates** (`CSRNumericTable::setArrays<T>`): abicheck can misread a removed instance at header depth. The final policy gives 0 breaking on the compare.
* **Snapshots carry no extractor generation.** The directory compare checks it per pair only, so the gate relies on the generation and digest pins in `ci.yml`.
* **Compiler skew:** the baseline is built with the tag's icx; the PR uses `main`'s pin. WEAK instantiation counts move with the compiler.
