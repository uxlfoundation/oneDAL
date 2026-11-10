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
libraries (header depth, clang AST) with the first public release of the same major version (the tag in `ABICHECK_BASELINE_TAG`). A binary or source break fails
the job; the `API/ABI breaking change` label lifts that. The abicheck pin lives in both workflows. Config: `.abicheck.yml` (auto-discovered), `policy.yaml` (`policy:` input; a reclassify file has no config key).

## How it differs from the libabigail check

`LinuxABICheck` (libabigail `abidiff`, `.ci/scripts/abi_check.sh`) compares two binaries: this
PR's `.so` files against the baseline build keyed by the parent commit. Its reference moves with
`main`, so a break that landed earlier is invisible once `main` carries it on both sides. It never
parses the headers.

`LinuxAbicheckScan` (this gate) compares the public headers and the exported symbols of this tree
against the published release baseline. Its reference is fixed, so a break stays reported until it
is fixed or accepted in the policy file, and a header edit that silently drops a declaration is
caught even when both binaries were built from that header.

Neither check covers ISAs other than the one built here (`avx2`), or Windows.

## Rotate

* **Publish:** `gh workflow run abicheck-baseline.yml -f baseline_tag=<tag>` (or on `release: published`) uploads the set to that tag's GitHub release.
  Assets are never overwritten; to republish, bump the generation.
* **New tag:** one PR sets `ABICHECK_BASELINE_TAG` and `ABICHECK_BASELINE_DIGEST` in `ci.yml`
  (digest from the publish summary); delete rules `unused_rules: warn` reports.
* **Dump change** (abicheck, pin, toolchain or flags): bump `ABICHECK_BASELINE_GENERATION` in both
  workflows, publish, then move `ci.yml` to the new digest.

## Rules

Three rules demote WEAK finding kinds to `risk`; three name `libonedal_thread` internals; three name header-only `detail` helpers. Other GLOBAL removals block.

| kind | binding | to | Evidence |
|---|---|---|---|
| `func_removed_elf_only` | weak | risk | #3604 `-fvisibility-inlines-hidden` hides WEAK inline and template functions (2026.1.0+). |
| `func_visibility_changed` | weak | risk | Same: default to hidden on WEAK inline functions. |
| `var_removed_elf_only` | weak | risk | Same: WEAK template static data members. |
| `func_removed_elf_only`, `_daal_(static_)?threader_for.*` | any | risk | #3471: C entry points with no installed header (`abi_check.sh` skips the library). |
| `func_`/`var_removed_elf_only`, `thread_pinner_impl_t`, `getCurrentThreadId`, `IMPL` | any | risk | #3987 hid these internals. |
| `header_only_function_params_changed`, `inline_function_removed`, `func_variadic_added` on named `detail` helpers | any | risk | #3471, #3702, #3950 changed header-only `oneapi::dal::detail` functions; nothing exported changed. |

* `symbol_pattern` is a full match; `'.*'` is required (abicheck refuses a weak-binding rule with no selector).
* No `expires`: the rules retire when the baseline reaches 2026.1.0, and rotation deletes them.
  The rules give exit 0 with 0 breaking on the six-library header compare (fork e2e).
* No `require-complete`: the libraries ship no DWARF, so layout evidence is always partial.

## Known gaps

* **No PR delta comment.** The compare is against the release baseline, so a comment would re-report
  everything merged since it; waits for abicheck's main-snapshot delta (abicheck/abicheck#1946).

