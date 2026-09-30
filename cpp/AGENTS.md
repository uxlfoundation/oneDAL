# AGENTS.md - C++ (cpp/)

## Purpose
The two C++ interfaces: DAAL (`cpp/daal/`, CPU) and oneAPI (`cpp/oneapi/`, CPU and SYCL GPU, primary development focus). Interface-specific rules are in `cpp/daal/AGENTS.md` and `cpp/oneapi/AGENTS.md`.

## Interface Conventions

| | DAAL (`cpp/daal/`) | oneAPI (`cpp/oneapi/`) |
| --- | --- | --- |
| Headers | `.h` with `#ifndef __FILE_NAME_H__` guards | `.hpp` with `#pragma once` |
| Namespaces | `daal::algorithms`, `daal::data_management`, `daal::services` | `oneapi::dal`, `oneapi::dal::<algorithm>`; `detail`, `backend`, `preview` sub-namespaces |
| Ownership | `daal::services::SharedPtr<T>` | `std::unique_ptr`, `std::shared_ptr` |
| Errors | Kernels return `services::Status`; the interface layer converts it with `services::throwIfPossible()` | Exceptions from `cpp/oneapi/dal/exceptions.hpp` (`invalid_argument`, `domain_error`, `unimplemented`, ...) with messages from `cpp/oneapi/dal/detail/error_messages.hpp` |
| Kernels | Template bodies in `.i` files under `cpp/daal/src/`, templated on `CpuType cpu` | `cpp/oneapi/dal/algo/<algorithm>/backend/{cpu,gpu}` |
| Naming | Classes `CamelCase`, functions and variables `lowerCamelCase` | `snake_case` throughout; private members end in `_` |

## Rules for Changes
- Never mix the two interfaces in one file. C++17 only; no C++20/23 features.
- Parallelize through the threading layer (`cpp/oneapi/dal/detail/threading.hpp`, `cpp/daal/src/threading/threading.h`), never TBB directly.
- Keep CPU dispatch intact: optimized code is templated on `CpuType` and selected at runtime.
- Removing anything from a public header needs a deprecation period and an entry in `docs/source/deprecation.rst`. Expected symbol removals go in `.github/.abignore`, which the ABI check reads.
- Don't change copy semantics (shallow vs deep) as a side effect of another change.
- Export macros (`DAAL_EXPORT`, `ONEDAL_EXPORT`) and symbol visibility must stay the same between the Bazel and Make builds.

## Review Checklist
- The interface contract in the table above is preserved.
- Ownership, lifetime, error propagation, type safety and bounds handling, where the change touches them.
- CPU dispatch is preserved and no C++20/23 features are introduced.
- Public API or ABI changes account for compatibility.

## CPU Dispatch

DAAL containers and kernels are templated on `CpuType cpu` (values per architecture in `cpp/daal/src/services/cpu_type.h`). oneAPI operations go through a dispatcher templated on the context (from `cpp/oneapi/dal/algo/kmeans/detail/train_ops.hpp`):

```cpp
template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};
```

Runtime CPU feature flags are in `cpp/oneapi/dal/detail/cpu.hpp`; see also `docs/source/contribution/cpu_features.rst`.

## Dependencies
- Math: Intel MKL (default), OpenBLAS (`BACKEND_CONFIG=ref`)
- Threading: oneTBB, only through the threading layer
- GPU: SYCL; distributed: `oneapi::dal::preview::spmd`

## Namespace Structure

- **oneAPI** (`oneapi::dal`):
  - `backend`: internal, not visible to users.
  - `detail`: visible to users but may change; not ABI-stable.
  - `preview`: experimental functionality; may change or be removed, not ABI-stable.
  - `<algorithm>`, for example `oneapi::dal::kmeans`.
- **DAAL** (`daal`):
  - `algorithms`: algorithms and their `Parameter`, `Input`, `Result` classes.
  - `data_management`: numeric tables and data sources.
  - `services`: error handling, `SharedPtr`, `Collection`.
  - `{...}::internal`: internal, not visible to users.

## Further Reading
- **[AGENTS.md](/AGENTS.md)** - Repository overview and context
- **[cpp/daal/AGENTS.md](/cpp/daal/AGENTS.md)** - DAAL interface specifics
- **[cpp/oneapi/AGENTS.md](/cpp/oneapi/AGENTS.md)** - oneAPI interface specifics
- **[dev/AGENTS.md](/dev/AGENTS.md)** - Build system architecture
- **[dev/bazel/AGENTS.md](/dev/bazel/AGENTS.md)** - Bazel-specific patterns
- **[docs/AGENTS.md](/docs/AGENTS.md)** - Documentation generation
- **[examples/AGENTS.md](/examples/AGENTS.md)** - Example integration patterns
