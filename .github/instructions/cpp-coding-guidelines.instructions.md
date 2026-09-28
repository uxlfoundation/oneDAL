---
applyTo: ["**/*.cpp", "**/*.hpp", "**/*.h", "**/*.i", "**/cpp/**", "**/include/**"]
---

# C++ Guidelines for GitHub Copilot

oneDAL has two C++ interfaces with different conventions. Never mix them in one file. C++17 only; no C++20/23 features.

## Interface Conventions

| | DAAL (`cpp/daal/`) | oneAPI (`cpp/oneapi/`) |
| --- | --- | --- |
| Headers | `.h` with `#ifndef __FILE_NAME_H__` guards | `.hpp` with `#pragma once` |
| Namespaces | `daal::algorithms`, `daal::data_management`, `daal::services` | `oneapi::dal`, `oneapi::dal::<algorithm>`; `detail`, `backend`, `preview` sub-namespaces |
| Ownership | `daal::services::SharedPtr<T>` | `std::unique_ptr`, `std::shared_ptr` |
| Errors | Kernels return `services::Status`; the interface layer converts it with `services::throwIfPossible()` | Exceptions from `cpp/oneapi/dal/exceptions.hpp` (`invalid_argument`, `domain_error`, `unimplemented`, ...) with messages from `cpp/oneapi/dal/detail/error_messages.hpp` |
| Kernels | Template bodies in `.i` files under `cpp/daal/src/`, templated on `CpuType cpu` | `cpp/oneapi/dal/algo/<algorithm>/backend/{cpu,gpu}` |

oneAPI error pattern, from `cpp/oneapi/dal/backend/common.hpp`:

```cpp
throw invalid_argument{ dal::detail::error_messages::queues_in_different_contexts() };
```

## Naming

- **DAAL**: classes in `CamelCase` (`BatchContainer`, `HomogenNumericTable`); functions and variables in `lowerCamelCase` (`getResult()`, `nClusters`, `nRowsTotal`); the floating-point template parameter is `algorithmFPType`.
- **oneAPI**: `snake_case` for types, functions, variables and constants (`train_ops`, `uniform_voting`, `get_data()`, `wait_and_throw()`, `row_count`); private members take a trailing underscore (`store_`, `comm_`).

## PR Review Checklist

- Preserve the interface contract: DAAL uses `services::Status`, `SharedPtr`, `.h` headers, and traditional guards; oneAPI uses exceptions, standard smart pointers, `.hpp` headers, and `#pragma once`.
- Check ownership, lifetime, error propagation, type safety, and bounds handling when the changed code makes them relevant.
- Preserve CPU dispatch and avoid introducing C++20/23 features.
- Flag public API or ABI changes unless the change explicitly accounts for compatibility.

## Cross-Reference
- [cpp/daal/AGENTS.md](../../cpp/daal/AGENTS.md) and [cpp/oneapi/AGENTS.md](../../cpp/oneapi/AGENTS.md) - Interface rules for changes
- [general.instructions.md](general.instructions.md) - Repository context
- [build-systems.instructions.md](build-systems.instructions.md) - Build system instructions
- [examples.instructions.md](examples.instructions.md) - Example code patterns
