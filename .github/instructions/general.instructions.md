---
applyTo: "**"
---

# General Repository Instructions for GitHub Copilot

oneDAL has two C++ interfaces: DAAL (`cpp/daal`, CPU) and oneAPI (`cpp/oneapi`, CPU and SYCL GPU). It is the backend for [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex).

## General Rules

- Use C++17; do not introduce C++20/23 features.
- Preserve the existing interface's ownership and error-handling model. DAAL uses `services::Status` and `SharedPtr`; oneAPI uses exceptions and standard smart pointers.
- Use the oneDAL threading layer rather than TBB directly.
- Check public API and ABI compatibility when applicable.

## Scoped Guidance

- [cpp-coding-guidelines.instructions.md](cpp-coding-guidelines.instructions.md) applies to C++ and DAAL implementation files.
- [build-systems.instructions.md](build-systems.instructions.md) applies to Make, Bazel, and CMake metadata.
- [examples.instructions.md](examples.instructions.md) and [documentation.instructions.md](documentation.instructions.md) apply to their respective content.

For directory-specific constraints and verification commands, read the nearest `AGENTS.md`.