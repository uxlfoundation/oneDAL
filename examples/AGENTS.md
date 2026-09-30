# AGENTS.md - Examples (examples/)

## Purpose
Standalone programs shipped in the release and shown in the docs, one directory per algorithm under `source/`.

## Layout
- `daal/cpp/`: DAAL interface examples; shared helpers in `source/utils/`
- `oneapi/cpp/`: oneAPI CPU examples; shared helpers in `source/example_util/`
- `oneapi/dpc/`: oneAPI SYCL examples, where `main` runs `run(sycl::queue &q)` once per device from `list_devices()`
- `cmake/setup_examples.cmake`: CMake globs `source/*/*.cpp`; `target_excludes.cmake` lists examples to skip per platform
- Input data comes from the top-level `data/` directory via `get_data_path()`

## Rules for Changes
- Each example is self-contained and runnable, and uses one interface; don't mix DAAL and oneAPI code in one file.
- For a new algorithm directory, add it to the `algos` list of `dal_algo_example_suite` (`daal_algo_example_suite` for DAAL) in that tree's `BUILD`. Examples that don't match one algorithm library get their own `dal_example_suite`, like `graph`.
- The docs pull in every file under `oneapi/{cpp,dpc}/source/` via `docs/rst_examples.py`, and link DAAL examples by path with `:cpp_example:`. Renaming or removing an example breaks those references, so update `docs/source/` in the same PR.
- Reuse `example_util` / `utils` helpers instead of adding local printing or data-loading code.

## Verification
```bash
bazel test //examples/oneapi/cpp:kmeans       # one algorithm
bazel test //examples/oneapi/cpp:all          # CI runs this and //examples/daal/cpp:all, with --test_link_mode variants
```
