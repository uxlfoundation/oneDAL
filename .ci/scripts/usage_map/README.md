# oneDAL usage map

Answers "who runs this code?" for every line and function of oneDAL. It builds
oneDAL with gcov instrumentation, runs each consumer's test suite as a
separate unit, and joins the coverage with a static call graph and the
per-file build cost.

Class of a line or function:

| class | meaning |
|---|---|
| A | run by a real consumer (sklearnex, daal4py estimators, model builders) |
| P | run only through the daal4py public API tests |
| C | run only by oneDAL examples or oneDAL's own tests (shown, filtered out) |
| B | never run, but reachable from run code in the call graph |
| E | never run and unreachable |

Examples and ABI-exported symbols do not count as consumers. They appear as
labels, because they can be removed together with the code they cover.

Outputs (`out-<build>/`): `REPORT.md`, `data.json`, `functions.csv`,
`usage-map.html` (treemap, table, per-component detail, unit heatmap),
`CANDIDATES.md` (removal tiers), `GPU.md` (GPU-only files).

This is a measurement tool for maintainers. It runs on a dev box, not in CI.

## Work directory

All scripts read and write under `USAGE_MAP_WORK` (default: the current
directory):

```
oneDAL-rel/  oneDAL-cov2/  oneDAL-gpu/   oneDAL worktrees, one per build
sklearnex-2026.1/                         sklearnex source (tests, setup.cfg)
venv/                                     CPU consumers
cov-<build>/u/<unit>/                     GCOV_PREFIX per test unit
units-<build>.json  compile-<build>.log   run log, per-TU compile times
data/<build>/                             per-unit coverage, call graph
out-<build>/                              reports and HTML
```

## Environment

CPU (pins used for the published numbers):

```sh
uv venv -p 3.12 venv
VIRTUAL_ENV=$PWD/venv uv pip install scikit-learn-intelex==2026.1.0 daal==2026.1.0 \
    -r sklearnex-2026.1/requirements-test.txt pytest-timeout treelite \
    impi-rt==2021.17.0 mpi4py==4.1.1 pytest-mpi==0.6
export MKL_PREFIX=/path/to/mkl   # build_cpu.sh
```

The scripts load the instrumented `libonedal*.so` with `LD_PRELOAD` in place of
the wheel's copies.

GPU (Intel GPU, oneAPI 2026.1):

- `genv`: micromamba from the Intel conda channel plus conda-forge, with
  `python=3.12 scikit-learn-intelex-gpu=2026.1.0 dal-gpu=2026.1.0 dpctl`,
  `pandas==2.3.3 array-api-strict==2.4.1 pytest-timeout==2.4.0` and
  `impi-rt==2021.17.0 impi-devel==2021.17.0 mpi4py==4.1.1 pytest-mpi==0.6`.
- `cenv`: icpx 2026.1.0 and onedpl-devel. Its `llvm-cov` reads the clang `.gcno`
  files.
- `menv`: `mkl-devel-dpcpp mkl-include mkl-static intel-sycl-rt`, all
  `==2026.1.0` from PyPI. The MKL and SYCL versions must match the compiler:
  MKL 2026.2 requires SYCL 2026.2 and fails with icpx 2026.1.
- `mpienv`: `impi-rt impi-devel ==2021.17.0`, used only as `MPIROOT` for the
  sklearnex build (genv headers break it with `-Werror`).

## Pipeline

```sh
# CPU
build_cpu.sh rel                 # O2, timed through tcxx -> compile-rel.log
build_cpu.sh cov2                # O0, --coverage, -fcallgraph-info
python cost.py rel               # build seconds and object bytes per file
python run_units.py cov2         # every pytest file / C++ example as one unit
bazel_tests.sh                   # oneDAL's own tests, gcc --coverage
python collect_bazel.py bazel_onedal-tests
python collect.py cov2 universe <unit...>
python callgraph.py cov2
python report2.py cov2
python viz.py cov2
python candidates.py cov2

# GPU (host side only; device kernels are not instrumented)
build_gpu.sh && build_sklearnex_gpu.sh
run_gpu_tests.sh && run_gpu_spmd.sh
python collect_gpu.py all
python gpu_report.py
```

## Reading the results

`CANDIDATES.md` sorts components into removal tiers:

- T1 drop: no A and no P lines. Only examples or oneDAL tests run it, or nothing does.
- T2 deprecate: no A lines, but the daal4py public API runs it. Removal needs a
  daal4py deprecation cycle.
- T3 trim: has A lines, but at least half the lines are C, B or E. Look for
  dead branches in `functions.csv`.

Limits:

- Coverage only sees what the test suites run. Users outside the tests are
  invisible, so T2 needs outside evidence such as code search or user reports.
- The CPU classes do not include GPU hits yet.
- Measures size and usage only, not performance.

## Known traps

- Preload `libonedal_dpc.so` without the host `libonedal.so`. With both
  loaded, the host `homogen_table_impl` vtable wins and the DPC code jumps
  to a bad address (segfault in `table2ndarray`).
- Run `sklearnex` and `onedal` with separate pytest calls (`-c setup.cfg`,
  `PYTHONHASHSEED=0`, `SCIPY_ARRAY_API=1`). One `--pyargs` call fails at collection.
- sklearnex/onedal SPMD tests skip on CPU. They need a GPU queue.
- gcc `gcov` cannot read clang `.gcno` files. Use `collect_gpu.py`, which
  parses the `llvm-cov gcov -i` format.
