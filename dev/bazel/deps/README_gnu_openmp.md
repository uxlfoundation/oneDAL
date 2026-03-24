# GNU OpenMP (libgomp) Integration

This module provides GNU OpenMP runtime library (libgomp) for Bazel builds.

## Purpose

`libgomp` is required by MKL when using `gnu_thread` threading layer:
- `libmkl_gnu_thread.so` depends on `libgomp.so`
- Used for OpenMP parallelization with GCC

## Files

- `gnu_openmp.bzl` - Repository rule for fetching/linking libgomp
- `gnu_openmp.tpl.BUILD` - BUILD template with cc_library targets

## Usage in MODULE.bazel

```python
gnu_openmp_repo = use_repo_rule("@onedal//dev/bazel/deps:gnu_openmp.bzl", "gnu_openmp_repo")
gnu_openmp_repo(
    name = "gnu_openmp",
    root_env_var = "GOMPROOT",  # Optional: use local installation
    urls = ["https://anaconda.org/conda-forge/libgomp/14.2.0/download/linux-64/libgomp-14.2.0-h77fa898_1.conda"],
    sha256s = ["1911c29975ec99b6b906904040c855772ccb265a1c79d5d75c8ceec4ed89cd63"],
)
```

## Usage in BUILD files

```python
cc_library(
    name = "my_lib",
    deps = [
        "@gnu_openmp//:gomp",  # Adds libgomp.so and -lgomp linker flag
    ],
)
```

## Sources

- **Conda-forge**: https://anaconda.org/conda-forge/libgomp
- **System**: Automatically detected from GCC installation if GOMPROOT is set
- **Version**: GCC 14.2.0 (compatible with most modern Linux distributions)

## Structure in downloaded package

```
lib/
├── libgomp.so -> libgomp.so.1.0.0
└── libgomp.so.1.0.0
```
