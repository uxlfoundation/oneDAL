workspace(name = "onedal")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("@onedal//dev/bazel/config:config.bzl", "declare_onedal_config")
declare_onedal_config(
    name = "config",
)

load("@onedal//dev/bazel/toolchains:cc_toolchain.bzl", "declare_onedal_cc_toolchain")
declare_onedal_cc_toolchain(
    name = "onedal_cc_toolchain",
)

load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "declare_onedal_extra_toolchain")
declare_onedal_extra_toolchain(
    name = "onedal_extra_toolchain",
)

load("@onedal//dev/bazel/deps:opencl.bzl", "opencl_repo")
opencl_repo(
    name = "opencl",
)

load("@onedal//dev/bazel/deps:openblas.bzl", "openblas_repo")
openblas_repo(
    name = "openblas",
    root_env_var = "OPENBLASROOT",
)

load("@onedal//dev/bazel/deps:tbb.bzl", "tbb_repo")
tbb_repo(
    name = "tbb",
    root_env_var = "TBBROOT",
    urls = [
        "https://files.pythonhosted.org/packages/93/4b/2e29f4266be5a66f21fc2dadcde48f9acea86119d6fb1bb2d2a451222ff7/tbb-2022.0.0-py2.py3-none-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/ff/8c/1eb1f856a7a328242d524f1bf64f2a212d46ce5651168e7c8bc7aeaf0f44/tbb_devel-2022.0.0-py2.py3-none-manylinux_2_28_x86_64.whl",
    ],
    sha256s = [
        "15a15a4e3ea4c3f3198bdb3c55fac75c589e15ed2ad0bbb080900d355c5b017e",
        "474e4ed1dce2efeea1d3652e295a97713df5d0ed854c937ee7d0464c38353c36",
    ],
    strip_prefixes = [
        "tbb-2022.0.0.data/data",
        "tbb_devel-2022.0.0.data/data",
    ]
)

load("@onedal//dev/bazel/deps:mpi.bzl", "mpi_repo")
mpi_repo(
    name = "mpi",
    root_env_var = "MPIROOT",
    urls = [
        "https://files.pythonhosted.org/packages/0a/7c/0f4de62a3463e4ebcf232352b231427f3b34c6a0a1b102a94da3246cad76/impi_rt-2021.14.0-py2.py3-none-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/10/9f/4ee3244c078b9e9e8f65ec51760e7a6e52988abba92a285ab8b0c4dbafff/impi_devel-2021.14.0-py2.py3-none-manylinux_2_28_x86_64.whl",
    ],
    sha256s = [
        "f06ac9eba3de9609fb257d714e3041f82334ccfe27a9bec0f90007d6381dd52e",
        "6a6ec66719ac4884a40a0504f4f186f51ee4169bece5f4486c31138ed6bcc87d",
    ],
    strip_prefixes = [
        "impi_rt-2021.14.0.data/data",
        "impi_devel-2021.14.0.data/data",
    ]
)

load("@onedal//dev/bazel/deps:ccl.bzl", "ccl_repo")
ccl_repo(
    name = "ccl",
    root_env_var = "CCL_ROOT",
    urls = [
        "https://files.pythonhosted.org/packages/c4/34/84fcf891faabfcd88e1e054a9268cdbefac8c22ab37d7eea2d9a3bda0f52/oneccl_devel-2021.14.0-py2.py3-none-manylinux_2_28_x86_64.whl",
    ],
    sha256s = [
        "580641c9d296b673d225ed3ca740b3356d7408a5c792de596a8836fde7d6c79e",
    ],
    strip_prefixes = [
        "oneccl_devel-2021.14.0.data/data",
    ]
)

load("@onedal//dev/bazel/deps:mkl.bzl", "mkl_repo")
mkl_repo(
    name = "mkl",
    root_env_var = "MKLROOT",
    urls = [
        "https://files.pythonhosted.org/packages/95/d8/76f53cde7c1df06fcd153b4f6fdf0516aafbfc3239ba8d5a8c354e20bbb2/mkl_static-2025.0.0-py2.py3-none-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/b1/91/b76ab204c03f90d5ce008ba7cf6efd77168059866e96b70277fec959b940/mkl_include-2025.0.0-py2.py3-none-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/b8/d7/ea82194db165d83e22dfedee4d45423477441202e2c321b9e96809d36e63/mkl_devel_dpcpp-2025.0.0-py2.py3-none-manylinux_2_28_x86_64.whl",
    ],
    sha256s = [
        "706f92fcd6e00cc94155097a87528da52b4c3dda4616c8c334963251773a0d13",
        "cf19e274bdd1449ef7285671576c545510bebff669363ee1926779192f618cdd",
        "455281a590920fb58628dbc06ac007f2618c7315e4c04748c2a1b62efa01afb3",
    ],
    strip_prefixes = [
        "mkl_static-2025.0.0.data/data",
        "mkl_include-2025.0.0.data/data",
        "mkl_devel_dpcpp-2025.0.0.data/data",
    ],
)

load("@onedal//dev/bazel/deps:onedal.bzl", "onedal_repo")
onedal_repo(
    name = "onedal_release",
    root_env_var = "DALROOT",
)

http_archive(
    name = "catch2",
    url = "https://github.com/catchorg/Catch2/archive/v3.7.1.tar.gz",
    sha256 = "c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c",
    strip_prefix = "Catch2-3.7.1",
)

http_archive(
    name = "fmt",
    url = "https://github.com/fmtlib/fmt/archive/11.0.2.tar.gz",
    sha256 = "6cb1e6d37bdcb756dbbe59be438790db409cdb4868c66e888d5df9f13f7c027f",
    strip_prefix = "fmt-11.0.2",
    build_file = "@onedal//dev/bazel/deps:fmt.tpl.BUILD",
)
