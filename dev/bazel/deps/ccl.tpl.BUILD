package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    features = [ "dpc++" ],
    hdrs = glob(["include/cpu_gpu_dpcpp/oneapi/*.hpp"],allow_empty=True,) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/*.hpp"],allow_empty=True,) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/*.h"],allow_empty=True,) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/*.hpp"],allow_empty=True,) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/sycl/*.hpp"],allow_empty=True,) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/sycl_l0/*.hpp"],allow_empty=True,),
    includes = [ "include/cpu_gpu_dpcpp/oneapi" ] + [ "include/cpu_gpu_dpcpp/" ],
)

cc_library(
    name = "libccl",
    srcs = [
        "lib/cpu_gpu_dpcpp/libccl.so.1.0",
    ],
)

cc_library(
    name = "ccl",
    deps = [
        ":headers",
        ":libccl",
    ],
)
