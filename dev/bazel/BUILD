load("@onedal//dev/bazel:stamp.bzl", "build_date")

# `--stamp`, which the release configs in .bazelrc set, selects the stamped date.
config_setting(
    name = "stamp",
    values = {"stamp": "true"},
)

build_date(
    name = "build_date",
    stamp = select({
        ":stamp": 1,
        "//conditions:default": -1,
    }),
)
