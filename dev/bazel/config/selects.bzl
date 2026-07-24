#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#===============================================================================

"""Central selects for the BUILD_PARAMETERS_LIB layout."""


def parameters_lib_enabled(values):
    """Returns values when parameter libraries use the separate layout."""
    return select({
        "@config//:build_parameters_lib_auto_linux": values,
        "@config//:build_parameters_lib_yes_linux": values,
        "//conditions:default": [],
    })


def parameters_lib_disabled(values):
    """Returns values when parameter objects must be folded into main libs."""
    return select({
        "@config//:build_parameters_lib_auto_linux": [],
        "@config//:build_parameters_lib_yes_linux": [],
        "//conditions:default": values,
    })


def release_dpc_parameters_lib_enabled(values):
    """Returns values when both DPC release and separate parameters are enabled."""
    return select({
        "@config//:release_dpc_build_parameters_lib_auto_linux": values,
        "@config//:release_dpc_build_parameters_lib_yes_linux": values,
        "//conditions:default": [],
    })
