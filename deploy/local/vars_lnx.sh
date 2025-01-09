#!/bin/sh
# shellcheck shell=sh
# shellcheck disable=SC2296

#===============================================================================
# Copyright 2014 Intel Corporation
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Copyright Intel Corporation
# SPDX-License-Identifier: MIT
# https://opensource.org/licenses/MIT


# ############################################################################

# Copy and include at the top of your `env/vars.sh` script (don't forget to
# remove the test/example code at the end of this file). See the test/example
# code at the end of this file for more help.


# ############################################################################

# Get absolute path to this script.
# Uses `readlink` to remove links and `pwd -P` to turn into an absolute path.

# Usage:
#   script_dir=$(get_script_path "$script_rel_path")
#
# Inputs:
#   script/relative/pathname/scriptname
#
# Outputs:
#   /script/absolute/pathname

# executing function in a *subshell* to localize vars and effects on `cd`
get_script_path() (
  script="$1"
  while [ -L "$script" ] ; do
    # combining next two lines fails in zsh shell
    script_dir=$(command dirname -- "$script")
    script_dir=$(cd "$script_dir" && command pwd -P)
    script="$(readlink "$script")"
    case $script in
      (/*) ;;
       (*) script="$script_dir/$script" ;;
    esac
  done
  # combining next two lines fails in zsh shell
  script_dir=$(command dirname -- "$script")
  script_dir=$(cd "$script_dir" && command pwd -P)
  printf "%s" "$script_dir"
)


# ############################################################################

# Determine if we are being executed or sourced. Need to detect being sourced
# within an executed script, which can happen on a CI system. We also must
# detect being sourced at a shell prompt (CLI). The setvars.sh script will
# always source this script, but this script can also be called directly.

# We are assuming we know the name of this script, which is a reasonable
# assumption. This script _must_ be named "vars.sh" or it will not work
# with the top-level setvars.sh script. Making this assumption simplifies
# the process of detecting if the script has been sourced or executed. It
# also simplifies the process of detecting the location of this script.

# Using `readlink` to remove possible symlinks in the name of the script.
# Also, "ps -o comm=" is limited to a 15 character result, but it works
# fine here, because we are only looking for the name of this script or the
# name of the execution shell, both always fit into fifteen characters.

# TODO: Edge cases exist when executed by way of "/bin/sh setvars.sh"
# Most shells detect or fall thru to error message, sometimes ksh does not.
# This is an odd and unusual situation; not a high priority issue.

_vars_get_proc_name() {
  if [ -n "${ZSH_VERSION:-}" ] ; then
    script="$(ps -p "$$" -o comm=)"
  else
    script="$1"
    while [ -L "$script" ] ; do
      script="$(readlink "$script")"
    done
  fi
  basename -- "$script"
}

_vars_this_script_name="vars.sh"
if [ "$_vars_this_script_name" = "$(_vars_get_proc_name "$0")" ] ; then
  echo "   ERROR: Incorrect usage: this script must be sourced."
  echo "   Usage: . path/to/${_vars_this_script_name}"
  # shellcheck disable=SC2317
  return 255 2>/dev/null || exit 255
fi


# ############################################################################

# Prepend path segment(s) to path-like env vars (PATH, CPATH, etc.).

# prepend_path() avoids dangling ":" that affects some env vars (PATH and CPATH)
# prepend_manpath() includes dangling ":" needed by MANPATH.
# PATH > https://www.gnu.org/software/libc/manual/html_node/Standard-Environment.html
# MANPATH > https://manpages.debian.org/stretch/man-db/manpath.1.en.html

# Usage:
#   env_var=$(prepend_path "$prepend_to_var" "$existing_env_var")
#   export env_var
#
#   env_var=$(prepend_manpath "$prepend_to_var" "$existing_env_var")
#   export env_var
#
# Inputs:
#   $1 == path segment to be prepended to $2
#   $2 == value of existing path-like environment variable

prepend_path() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ] ; then   # avoid dangling ":"
    printf "%s" "${path_to_add}"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)

prepend_manpath() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ] ; then   # include dangling ":"
    printf "%s" "${path_to_add}:"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)


# ############################################################################

# Extract the name and location of this sourced script.

# Generally, "ps -o comm=" is limited to a 15 character result, but it works
# fine for this usage, because we are primarily interested in finding the name
# of the execution shell, not the name of any calling script.

vars_script_name=""
vars_script_shell="$(ps -p "$$" -o comm=)"
# ${var:-} needed to pass "set -eu" checks
# see https://unix.stackexchange.com/a/381465/103967
# see https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02
if [ -n "${ZSH_VERSION:-}" ] && [ -n "${ZSH_EVAL_CONTEXT:-}" ] ; then     # zsh 5.x and later
  # shellcheck disable=2249
  case $ZSH_EVAL_CONTEXT in (*:file*) vars_script_name="${(%):-%x}" ;; esac ;
elif [ -n "${KSH_VERSION:-}" ] ; then                                     # ksh, mksh or lksh
  if [ "$(set | grep -Fq "KSH_VERSION=.sh.version" ; echo $?)" -eq 0 ] ; then # ksh
    vars_script_name="${.sh.file}" ;
  else # mksh or lksh or [lm]ksh masquerading as ksh or sh
    # force [lm]ksh to issue error msg; which contains this script's path/filename, e.g.:
    # mksh: /home/ubuntu/intel/oneapi/vars.sh[137]: ${.sh.file}: bad substitution
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: \(.*\)\[[0-9]*\]:')" ;
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then        # bash
  # shellcheck disable=2128,3028
  (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE}" ;
elif [ "dash" = "$vars_script_shell" ] ; then # dash
  # force dash to issue error msg; which contains this script's rel/path/filename, e.g.:
  # dash: 146: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  vars_script_name="$(expr "${vars_script_name:-}" : '^.*dash: [0-9]*: \(.*\):')" ;
elif [ "sh" = "$vars_script_shell" ] ; then   # could be dash masquerading as /bin/sh
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$vars_script_name" | grep -Eq "sh: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash as sh
    # sh: 155: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: [0-9]*: \(.*\):')" ;
  fi
else  # unrecognized shell or dash being sourced from within a user's script
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$vars_script_name" | grep -Eq "^.+: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash
    # .*: 164: intel/oneapi/vars.sh: Bad substitution
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*: [0-9]*: \(.*\):')" ;
  else
    vars_script_name="" ;
  fi
fi

if [ "" = "$vars_script_name" ] ; then
  >&2 echo "   ERROR: Unable to proceed: possible causes listed below."
  >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
  >&2 echo "   Unrecognized/unsupported shell (supported: bash, zsh, ksh, m/lksh, dash)." ;
  >&2 echo "   May fail in dash if you rename this script (assumes \"vars.sh\")." ;
  >&2 echo "   Can be caused by sourcing from ZSH version 4.x or older." ;
  # shellcheck disable=SC2317
  return 255 2>/dev/null || exit 255
fi


# ############################################################################
my_script_path=$(get_script_path "${vars_script_name:-}")
component_root=$(dirname -- "${my_script_path}")/..

__daal_tmp_dir="<INSTALLDIR>"
if [ ! -d $__daal_tmp_dir ]; then
    __daal_tmp_dir=${component_root}
fi

ARCH_ONEDAL=${ARCH_ONEDAL:-$(uname -m)}

if [ "${ARCH_ONEDAL}" = "x86_64" ]; then
    ARCH_DIR_ONEDAL="intel64"
elif [ "${ARCH_ONEDAL}" = "aarch64" ]; then
    ARCH_DIR_ONEDAL="arm"
elif [ "${ARCH_ONEDAL}" = "riscv64" ]; then
    ARCH_DIR_ONEDAL="riscv64"
else
    echo "Unsupported CPU architecture '${ARCH_ONEDAL}'"
    exit 1
fi

if [ "$(basename "${my_script_path}")" = "env" ] ; then   # assume stand-alone
# case "${my_script_path}" in
  # *"env"*)
    component_root=$(dirname -- "${my_script_path}")
    __daal_tmp_dir=${component_root}
    export DAL_MAJOR_BINARY=__DAL_MAJOR_BINARY__
    export DAL_MINOR_BINARY=__DAL_MINOR_BINARY__
    export DALROOT="$__daal_tmp_dir"
    export PKG_CONFIG_PATH="$__daal_tmp_dir/lib/pkgconfig${PKG_CONFIG_PATH+:${PKG_CONFIG_PATH}}"
    export CMAKE_PREFIX_PATH="$__daal_tmp_dir${CMAKE_PREFIX_PATH+:${CMAKE_PREFIX_PATH}}"
    export CPATH="$__daal_tmp_dir/include:$__daal_tmp_dir/include/dal${CPATH+:${CPATH}}"
    if [ -d "${component_root}/include/dal" ]; then
      export LIBRARY_PATH="$__daal_tmp_dir/lib${LIBRARY_PATH+:${LIBRARY_PATH}}"
      export LD_LIBRARY_PATH="$__daal_tmp_dir/lib${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}"
    else
      export LIBRARY_PATH="$__daal_tmp_dir/lib/$ARCH_DIR_ONEDAL${LIBRARY_PATH+:${LIBRARY_PATH}}"
      export LD_LIBRARY_PATH="$__daal_tmp_dir/lib/$ARCH_DIR_ONEDAL${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}"
    fi
  # ;;
else   # must be a consolidated layout
    # within this "else" reference $ONEAPI_ROOT **not** $my_script_path

    if [ -z "${SETVARS_CALL:-}" ] ; then
    >&2 echo " "
    >&2 echo ":: ERROR: This script must be sourced by oneapi-vars.sh."
    >&2 echo "   Try 'source <install-dir>/oneapi-vars.sh --help' for help."
    >&2 echo " "
    return 255
    fi

    if [ -z "${ONEAPI_ROOT:-}" ] ; then
    >&2 echo " "
    >&2 echo ":: ERROR: This script requires that the ONEAPI_ROOT env variable is set."
    >&2 echo "   Try 'source <install-dir>\oneapi-vars.sh --help' for help."
    >&2 echo " "
    return 254
    fi

  # *"etc"*)
    export DALROOT="$ONEAPI_ROOT"
    export CPATH="$ONEAPI_ROOT/include:$ONEAPI_ROOT/include/dal${CPATH+:${CPATH}}"
  # ;;
# esac
fi
