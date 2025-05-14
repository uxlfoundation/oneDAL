#!/bin/sh
#===============================================================================
# Copyright 2014 Intel Corporation
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

# shellcheck shell=sh
# ############################################################################

# Get absolute path/filename of this script.
# Uses only POSIX compliant commands.
#
# Attribution of this rreadlink() function goes to Michael Klement.
# Based on https://github.com/mklement0/rreadlink/blob/master/bin/rreadlink#L125
# Above licensed under MIT license > https://spdx.org/licenses/MIT#licenseText
# See https://stackoverflow.com/a/29835459/2914328 for detailed "how it works."
#
# This POSIX-compliant shell function implements an equivalent to the GNU
# `readlink -e` command and is a reasonably robust solution that only fails
# in two rare edge cases:
#   * paths with embedded newlines (very rare)
#   * filenames containing the literal string " -> " (also rare)

# Usage:
#   script_path=$(rreadlink "$vars_script_rel_path")
#   script_dir_path=$(dirname -- "$(rreadlink "$vars_script_rel_path")")
#
# Inputs:
#   script/relative/pathname/scriptname
#
# Outputs:
#   /script/absolute/pathname/scriptname

# executing function in a *subshell* to localize vars and effects on `cd`
rreadlink() (
  target=$1 fname="" targetDir="" CDPATH=
  { \unalias command; \unset -f command; } >/dev/null 2>&1 || :
  # shellcheck disable=SC2034
  [ -n "${ZSH_VERSION:-}" ] && options[POSIX_BUILTINS]=on
  while :; do
    [ -L "$target" ] || [ -e "$target" ] || { command printf '%s\n' "   ERROR: rreadlink(): '$target' does not exist." >&2; return 1; }
    command cd "$(command dirname -- "$target")" >/dev/null 2>&1
    fname=$(command basename -- "$target")
    [ "$fname" = '/' ] && fname=''
    if [ -L "$fname" ] ; then
      target=$(command ls -l "$fname")
      target=${target#* -> } # delete everything left of first " -> " string
      continue
    fi
    break
  done
  targetDir=$(command pwd -P)
  if   [ "$fname" = '.' ] ;  then
    command printf '%s\n' "${targetDir%/}"
  elif [ "$fname" = '..' ] ; then
    command printf '%s\n' "$(command dirname -- "${targetDir}")"
  else
    command printf '%s\n' "${targetDir%/}/$fname"
  fi
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
  return 255 2>/dev/null || exit 255
fi


# ############################################################################

# Prepend path segment(s) to path-like env vars (PATH, CPLUS_INCLUDE_PATH, etc.).

# prepend_path() avoids dangling ":" that affects some env vars (PATH and CPLUS_INCLUDE_PATH)
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
  # shellcheck disable=2249,2296
  case $ZSH_EVAL_CONTEXT in (*:file*) vars_script_name="${(%):-%x}" ;; esac ;
elif [ -n "${KSH_VERSION:-}" ] ; then                                     # ksh, mksh or lksh
  if [ "$(set | grep -Fq "KSH_VERSION=.sh.version" ; echo $?)" -eq 0 ] ; then # ksh
    # shellcheck disable=2296
    vars_script_name="${.sh.file}" ;
  else # mksh or lksh or [lm]ksh masquerading as ksh or sh
    # force [lm]ksh to issue error msg; which contains this script's path/filename, e.g.:
    # mksh: /home/ubuntu/intel/oneapi/vars.sh[137]: ${.sh.file}: bad substitution
    # shellcheck disable=2296
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: \(.*\)\[[0-9]*\]:')" ;
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then        # bash
  # shellcheck disable=2128
  (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE}" ;
elif [ "dash" = "$vars_script_shell" ] ; then # dash
  # force dash to issue error msg; which contains this script's rel/path/filename, e.g.:
  # dash: 146: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
  # shellcheck disable=2296
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  vars_script_name="$(expr "${vars_script_name:-}" : '^.*dash: [0-9]*: \(.*\):')" ;
elif [ "sh" = "$vars_script_shell" ] ; then   # could be dash masquerading as /bin/sh
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  # shellcheck disable=2296
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$vars_script_name" | grep -Eq "sh: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash as sh
    # sh: 155: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: [0-9]*: \(.*\):')" ;
  fi
else  # unrecognized shell or dash being sourced from within a user's script
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  # shellcheck disable=2296
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
  return 255 2>/dev/null || exit 255
fi


# ############################################################################
my_script_path=$(dirname -- "$(rreadlink "${vars_script_name:-}")")
component_root=$(dirname -- "${my_script_path}")

__daal_tmp_dir="<INSTALLDIR>"
__daal_tmp_dir=$__daal_tmp_dir/dal
if [ ! -d $__daal_tmp_dir ]; then
    __daal_tmp_dir=${component_root}
fi

export DAL_MAJOR_BINARY=__DAL_MAJOR_BINARY__
export DAL_MINOR_BINARY=__DAL_MINOR_BINARY__
export DALROOT=$__daal_tmp_dir
export CPLUS_INCLUDE_PATH=$__daal_tmp_dir/include${CPLUS_INCLUDE_PATH+:${CPLUS_INCLUDE_PATH}}
export LIBRARY_PATH=$__daal_tmp_dir/lib${LIBRARY_PATH+:${LIBRARY_PATH}}
export DYLD_LIBRARY_PATH=$__daal_tmp_dir/lib${DYLD_LIBRARY_PATH+:${DYLD_LIBRARY_PATH}}
export CLASSPATH=$__daal_tmp_dir/lib/onedal.jar${CLASSPATH+:${CLASSPATH}}
export CMAKE_PREFIX_PATH=$__daal_tmp_dir${CMAKE_PREFIX_PATH+:${CMAKE_PREFIX_PATH}}
export PKG_CONFIG_PATH=$__daal_tmp_dir/lib/pkgconfig${PKG_CONFIG_PATH+:${PKG_CONFIG_PATH}}
