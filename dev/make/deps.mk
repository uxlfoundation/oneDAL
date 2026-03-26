#===============================================================================
# Copyright 2012 Intel Corporation
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

# -------------------------------------------
# Incremental building support implementation
# ---
# Depends: DIR, _OS
#
# Consists of two parts:
#   - handling of build command line updates (no rebuild if makefile has been changed but resulting command line is same)
#   - generation of source code dependency information for following build procedures:
#       C.COMPILE
#       LINK.STATIC
#
# Debug options:
# DEBUG_INCR_BUILDING = yes
# 	- to log decisions made on whether target should be recompiled and why
# DEBUG_INCR_BUILDING_EVAL = yes
#   - to log macro enhancements before evaluation
#

.basedir := $(dir $(lastword $(MAKEFILE_LIST)))

# Disable incrementality at all by providing some value to DISABLE_INCR macro:
ifndef DISABLE_INCR

# Debug verbosity
ifdef DEBUG_INCR_BUILDING
.debug.message = $@$(if $(.sources-changed), [UpSrc: $(.sources-changed)])$(if $(.mkfiles-changed), [UpMks: $(.mkfiles-changed)])
.debug.compile = $(info [DEBUG] INCR: EXEC: $(call .debug.message)$(if $(call .trigger-changed,$1), [Trigger: Was:"$($(.trigger-var))" Now:"$1"]))
.debug.skip    = $(info [DEBUG] INCR: SKIP: $(call .debug.message))
.keep-raw-deps = cp $(.dep-file-tmp){,.orig} &&
endif


# ---
# Injects dependency generation into supplied macro
#
# Args:
#   $1: Macro to enhance with dep generation.
#       After injection this macro will (at least) keep dependency of target file on makefiles
#       and monitor whether build command change (i.e. change of its (macro's) own value for particular target file),
#       or change of some other macro (see arg $3)
#   $2: [optional]
#       Line to evaluate and execute instead of original macro.
#       This (if supplied) is supposed to enhance original macro with source code dependency generation.
#       Original macro can be accessed via value of SELF macro.
#       If not set, plain original macro ($1) will be executed.
#   $3: [optional]
#       Line, which evaluated value will be monitored and used to trigger rebuild in case of makefile change.
#       If not set, then change of original macro value (i.e. build command) will be monitored.
#
.inject.dep.gen = $(strip $(call .inject.dep.gen.stripped,$(strip $1),$(strip $2),$(strip $3)))
.inject.dep.gen.stripped = $(call .LE,$1.orig = $(value $1))$(call .LE,$(call .inject.dep.gen.impl,$1,$(if $2,$$(eval $$@: SELF := $1.orig) $2,$$($1.orig)),$(or $3,$$($1.orig))))
.inject.dep.gen.impl =\
$1 = $$(if $$(or $$(.sources-changed),$$(and $$(.mkfiles-changed),$$(call .trigger-changed,$3))),\
	$$(call .debug.compile,$3) $2 && $$(call .save-trigger-value,$3) && $$(call .finalize-dep-gen),$$(strip \
	$$(call .debug.skip)))

# Evaluate GNU Make construct after logging it to console
.LE = $(if $(DEBUG_INCR_BUILDING_EVAL),$(info $1)) $(eval $1)

# ---
# Enhance build commands with support for dependency generation
# (does not depend on macro from which particular command if called, only on command executable name)
dep-gen-enhanced-common   = $(call $(SELF),$1 $(.copt-gen-deps)) && $(.keep-raw-deps) sed -n $(sed.-i) $(sed.fix-deps) $(sed.rm-abs-paths) -e '/./{ p; $(sed.mk-phony-targets)}' $(.dep-file-tmp)
dep-gen-enhanced.icl   = $(dep-gen-enhanced-common)
dep-gen-enhanced.icx   = $(if $(COMPILER_is_vc),,$(dep-gen-enhanced-common))
dep-gen-enhanced.g++   = $(dep-gen-enhanced-common)
dep-gen-enhanced.icpx   = $(dep-gen-enhanced-common)
cmd-enhanced-with-dep-gen = $(or $(dep-gen-enhanced.$(call get-command-name,$($(SELF)))),$($(SELF)))

$(call .inject.dep.gen, C.COMPILE,     $$(cmd-enhanced-with-dep-gen))
$(call .inject.dep.gen, DPC.COMPILE,   $$(cmd-enhanced-with-dep-gen))
$(call .inject.dep.gen, LINK.STATIC)  # TODO: md5 removed because buggy
$(call .inject.dep.gen, LINK.DYNAMIC,  ,$$(call md5,$$(link.dynamic.cmd)))

#
# Dependency generation implementation {

# var name to keep trigger value between make launches
.trigger-var = $@.trigger
# check if trigger value has been changed
.trigger-changed = $(subst $($(.trigger-var)),,$1)
# remember trigger value to compare it on next make launch in case of makefiles changes
.save-trigger-value = printf '\n%s\n' '$(mkdeps-var-name) += $(.mkfiles)' '$$($(mkdeps-var-name)):' '$@: $$($(mkdeps-var-name))' '$(.trigger-var) = $1' >> $(.dep-file-tmp)
# rename temp file to actual dependency holding file
.finalize-dep-gen = mv -f $(.dep-file-tmp) $(.dep-file)

# makefiles currently employed (excluding dependency and list files)
.mkfiles = $(call filter-out-containing,list,$(filter-out %.d %.lst,$(MAKEFILE_LIST)))
# list of.sources-changed
.sources-changed = $(filter-out $(.mkfiles) $($(mkdeps-var-name)),$?)
# list of changed makefiles
.mkfiles-changed = $(filter $(.mkfiles),$?)
# dependency file to be generated
.dep-file = $@.d
# do all dep generation work on temp file, rename it on finish (to survive build halts: DPD200288373)
.dep-file-tmp = $@.d.tmp
# gcc compiler options to generate dependency files
.copt-gen-deps = $(-Q)MMD $(-Q)MT $@ $(-Q)MF $(.dep-file-tmp)

# -------------------------------------------
# SED scripts used for dep files processing {

# Common procedure for almost all tools:
#  - joins backslash ended lines,
#  - processes windows paths,
#  - removes dependency on source (we have this info in makefiles anyway, but in dep files this info breaks build if source moves between the folders)
sed.fix-deps = $(sed.join-backslashed) -e '$(sed.fix-win-paths) $(sed.remove-source); s/  */ /g; s/^.*: *$$//;'
sed.rm-abs-paths = -e 's@ /[^ ][^ ]*@ @g; s/^.*: *$$//;'

sed.join-backslashed := -e ':beg' -e '/\\$$/N; s/\\\n/ /; tbeg'
sed.remove-duplicates := s/\( [^ ]\+\)\1\+/\1/g;
sed.mk-phony-targets = $(sed.mk-phony-targets.$(_OS))
sed.mk-phony-targets.lnx = s/^.*://; s/^ *//; /./{ s/ *$$/:/; s/  */:\n/g; p; };
sed.mk-phony-targets.win = s/^.*://; s/^ *//; /./{ s/ *$$/:/; s/  */:\n/g; p; };
sed.mk-phony-targets.mac = s/^.*://; s/^ *//; /./{ s/ *$$/:/; s/  */:\'$$'\n''/g; p; };
sed.remove-source = s@$<\( \|$$\)@@

# } sed scripts
# } dependency generation implementation

endif

# ----------------------
# General make tunings {
# ---

# increase make's performance by cancelling make defaults reasonings
# (make's switch is more strong: -r, --no-builtin-rules      Disable the built-in implicit rules.)
.SUFFIXES:

# the following tells make to not delete intermediate (%_gen.c etc) files
#.SECONDARY:

# } tuning make

# ---------------------------------------------------
# Incremental building support implementation notes {
# ---

# 1. Split compound rules to let make to discover dependencies on its own
# 2. Don't explicitly remove intermediate files if you want to keep dependency information!
#    let make itself to decide when to delete intermediate files (see '10.4 Chains of Implicit Rules')
# 3. asm sources does not heavily depend on other files
#    if deps are needed, then first move compilation to yasm (which supports dep files generation)
#    asm rules are in the form: $(ASM) $< $(AOUT)$@
# 4. fortran deps handling needs to be done only for ifort,
#    other fortrans files (gfortran, pgf95) are generated by icl preprocessor which also produces dep's information
# 5. obvious: do not do any cd in target recipes before compilation as it will break relative source path ($<) passed by make to rule recipe
# }
