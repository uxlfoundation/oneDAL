#!/bin/sh
# Workspace status for release builds (`--workspace_status_command`, .bazelrc).
# The stable key changes daily, so no stamped cache entry reuses an older date.
echo "STABLE_ONEDAL_BUILD_DATE $(date +%Y%m%d)"
