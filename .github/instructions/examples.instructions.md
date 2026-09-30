---
applyTo: ["**/examples/**", "**/samples/**"]
---

# Examples Instructions for GitHub Copilot

Read [../../examples/AGENTS.md](../../examples/AGENTS.md) for the example layout, `BUILD` registration and docs coupling.

## Review Focus

- Keep each example self-contained, runnable, and limited to one interface: DAAL, oneAPI CPU, or oneAPI DPC++.
- Follow the ownership and error-handling conventions of the selected interface.
- Update the applicable build metadata and validate the example when its source, dependencies, or data change.