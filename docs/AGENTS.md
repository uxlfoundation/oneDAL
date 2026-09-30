# AGENTS.md - Documentation (docs/)

## Purpose
Sphinx sources for the oneDAL documentation. Authoring conventions are in `docs/README.md`.

## Layout
- `source/`: reStructuredText pages; `source/conf.py` holds extensions, substitutions (`|short_name|`) and `extlinks` (`:cpp_example:`)
- `dalapi/`: custom Sphinx extension that renders the C++ API from Doxygen XML
- `doxygen/oneapi/Doxyfile`: Doxygen input is `cpp/oneapi/dal` only; DAAL API pages are written by hand
- `rst_examples.py`: generates `source/examples/{cpp,dpc}/*.rst` from `examples/oneapi/`; those files are gitignored, so don't edit or commit them

## Rules for Changes
- The build runs Sphinx with `-W --keep-going -n`: warnings, including broken references, fail it.
- Cross-reference with explicit targets (`.. _my-page:` and `:ref:`), not headings.
- Record removals and deprecations of public functionality in `source/deprecation.rst`.

## Verification
```bash
cd docs
pip install -r requirements.txt
make html            # what CI runs (.ci/pipeline/docs.yml); make html-github builds without Intel branding
```
Warnings are also written to `docbuild-log.txt`.
