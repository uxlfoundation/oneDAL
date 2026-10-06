<!--
  ~ Copyright contributors to the oneDAL project
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# oneDAL agent evals

Tasks and graders for measuring how coding agents do on real oneDAL work: building, fixing bugs, writing tests and
reviewing changes. Comparing *arms* (variants of the repository's agent guidance, tooling or layout) on the same
tasks lets a change to `AGENTS.md`, `.github/instructions`, the build system or the repo layout be judged by the
agent behaviour it produces rather than by reading it.

This directory holds only what is specific to oneDAL: the tasks, the graders that rebuild and rerun oneDAL tests,
the eval image and the oneDAL rules of the guidance claim check. Running agents (workspace preparation, the
harness, isolation, traces, reports) is done by agent-benchmark, which reads `repo-eval.yaml` (the
repository-evaluation contract). No LLM judges a result: graders rebuild and rerun tests.

| tier | what | cost |
|---|---|---|
| T0 static | `run.py static`: every link, path, Bazel label, bazel/make command, Starlark macro and identifier named in the guidance is checked against the tree | $0, seconds |
| self-check | `run.py check`: for every task the untouched workspace fails its grader and the reference solution passes | $0, CPU |
| T1 tasks | agent-benchmark `repo run`: an agent solves each task under each arm; these graders score it | LLM cost + CPU |

## Quick start

```sh
# Python 3.9+, git, bazelisk, uv (for the pinned clang-format)
export ONEDAL_EVAL_SRC=$PWD                     # full-history clone; mined tasks read fix commits from it
python3 dev/agent_evals/run.py list
python3 dev/agent_evals/run.py validate         # task.json shape, bases resolve, patches exist; no build
python3 dev/agent_evals/run.py static           # T0
python3 dev/agent_evals/run.py check            # grader self-check for every task, no LLM
```

Other settings (Bazel binary, shared caches, where `check` writes) are environment variables documented in
`config.py`.

## Continuous integration

`.github/workflows/agent-evals.yml` runs the checks that need no model and no token:

- on every pull request that touches this directory or the agent guidance (`AGENTS.md`, `.github/instructions`,
  `.github/copilot-instructions.md`, `CONTRIBUTING.md`): `run.py validate` (a gate) and `run.py static` (report only,
  as a job summary and an artifact);
- `run.py check` in the eval image, one job per task: on a pull request for the tasks it changes, plus one task per
  grader when the harness or a grader changes; weekly, and on demand (`workflow_dispatch`), for every task.

Agent runs stay manual: they cost tokens and depend on the agent being measured. Rerun them after
a large change to guidance, tooling, build system or layout. A change that breaks `run.py check` for a task updates
that task in the same pull request.
## Running agents

With agent-benchmark (see its `docs/repo-evaluation-contract.md`, "Running"):

```sh
python cli.py repo tasks check <oneDAL>@<sha> --manifest dev/agent_evals/repo-eval.yaml       # self-check in the image
python cli.py repo run <oneDAL>@<sha> --manifest dev/agent_evals/repo-eval.yaml \
    --arm guidance:none,guidance:raw --model <model id> --reps 3 --batch b1
python cli.py repo report repo-runs/b1
```

The framework prepares each workspace (history-free `git archive` of the task base without this directory, setup
patches, the arm, the review patch), runs the agent in its own container, writes `events.jsonl`, `answer.txt` and
`meta.json`, and then calls the entry points below in a separate container:

| entry point | does |
|---|---|
| `run.py grade --contract <run_dir> --source <clone>` | grades `run_dir/workspace` with the task's grader, writes `grade.json` (`repo_grade.v1`) |
| `run.py oracle --contract <run_dir> --source <clone>` | applies the reference solution (and answer) for the self-check |
| `run.py static --out <file>` | guidance claim check as one JSON document (`repo_static.v1`) |

Arms are `guidance:none` (every file matched by `guidance.globs` deleted) and `guidance:raw` (as checked in), both
made by the framework from `repo-eval.yaml`. Fix and feature tasks declare `"gate": ["strict_pass"]`; score-only
review tasks declare `"score": {"metric": "recall", "max": 1.0}`.

## Tasks

`tasks/<id>/task.json`, plus any patches, manifests and hidden files it names. What each task asks, where it comes
from and why it is built that way: [`tasks/README.md`](tasks/README.md).

| key | meaning |
|---|---|
| `base` | commit the snapshot is taken from (`<sha>^` for mined fixes) |
| `family` | `build`, `fix`, `test`, `review` |
| `prompt` | what the agent is told; a final ```` ```json ```` block is requested when the grader needs a structured answer |
| `grader` | name registered by a module in `graders/` |
| `setup_patch` / `setup_patches` | applied to the snapshot before the arm (injected bugs) |
| `review_patch` | `git am`-ed on top as the HEAD commit under review |
| `gate` | metrics that count as success for agent-benchmark (default `["pass"]`); `["strict_pass"]` on every `test_pass`/`test_pass_files` task, since they all have regression suites |
| grader keys | see the docstring of the grader: `hidden_from`/`hidden_files`/`targets`/`regression_targets` for `test_pass`, `manifest.json` for `review`, ... |

Rules every task follows:

- `run.py check <task>` passes: the untouched workspace fails the grader and the reference solution passes it.
- Fixes and answers cannot leak. The snapshot is `git archive` of `base` with no history, and `dev/agent_evals`
  is deleted from it.
- Grading is deterministic: tests are rerun with `--nocache_test_results`, and regrading a run gives the same result.
  Graders write hidden tests over the agent's tree and take library sources out of it; both are undone when
  grading ends, so the run directory holds only what the agent did and the second grade reads the same inputs as
  the first.
- Targets must be runnable on a CPU-only host without oneAPI compilers (`*_host` test targets), unless the task is
  about that.

Adding a grader: a new module in `graders/` exporting `GRADERS = {name: fn(rd, ws, task, trace)}` and, for the
self-check, `ORACLES = {name: fn(rd, ws, task)}`. The registry imports every module in the package.
## Reading results

`grade.json` holds the grader result plus process metrics from `events.jsonl` (`metrics.py`): tool calls,
bazel/make invocations and failures, whether the agent hit `icpx is not found`, which guidance files it opened,
and how many commands were refused.

For fix tasks report `strict_pass` (hidden tests and regression suites), not `pass`. `format_ok` is
clang-format 20.1.8 on the changed files and `diff_lines` is the size of the non-test diff. Review tasks report
seed recall by class and flagged decoys; a clean-diff control counts findings of blocker/major severity.
Agent variance is large (haiku review recall ranged 0.375–0.75 within one cell in the pilot), so compare cells at
8+ repeats per cell.

## Eval image

The image is `Dockerfile`, built from a `git archive` of the repository without `tasks/`. Runs are offline; four
cache directories are mounted from the host: `/cache/bazel-repo`, `/cache/bazel-disk`, `/cache/bazel-registry` (a
BCR mirror: task workspaces have no `MODULE.bazel.lock`) and `/cache/bazel-install` (`bin/bazel` passes it as
`--output_user_root`, so Bazel's install base is shared and stays off the container layer). Fill them once with
network access; the full `check` is the warm-up, since task bases pin different Bazel versions and dependency sets:

```sh
mkdir ctx && git archive HEAD | tar -x -C ctx && rm -r ctx/dev/agent_evals/tasks   # as agent-benchmark builds it
docker build -f ctx/dev/agent_evals/Dockerfile -t onedal-agent-evals ctx
docker run --rm --user "$(id -u):$(id -g)" -e USER=eval \
  -v "$PWD":/harness:ro -v <oneDAL clone or .git dir>:/src:ro -e ONEDAL_EVAL_SRC=/src \
  -v <cache>/bazel-repo:/cache/bazel-repo -v <cache>/bazel-disk:/cache/bazel-disk \
  -v <cache>/bazel-registry:/cache/bazel-registry -v <cache>/bazel-install:/cache/bazel-install \
  -v <root>:/evalroot -e ONEDAL_EVAL_ROOT=/evalroot -w /harness onedal-agent-evals \
  bash -c 'test -e /cache/bazel-registry/bazel_registry.json ||
             git clone -q --depth 1 https://github.com/bazelbuild/bazel-central-registry /cache/bazel-registry
           python3 dev/agent_evals/run.py check'
```

After that, the same command with `--network none` passes `check` for every task. `build_make_gnu` builds
against the oneMKL/oneTBB the image installs under `/opt/onedal-make-deps` (its prompt says so); outside the image
its oracle pip-installs the same pins, with network. Behind a proxy, pass `http_proxy`/`https_proxy` to the warm-up.
