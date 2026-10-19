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
the eval image and the oneDAL rules of the guidance claim check. Everything else (workspace preparation, the task
self-check, running agents, isolation, traces, reports) is done by [dokimi](https://github.com/intel/dokimi), which
reads `repo-eval.yaml` (the repository-evaluation contract). No LLM judges a result: graders rebuild and rerun tests.

The generic parts still here are moving to dokimi too ([intel/dokimi#41](https://github.com/intel/dokimi/issues/41)):
the review and hidden-test graders, `contract.py` and `metrics.py` are replaced by dokimi built-ins
step by step, until this directory holds only the tasks, the oneDAL build graders and the eval image.

Terms used below:

- **task**: one job for an agent on a copy of oneDAL at a fixed commit (build X, fix bug Y, review change Z).
- **arm**: one variant of the repository the same tasks run against, e.g. with the agent guidance files (`raw`)
  and without them (`none`).
- **grader**: the code that decides whether a run succeeded, by rebuilding and rerunning oneDAL tests.
- **oracle**: the reference solution of a task, used to prove the grader accepts a correct answer.
- **hidden test**: a test the agent never sees; the grader adds it only while grading.
- **strict pass**: the hidden tests pass and the component's existing tests still pass.
- **contract**: the file layout and commands (`repo-eval.yaml`, `run.py grade|oracle --contract`) through which
  dokimi uses this directory.

| tier | what | cost |
|---|---|---|
| T0 static | `dokimi repo scan`: every link, path, Bazel label, bazel/make/cmake command, Starlark macro and identifier named in the guidance is checked against the tree | $0, seconds |
| self-check | `dokimi repo tasks check`: for every task the untouched workspace fails its grader and the reference solution passes | $0, CPU, docker |
| T1 tasks | `dokimi repo run`: an agent solves each task under each arm; these graders score it | LLM cost + CPU |

## Quick start

Install dokimi at the commit CI pins (`DOKIMI_SHA` in `.github/workflows/agent-evals.yml`), then, from a
full-history clone of oneDAL (mined tasks read their fix commits from it):

```sh
python3 -m venv ~/dokimi && ~/dokimi/bin/pip install \
  -c https://raw.githubusercontent.com/intel/dokimi/$DOKIMI_SHA/dokimi/requirements.lock.txt \
  "dokimi @ git+https://github.com/intel/dokimi@$DOKIMI_SHA#subdirectory=dokimi"
export PATH=~/dokimi/bin:$PATH
M=dev/agent_evals/repo-eval.yaml
dokimi repo validate .@HEAD --manifest $M      # task.json shape, bases resolve, patches exist; no build
dokimi repo scan .@HEAD --manifest $M                                       # T0
dokimi repo tasks check .@HEAD --manifest $M --task fix_finiteness_mut --warm  # self-check in the eval image
```

`repo tasks check` needs docker; `--task` is repeatable and without it every task is checked. Its run directories
are under `--out` (default `repo-check/check/<task>__null` and `__oracle`, with `grade.json` and the grader logs).
See "Eval image" below for the caches.

## Continuous integration

`.github/workflows/agent-evals.yml` runs the checks that need no model and no token:

- on every pull request that touches this directory or the agent guidance (`AGENTS.md`, `.github/instructions`,
  `.github/copilot-instructions.md`, `CONTRIBUTING.md`): `dokimi repo validate` (a gate) and `dokimi repo scan`
  (report only, as a job summary and an artifact);
- `dokimi repo tasks check` in the eval image, one job per task: on a pull request for the tasks it changes, plus one
  task per grader when the entry points or a grader change; weekly, and on demand (`workflow_dispatch`), for every
  task. `fix_cel_thread_race` is left out until dokimi can mark a task unsupported on hosts that cannot show its bug
  (intel/dokimi#41, step D3).

dokimi is pinned by commit in the workflow. Agent runs stay manual: they cost tokens and depend on the agent being
measured. Rerun them after a large change to guidance, tooling, build system or layout. A change that breaks
`dokimi repo tasks check` for a task updates that task in the same pull request.

## Running agents

With dokimi (see its `docs/repo-evaluation-contract.md`, "Running"):

```sh
dokimi repo run <oneDAL>@<sha> --manifest dev/agent_evals/repo-eval.yaml \
    --arm guidance:none,guidance:raw --model <model id> --reps 3 --batch b1
dokimi repo report repo-runs/b1
dokimi repo compare repo-runs/b1 --baseline guidance:none
```

The framework prepares each workspace (history-free `git archive` of the task base without this directory, setup
patches, the arm, the review patch), runs the agent in its own container, writes `events.jsonl`, `answer.txt` and
`meta.json`, and then calls the entry points below in a separate container:

| entry point | does |
|---|---|
| `run.py grade --contract <run_dir> --source <clone>` | grades `run_dir/workspace` with the task's grader, writes `grade.json` (`repo_grade.v1`) |
| `run.py oracle --contract <run_dir> --source <clone>` | applies the reference solution (and answer) for the self-check |

Arms are `guidance:none` (every file matched by `guidance.globs` deleted) and `guidance:raw` (as checked in), both
made by the framework from `repo-eval.yaml`. Fix and feature tasks declare `"gate": ["strict_pass"]`; score-only
review tasks declare `"score": {"metric": "recall", "max": 1.0}`.

## Tasks

`tasks/<id>/task.json`, plus any patches, manifests and hidden files it names. What each task asks, where it comes
from and why it is built that way: [`tasks/README.md`](tasks/README.md).

| key | meaning |
|---|---|
| `base` | commit the workspace is taken from (`<sha>^` for mined fixes) |
| `family` | `build`, `fix`, `test`, `review` |
| `prompt` | what the agent is told; a final ```` ```json ```` block is requested when the grader needs a structured answer |
| `grader` | name registered by a module in `graders/` |
| `setup_patch` / `setup_patches` | applied to the workspace before the arm (injected bugs) |
| `review_patch` | `git am`-ed on top as the HEAD commit under review |
| `gate` | metrics that count as success for dokimi (default `["pass"]`); `["strict_pass"]` on every `test_pass`/`test_pass_files` task, since they all have regression suites |
| grader keys | see the docstring of the grader: `hidden_from`/`hidden_files`/`targets`/`regression_targets` for `test_pass`, `manifest.json` for `review`, ... |

Rules every task follows:

- `dokimi repo tasks check --task <id>` reports it `ok`: the untouched workspace fails the grader, the reference
  solution passes it, and the agent's container sees neither this directory nor the full-history clone.
- Fixes and answers cannot leak. dokimi makes the workspace from `git archive` of `base` with no history, and
  `dev/agent_evals` is deleted from it.
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

The image is `Dockerfile`; dokimi builds it from the repository at the eval revision without `tasks/` (tag
`dokimi/onedal:<sha>`). Runs are offline (`image.network: none`); the four caches of `repo-eval.yaml` are host
directories mounted at `/cache/bazel-repo`, `/cache/bazel-disk`, `/cache/bazel-registry` (a BCR mirror: task
workspaces have no `MODULE.bazel.lock`) and `/cache/bazel-install` (`bin/bazel` passes it as `--output_user_root`,
so Bazel's install base is shared and stays off the container layer). dokimi puts them under `--cache-root`
(default `~/.cache/dokimi/caches/onedal/<name>`). Clone the BCR mirror once, then fill the other caches with one
`--warm` pass (null and oracle with network), since task bases pin different Bazel versions and dependency sets:

```sh
C=~/.cache/dokimi/caches/onedal
git clone -q --depth 1 https://github.com/bazelbuild/bazel-central-registry $C/bazel-registry
dokimi repo tasks check .@HEAD --manifest dev/agent_evals/repo-eval.yaml --warm -j 4
dokimi repo tasks check .@HEAD --manifest dev/agent_evals/repo-eval.yaml -j 4     # offline, as agents run
```

After the warm-up, the offline check passes for every task. `build_make_gnu` builds
against the oneMKL/oneTBB the image installs under `/opt/onedal-make-deps` (its prompt says so); outside the image
its oracle pip-installs the same pins, with network. Behind a proxy, pass `http_proxy`/`https_proxy` to the warm-up.
