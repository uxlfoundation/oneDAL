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

A harness for measuring how coding agents do on real oneDAL work: building, fixing bugs, writing tests and
reviewing changes. It compares *arms* (variants of the repository's agent guidance) on the same tasks, so that a
change to `AGENTS.md`, `.github/instructions`, tooling or repo layout can be judged by the agent behaviour it
produces rather than by reading it.

The harness drives [Claude Code](https://docs.claude.com/en/docs/claude-code) headless (`claude -p`). Every run
gets a fresh, history-free copy of the tree, its own Bazel output base and its own Claude config directory, and is
graded by rebuilding and rerunning tests. No LLM judges the result.

## Tiers

| tier | what | cost |
|---|---|---|
| T0 static | `run.py static`: every link, path, Bazel label, bazel/make command, Starlark macro and identifier named in the guidance is checked against the tree | $0, seconds |
| T1 tasks | `run.py matrix`: an agent solves each task under each arm; graders rebuild and rerun tests | LLM cost + CPU |

## Quick start

```sh
# Python 3.9+, git, bazelisk, uv (for the pinned clang-format), Claude Code CLI with credentials
export ONEDAL_EVAL_ROOT=/scratch/onedal-evals   # no CLAUDE.md / AGENTS.md / .claude in any parent directory
export ONEDAL_EVAL_SRC=$PWD                     # full-history clone; mined tasks read fix commits from it
python3 dev/agent_evals/run.py list
python3 dev/agent_evals/run.py static           # T0
python3 dev/agent_evals/run.py check            # grader self-check for every task, no LLM
python3 dev/agent_evals/run.py matrix --arms none,main-raw,main-claude --models haiku --reps 3 -j 6 --batch b1
python3 dev/agent_evals/run.py summary --batch b1
```

Other settings (Bazel binary, shared caches, timeout) are environment variables documented in `config.py`.
Model aliases (`haiku`, `sonnet`, `opus`) resolve to Bedrock ids when `CLAUDE_CODE_USE_BEDROCK` is set and to
Anthropic API ids otherwise; a full model id also works.

## Arms

| arm | repository state |
|---|---|
| `none` | every `AGENTS.md` and `.github/instructions/*.md` deleted |
| `main-raw` | as checked in. Claude Code does not auto-load `AGENTS.md`, so the agent only sees it if it opens it |
| `main-claude` | a `CLAUDE.md` containing `@AGENTS.md` next to every `AGENTS.md`, so Claude loads the hierarchy |

An arm is one function in `common.py` that edits the task repo in place (another guidance revision, a placebo of
the same length, extra tooling).

## Tasks

`tasks/<id>/task.json`, plus any patches, manifests and hidden files it names.

| key | meaning |
|---|---|
| `base` | commit the snapshot is taken from (`<sha>^` for mined fixes) |
| `family` | `build`, `fix`, `test`, `review` |
| `prompt` | what the agent is told; a final ```` ```json ```` block is requested when the grader needs a structured answer |
| `grader` | name registered by a module in `graders/` |
| `setup_patch` / `setup_patches` | applied before the arm (injected bugs) |
| `review_patch` | `git am`-ed on top as the HEAD commit under review |
| grader keys | see the docstring of the grader: `hidden_from`/`hidden_files`/`targets`/`regression_targets` for `test_pass`, `manifest.json` for `review`, ... |

Rules every task follows:

- `run.py check <task>` passes: the untouched task repo fails the grader and the reference solution passes it.
- Fixes and answers cannot leak. The snapshot is `git archive` of `base` with no history, and `dev/agent_evals`
  is deleted from it.
- Grading is deterministic: tests are rerun with `--nocache_test_results`, and regrading a run gives the same result.
- Targets must be runnable on a CPU-only host without oneAPI compilers (`*_host` test targets), unless the task is
  about that.

Adding a grader: a new module in `graders/` exporting `GRADERS = {name: fn(rd, task, trace)}` and, for the
self-check, `ORACLES = {name: fn(rd, repo, task)}`. The registry imports every module in the package.

## Run isolation

- Runs live under `$ONEDAL_EVAL_ROOT/runs/<batch>/<task>__<arm>__<model>__r<rep>/` with `repo/`, `trace.jsonl`,
  `meta.json`, `grade.json` and the grader logs. `run.py matrix` skips runs that already have `grade.json`.
- `CLAUDE_CONFIG_DIR` is per run, so nothing from the user's `~/.claude` (memory, settings, plugins) is loaded.
  `ONEDAL_EVAL_ROOT` is refused if a parent directory holds `CLAUDE.md`/`AGENTS.md`/`.claude`, since Claude Code
  loads those into every arm.
- Background commands are disabled (`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` and a `PreToolUse` hook,
  `bin/no_background.py`). `claude -p` ends when the model ends its turn, so a backgrounded build would be killed
  and graded as a task failure.
- `bin/bazel` is first on the agent's `PATH`: it pins the run's `--output_base` and adds the shared repository and
  disk caches. `MKLROOT` is removed from the environment.

## Reading results

`grade.json` holds the grader result plus process metrics from the trace: cost, turns, tool calls, bazel/make
invocations and failures, whether the agent hit `icpx is not found`, which guidance files it opened, and how many
background commands the hook refused.

For fix tasks report `strict_pass` (hidden tests and regression suites), not `pass`. `format_ok` is
clang-format 20.1.8 on the changed files and `diff_lines` is the size of the non-test diff. Review tasks report
seed recall by class and flagged decoys; a clean-diff control counts findings of blocker/major severity.
Agent variance is large (haiku review recall ranged 0.375–0.75 within one cell in the pilot), so compare cells at
8+ repeats per cell.
