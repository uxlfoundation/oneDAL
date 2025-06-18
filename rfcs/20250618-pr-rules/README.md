# oneDAL Pull Request Rules

## Introduction

Generating, reviewing and merging code in oneDAL are key pillars in providing
a high quality and performant product.  The oneDAL codebase is both large and
intricate requiring domain knowledge and precision.  In order to guarantee these
qualitiies in oneDAL for all contributors, a set of rules should be added.


## Proposal

This enumerates in detail the aspects and proceedures necessary to getting a
Pull Request (PR) merged into oneDAL.  This (as listed in the preamble below)
will assist new developers as well as standardize for all developers in
completing the necessary steps. The goal would be to add this to oneDAL's
public documenation, and refer to it in the template (as a link) to all future
Pull Requests to oneDAL.

# Text to be added:

## Preamble:

This is the rules for merging PRs into the oneDAL code base on Github. This
gives the minimum requirements for PR submitters and reviewers. The goal is
to enumerate the steps for better efficiency in oneDAL development.

## Section 1: Opening a PR, Ready for Review

1. PR Template and Task Checklist
   * All PRs must use the provided template, including the task checklist.
2. Draft PRs
   * PRs should initially be opened as Draft.
3. Marking PRs as Ready for Review
   * Before marking a PR as Ready for Review:
     * All tasks in the checklist must be completed.
     * Features and changes must be fully implemented.
     * Both private and public CI pipelines must pass, or clear justification for any
       failing tests must be provided.
4. Reverting a PR Back to Draft
   * Once a PR is converted from Draft to Ready for Review, it should only be
     reverted to Draft if:
     * Significant changes or new feature requests arise during the review.
     * The requested changes depend on future dependencies (e.g., waiting for a new
       release) or require substantial time to implement.
     * Other blockers prevent the changes from being merged into the current main branch.

## Section 2: General Rules

1. A PR cannot introduce new failures in Github checks (public CI) (i.e. stays
green) or new failures in private CI (pre-commit\*).
2. Removal of CI tests can only be allowed with the written and documented
consent\*\* of the code owners in extreme circumstances.
3. Code which causes new CI failures or problems should apply additional
fixes within a day, a PR should be reverted. 
4. Code scans often do not run with PRs, as such merged PRs should not increase
the number of issues for the Intel internal code scans\*\*\*, and may be
grounds for a reversion.
5. A merged PR which causes a code scan failure should be fixed before the next
scheduled code scan can be reverted.
6. A designated person is assigned (for a determined period of time), who
determines which PR was responsible for the failure.
7. All PRs must have a label describing the purpose.
8. Suggested labels are:
   * bug (Section 3: Bug Fix)
   * dependencies (Section 4: Dependencies)
   * docs (Section 5: Documentation)
   * enhancement (Section 6: Feature)
   * infra (Section 7: Infrastructure)
   * perf (Section 8: Performance)
   * testing (Section 9: Tests)
   * API/ABI breaking change (Section 10: API/ABI changes)
9. Associated reviewers from CODEOWNERS must be included in review based on the
   purpose/type of the PR, for example:
   * Cases which require special consideration - CI failures/issues
   * Changes to public facing header files etc.
   * Changes to build structure (e.g. makefiles, bazel. etc.)
11. Commits should follow commit message rules
12. During a code-freeze period (before drop), PRs should only be pulled in by
main maintainers.
13. An approval should be given with the expectation to merge, use best
judgement in reviewing and approving PRs (get help as when not confident).
14. The submitter is responsible for the maintenance (including closing) of
PRs.
15. The submitter is responsible for guiding, answering, scheduling, and
aligning the correct reviewers for a PR (setting up meetings, collecting
information, etc.).
16. The person who merges a PR is expected to understand the consequences,
especially with respect to approvals and review (e.g. a single approval may
not be sufficient for merge).
17. Let reviewer mark comment as resolved unless PR author directly addresses
their comment with code change
18. To avoid overuse of "will save for follow-up PR"/"not in the scope of PR"
the author is responsible resolving with reviewer the follow up steps in
detail.
19. Changes to these rules must be proposed and agreed to by unanimous vote of
participants in an architecture meeting.
20. Additional private CI runs can be requested.
21. While oneDAL is open source, hardware IP should be protected, especially in
public forums like Slack and GitHub.
22. An admin can wipe a review or otherwise change a PR as necessary. (Reviewer
unavailable to re-review, etc.)
23. Description must describe the change and reasoning behind it.
24. These are not hard and fast and can be changed in cases that warrant it.
25. When merging, use the title of the PR and not of the commit(s).

## Section 3: Bug Fix

1. PR Title must include the associated algorithm or oneDAL code feature and
the name of the bug.
2. Description must describe the issue observed and how the fix solves the
bug/issue.
3. If the bug fix is associated with an issue raised on Github, it must be
linked.
4. If the bug fix solves the Github issue and is merged, the Github issue
should be closed.
5. Fixes which solves a CI failure must be listed for verification by the
reviewer.

## Section 4: Dependencies

1. Dependency PRs not handled by automation should be handled per the
guidance of reviewer with knowledge of the change (e.g. code owner).
2. Questions about dependency changes can/should be addressed to the team at
large.

## Section 5: Documentation

1. The PR title must include the related feature and/or DAAL/oneDAL component.
2. The documentation must follow DAAL or oneDAL documentation guidelines.

## Section 6: Feature

1. Features which change (not just add to) the user-facing API or ABI or change
dependencies in a meaningful way should have an RFC (see section 10)
2. Features in oneDAL must follow oneDAL conventions (especially for API and
structure).
3. Features in DAAL must follow DAAL conventions.
4. New features must have associated new tests.

## Section 7: Infrastructure

1. As infra has wide-ranging impact, code review should be stricter and closely
reviewed by those knowledgeable.  
2. Those which impact a specific ISA should have a representative with domain
knowledge review those changes.

## Section 8: Performance

1. Performance benchmarks must include scikit-learn_bench\*\*\*\*\* if
the algorithm is included or to be included in scikit-learn-intelex.
2. Performance benchmarks must be provided to reviewers via email.
3. Code with SYCL-specific implementations should include specific benchmarking
for associated GPU hardware.
4. Relative improvements in performance should be listed in the PR description.
5. Any degradation in performance should be discussed before merging.

## Section 9: Tests

1. The PR title should include the related feature or components.
2. Tests should be added for new features (ideally in the same PR).
3. Ideally, the PR should refer to the feature introduction/or latest change
to feature PR.

## Section 10: API/ABI changes

1. Changes to the API/ABI should be specifically scheduled for merging at
specific times determined by the main maintaners of oneDAL.
2. Changes to the API/ABI must be described in detail in the description of the
pull request.
3. Information about the API/ABI change should be communicated in some fashion
to users of oneDAL.
4. Changes in dependency API/ABIs should be noted, and may be exempt from these
rules.
5. In most cases, and API/ABI breaking change should first be discussed in an
RFC process.

### Notes:

\* Comment with text '/intelci: run' on the Github PR will automatically run
the CI, labeled as 'pre-commit'.

\*\* Documented consent must mean that developers should be able to
independently find and understand CI regressions via some medium outside the PR
itself (for example confluence, email chain, etc.)

\*\*\* Some code scans can be found in the DAAL CI_DAAL Run, some may only run
after integration.

\*\*\*\* Cleanly means without regression.

\*\*\*\*\* scikit-learn_bench report structure is the standard, but other runs
can be used at reviewers discretion.

\*\*\*\*\*\* direct links to the run should be modified to refer to
http://intel-ci.intel.com/ and not internal Intel servers.

# end of text

## Open Questions
How to relate to non-Intel parties in the case some of special requirements on
impacts to this hardware.
Additions or removals to the rules as per discussion.

## Next steps:
Upon acceptance, add to the oneDAL documents a web page under `contributing` for
reference containing this text.  Secondly add to the PR template a link to this new
page for easy access.
