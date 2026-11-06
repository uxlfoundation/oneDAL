## Description

<!--
Add a comprehensive description and describe the impact of proposed changes

List associated issue number(s) and documentation or benchmark PR(s), if any.
-->

## AI assistance

<!--
Which AI tools were used and for which parts of the change, or "None".
For example: "Copilot generated the kernel loop; I wrote the tests and rewrote the comments."
-->

---

<!--
Start the PR as a draft and mark it ready for review once CI passes and every box is ticked.
Each item is worded so it can be ticked truthfully on any PR, so don't delete items.
-->

**Checklist**

- [ ] I have reviewed the full diff myself and can explain the changes to each file.
- [ ] The description is concise and clearly communicates the impact of the changes.
- [ ] Code is commented as necessary and long blocks have been rewritten to be concise.
- [ ] This PR contains one logical change and unrelated fixes and large work have been split into separate PRs as needed.
- [ ] Tests, benchmarks and docs are extended where the change needs it, and no existing test was weakened or removed to make CI pass.
- [ ] All CI jobs are green or I have provided justification why they aren't.
- [ ] If a performance change is expected, I have measured the affected algorithms using [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench) and provided at least a summary table with measured data.
