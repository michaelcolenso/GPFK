# GPFK

## Suggested Unexpected Features

### 1. Mood-Based Code Review
Analyze the emotional tone of commit messages and PR descriptions over time to generate a "team health" dashboard. Detect frustration spikes ("fix this AGAIN", "why doesn't this work"), celebrate streaks of calm commits, and surface burnout signals before they become attrition. Goes beyond code quality into human quality.

### 2. Entropy Scoring for Files
Assign each file a "chaos score" based on the number of authors, churn rate, cyclomatic complexity, and time-since-last-test-pass. Files with high entropy get flagged as "danger zones" and auto-tagged in PRs. Over time, the entropy map becomes a living heatmap of technical debt that teams can actually feel rather than argue about in retrospect.

### 3. Counterfactual Diff Explainer
When a bug is introduced, instead of just showing what changed, reconstruct *what would have had to be true* for the bug not to exist — surfacing the missing test, the skipped review step, or the implicit assumption that was never written down. Turns post-mortems from blame assignment into structural learning.

### 4. Silent Dependency Obituaries
Track when upstream libraries you depend on go unmaintained (no commits, issues unanswered, maintainer accounts inactive) and generate a quiet, non-alarming "dependency obituary" — a plain-language note about what the library did, why it mattered, and a ranked list of successors. Surfaces rot before it becomes a CVE, without the noise of a security alert.
