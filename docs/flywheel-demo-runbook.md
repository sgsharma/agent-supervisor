# Flywheel Demo Runbook

The flywheel demo is a **1-2 week "improvement burst"**: the agent starts from a deliberately degraded baseline, the flywheel opens one PR per day, you review and merge the good ones, and the metric trend in Braintrust climbs over the cycle. This runbook covers (a) one-time setup, (b) running a cycle, and (c) resetting between cycles.

## One-time setup

After the baseline has been degraded (see `src/config.py` — vague agent descriptions and prompts), tag the commit so future cycles can reset to it.

```bash
git tag demo-baseline
git push origin demo-baseline
```

If you change the baseline later (e.g. degrade further, or restore something), re-tag:

```bash
git tag -f demo-baseline
git push --force origin demo-baseline
```

Confirm the required secrets exist:

```bash
gh secret list
# Expected: BRAINTRUST_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY, ANTHROPIC_API_KEY
```

## Running a cycle

The two workflows run themselves on schedule:

| Workflow | Cron (UTC) | Approx PT | What it does |
|---|---|---|---|
| `run_on_schedule.yml` | `0 15 * * *` | 7AM PST / 8AM PDT | Replays scripted conversations through the supervisor → produces fresh production traces in Braintrust |
| `flywheel.yml` | `0 16 */2 * *` | 8AM PST / 9AM PDT, every other day | Reads those traces, proposes optimizations, opens a PR |

Each flywheel PR triggers `ci.yml` (the Braintrust `eval-action`), which runs `eval_supervisor.py`, `eval_math_agent.py`, and `eval_research_agent.py` and posts a comment with scores. Use that comment to decide whether to merge.

To trigger either workflow on demand (e.g. smoke test, off-schedule demo):

```bash
gh workflow run run_on_schedule.yml
gh workflow run flywheel.yml
```

## Resetting between cycles

### 1. Close any open flywheel PRs

```bash
gh pr list --search "flywheel in:title" --state open --json number --jq '.[].number' \
  | xargs -I {} gh pr close {} --delete-branch --comment "Closing for demo cycle reset."
```

### 2. Reset main to the degraded baseline

⚠ This force-pushes `main`. Only do it on this demo repo.

```bash
git checkout main
git fetch origin
git reset --hard demo-baseline
git push --force-with-lease origin main
```

### 3. (Optional) Filter out the previous cycle in Braintrust

The Braintrust UI's experiment view lets you filter by date — easiest path is to filter to "since the reset" when showing the trend. No data deletion needed. The eval-action's experiments stay associated with the old commits, so they don't pollute the new cycle as long as you filter by time.

## Verifying the reset

```bash
git diff demo-baseline main       # should be empty
gh pr list --search "flywheel"    # should be empty
```

Then trigger one flywheel run manually to confirm the loop still works end-to-end:

```bash
gh workflow run flywheel.yml
```

## Mid-cycle rewinds (non-destructive)

To undo specific merged PRs without resetting the whole cycle:

```bash
git revert <merge-commit-sha>
git push
```

The flywheel may re-propose the same change on its next run — that's expected, and is itself a useful demo moment ("look, it still thinks this matters even after we rejected it once").
