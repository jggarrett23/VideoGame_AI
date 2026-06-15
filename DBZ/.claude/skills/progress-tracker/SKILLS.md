---
name: progress-tracker
description: Reads the current conversation and project state, then writes a concise update to PROGRESS.md tracking what has been accomplished toward the overall goal of building RL agents that play popular video games.
---

# progress-tracker

Your job is to update `.\PROGRESS.md` based on what has been accomplished in the current conversation and what already exists in the file. You are a scribe, not an analyst — record facts, not opinions.

## Rules

- **Hard limit: 200 lines.** If the file would exceed 200 lines, summarize or merge older completed entries before adding new ones. Never exceed the limit.
- Do not duplicate entries. If something is already recorded, update it in place rather than adding a second entry.
- Do not record speculative or planned work as done. Only log what has actually been completed in this conversation.
- Keep every entry to one line where possible. Use sub-bullets only when a single line would lose critical detail.
- Do not add section headers for empty sections.

## PROGRESS.md structure

Maintain exactly these top-level sections in this order:

```
# Project Goal
# Experiments
# Infrastructure & Tooling
# Known Issues
```

### `# Project Goal`
One sentence, always present. Describes the overarching purpose of the repository. Update only if the user explicitly changes it.

### `# Experiments`
One entry per model/algorithm tested. Format:

```
## <model_name> — <status: Planned | In Progress | Complete>
- **Algorithm**: <e.g. DQN, PPO>
- **Result**: <avg reward last 20 ep, peak reward, wall time/ep, VRAM peak> — fill in after training
- **Verdict**: <Best | Candidate | Rejected> with one-line reason — fill in after comparison
```

Only create an entry when a model has been proposed and approved. Move status forward as work progresses.

### `# Infrastructure & Tooling`
Bullet list of completed setup items: repo conventions, scripts, skills, config files, refactors. Date each item `(YYYY-MM-DD)`.

### `# Known Issues`
Bullet list of bugs or limitations noted in code comments or conversation. Remove an item when it is resolved.

## How to run this skill

1. Read `.\PROGRESS.md` if it exists.
2. Read `.\CLAUDE.md` for project context if needed.
3. Identify what was accomplished in the current conversation that is not yet reflected in `PROGRESS.md`.
4. Write the updated file, respecting the 200-line limit and the structure above.
5. Report to the user what was added or changed (one sentence).
