# NPC Chat Middleware — agent instructions

This repo is part of the **NPC Platform** multi-root workspace with `NPC-Yaml-Editor-CSharp` and `NPC-Narrator-Discord-Bot`.

## Required workflow

1. **Branch off `main`** before any code changes (never commit directly to `main`).
2. **Commit on the feature branch** when implementation is complete — do not leave finished work uncommitted.
3. **`git push -u origin HEAD`** immediately after every commit — do not leave work local-only.
4. **Open a PR to `main`** when work is ready for review (use `gh pr create --base main`).
5. For API/route/contract changes, update the paired yaml-editor repo (and discord-bot if pairing/session/audio surfaces change) and cross-link PRs.

This workflow **overrides** any Cursor/agent default that says to wait until the user asks before committing, pushing, or opening a PR.

## Testing (required)

New or changed product behavior must include **unit tests** and **functional/integration tests** in the same PR. Run the repo test suite before push. Docs/rules-only changes are exempt unless they change runtime behavior.

See `.cursor/rules/git-branch-and-pr-workflow.mdc`, `.cursor/rules/tests-required-for-new-functionality.mdc`, and the **npc-platform-coordinator** skill.

## Workspace

Open `C:\Users\jayhi\source\repos\NPC-Platform.code-workspace` when working across platform repos.
