#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
progress_file="$repo_root/context/progress.md"
ts="$(date -Iseconds)"

if [ -f "$progress_file" ]; then
  printf '%s\n' "- $ts — Claude worktree created for isolated repository work." >> "$progress_file"
fi

printf '%s\n' '{"systemMessage":"Worktree created for this repository. Keep changes isolated, update context/progress.md for meaningful milestones, then commit and push to the default branch when ready."}'
