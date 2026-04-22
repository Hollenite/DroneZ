#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
progress_file="$repo_root/context/progress.md"
ts="$(date -Iseconds)"

if [ -f "$progress_file" ]; then
  printf '%s\n' "- $ts — Claude worktree removed or exited." >> "$progress_file"
fi

printf '%s\n' '{"systemMessage":"Worktree exit recorded for this repository."}'
