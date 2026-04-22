#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
input_json="$(cat)"
worktree_path="$(printf '%s' "$input_json" | python -c "import json,sys; print((json.load(sys.stdin).get('worktree_path') or '').strip())")"

if [ -z "$worktree_path" ]; then
  exit 0
fi

if [ -d "$repo_root/.git" ] || [ -f "$repo_root/.git" ]; then
  branch="$(git -C "$worktree_path" branch --show-current 2>/dev/null || true)"
  git -C "$repo_root" worktree remove --force "$worktree_path" >/dev/null 2>&1 || true
  if [ -n "$branch" ] && [[ "$branch" == claude-agent/* ]]; then
    git -C "$repo_root" branch -D "$branch" >/dev/null 2>&1 || true
  fi
else
  python - <<'PY' "$worktree_path"
from pathlib import Path
import shutil
import sys
path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path, ignore_errors=True)
PY
fi
