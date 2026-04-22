#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
input_json="$(cat)"
name="$(printf '%s' "$input_json" | python -c "import json,re,sys; raw=(json.load(sys.stdin).get('name') or 'worktree'); safe=re.sub(r'[^A-Za-z0-9._-]+','-',raw).strip('-'); print(safe or 'worktree')")"
worktree_root="$repo_root/.openclaude/agent-worktrees"
worktree_path="$worktree_root/$name"

mkdir -p "$worktree_root"

if [ -e "$worktree_path" ]; then
  printf '%s' "$worktree_path"
  exit 0
fi

if [ -d "$repo_root/.git" ] || [ -f "$repo_root/.git" ]; then
  branch="claude-agent/$name"
  if git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$repo_root" worktree add "$worktree_path" "$branch" >/dev/null
  else
    git -C "$repo_root" worktree add -b "$branch" "$worktree_path" HEAD >/dev/null
  fi
else
  python - <<'PY' "$repo_root" "$worktree_path"
from pathlib import Path
import shutil
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ignore = shutil.ignore_patterns('.git', '.openclaude/agent-worktrees', '__pycache__', '.pytest_cache', '.venv')
shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=False)
PY
fi

printf '%s' "$worktree_path"
