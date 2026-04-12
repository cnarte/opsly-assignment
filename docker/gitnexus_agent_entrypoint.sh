#!/bin/bash
# Re-register any previously analyzed repos in the workspace
set -e

WORKSPACE="${WORKSPACE_PATH:-/workspace/repos}"

if [ -d "$WORKSPACE" ]; then
    for repo_dir in "$WORKSPACE"/*/; do
        if [ -d "${repo_dir}.gitnexus" ]; then
            echo "Registering existing repo: $repo_dir"
            gitnexus index "$repo_dir" --force 2>&1 || true
        fi
    done
fi

exec python -m src.gitnexus_agent.server
