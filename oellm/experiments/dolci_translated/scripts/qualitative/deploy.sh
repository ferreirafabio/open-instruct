#!/usr/bin/env bash
# Deploy gate for olmo3-multilingual-dolci-sft-progression site.
#
# Runs the Playwright suite against the local site/ bundle. ONLY pushes to
# github.io and HF Space if every test passes. Any failure aborts before any
# external mutation.
#
# Usage:
#   bash oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh         # test + deploy both targets
#   bash oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh test    # tests only, no deploy
#   bash oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh github  # tests + github.io only
#   bash oellm/experiments/dolci_translated/scripts/qualitative/deploy.sh hf      # tests + HF only

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
SITE_DIR="$PROJECT_ROOT/oellm/experiments/dolci_translated/site"
HF_STAGING="$PROJECT_ROOT/oellm/experiments/dolci_translated/.hf-staging"
TESTS_DIR="$PROJECT_ROOT/oellm/experiments/dolci_translated/site_tests"
GH_PAGES="/work/dlclarge2/ferreira-oellm/ferreirafabio.github.io/olmo3-multilingual-dolci-sft-progression"

MODE="${1:-all}"

cyan() { printf "\033[36m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
red() { printf "\033[31m%s\033[0m\n" "$1"; }

# 1. Run tests against local site
cyan "==> Running Playwright tests against $SITE_DIR"
pushd "$TESTS_DIR" > /dev/null
if ! CI=1 npx playwright test; then
    red "✗ Tests failed — aborting deploy."
    popd > /dev/null
    exit 1
fi
popd > /dev/null
green "✓ All tests passed"

if [ "$MODE" = "test" ]; then
    green "Test-only mode; not deploying."
    exit 0
fi

# 2. Sync site/ → HF staging dir (drops the github.io README, adds HF README)
if [ "$MODE" = "all" ] || [ "$MODE" = "hf" ]; then
    cyan "==> Syncing HF staging dir"
    mkdir -p "$HF_STAGING"
    cp "$SITE_DIR"/{index.html,app.js,style.css,completions.json,prompts_lmarena.json} "$HF_STAGING/"
    if [ ! -f "$HF_STAGING/README.md" ]; then
        red "✗ Missing $HF_STAGING/README.md (must contain HF Space frontmatter)"
        exit 1
    fi

    cyan "==> Uploading to HF Space (ferreirafabio/olmo3-multilingual-dolci-sft-progression)"
    "$PROJECT_ROOT/.venv/bin/python" - <<PY
from huggingface_hub import HfApi
api = HfApi()
url = api.upload_folder(
    folder_path="$HF_STAGING",
    repo_id="ferreirafabio/olmo3-multilingual-dolci-sft-progression",
    repo_type="space",
    commit_message="Sync from local site/ via deploy.sh",
)
print(f"HF Space updated: {url}")
PY
    green "✓ HF push complete"
fi

# 3. Push to ferreirafabio.github.io/olmo3-multilingual-dolci-sft-progression
if [ "$MODE" = "all" ] || [ "$MODE" = "github" ]; then
    cyan "==> Syncing github.io clone"
    if [ ! -d "$GH_PAGES" ]; then
        red "✗ github.io clone missing at $GH_PAGES — see site/README.md for clone instructions"
        exit 1
    fi
    cp -r "$SITE_DIR"/* "$GH_PAGES/"

    pushd "$GH_PAGES/.." > /dev/null
    if ! git diff --quiet HEAD -- olmo3-multilingual-dolci-sft-progression; then
        cyan "==> Committing and pushing"
        git add olmo3-multilingual-dolci-sft-progression
        git commit -m "$(cat <<EOF
olmo3-multilingual-dolci-sft-progression: deploy ($(date -u +%Y-%m-%d\ %H:%M\ UTC))

Tests: 15/15 passed (Playwright)
EOF
)"
        git push
        green "✓ github.io push complete"
    else
        green "✓ github.io already up to date"
    fi
    popd > /dev/null
fi

green ""
green "All deployments green."
green "  github.io: https://ferreirafabio.github.io/olmo3-multilingual-dolci-sft-progression/"
green "  HF Space:  https://huggingface.co/spaces/ferreirafabio/olmo3-multilingual-dolci-sft-progression"
