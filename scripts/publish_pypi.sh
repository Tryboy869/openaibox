#!/usr/bin/env bash
# ============================================================
# publish_pypi.sh — Publish built distribution to PyPI
#
# Requires: PYPI_TOKEN environment variable
# Set it in GitHub repo: Settings → Secrets → PYPI_TOKEN
# ============================================================

set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${BLUE}[pypi]${NC} $1"; }
ok()   { echo -e "${GREEN}[ok]${NC} $1"; }
fail() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── Check token ───────────────────────────────────────────
[ -z "${PYPI_TOKEN:-}" ] && fail "PYPI_TOKEN not set. Add it to GitHub Secrets."

# ── Check dist folder ─────────────────────────────────────
[ -d "dist" ] || fail "dist/ not found. Run release.sh first."

# ── Install twine if needed ───────────────────────────────
if ! command -v twine &>/dev/null; then
    log "Installing twine..."
    pip install twine -q
fi

# ── Upload ────────────────────────────────────────────────
log "Uploading to PyPI..."
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_TOKEN" \
twine upload dist/* --non-interactive

ok "Published to PyPI ✅"
log "Package URL: https://pypi.org/project/graphruntime/"
