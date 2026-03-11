#!/usr/bin/env bash
# ============================================================
# release.sh — Auto-detect version bump in CHANGELOG + publish
#
# Usage: ./scripts/release.sh
#
# What it does:
#   1. Reads the latest version from CHANGELOG.md
#   2. Updates pyproject.toml version to match
#   3. Builds the package
#   4. Creates a GitHub release + tag
#   5. Triggers PyPI publish (via publish_pypi.sh)
# ============================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BLUE}[release]${NC} $1"; }
ok()   { echo -e "${GREEN}[ok]${NC} $1"; }
fail() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── Read version from CHANGELOG ───────────────────────────
log "Reading version from CHANGELOG.md..."
VERSION=$(grep -m1 '## \[' CHANGELOG.md | sed 's/## \[//;s/\].*//')
[ -z "$VERSION" ] && fail "No version found in CHANGELOG.md"
ok "Version detected: $VERSION"

# Convert to PEP 440 (1.0.0-beta → 1.0.0b1)
PEP_VERSION=$(echo "$VERSION" | sed 's/-beta/b1/;s/-alpha/a1/;s/-rc\([0-9]*\)/rc\1/')
log "PEP 440 version: $PEP_VERSION"

# ── Update pyproject.toml ─────────────────────────────────
log "Updating pyproject.toml..."
sed -i "s/^version.*=.*/version         = \"$PEP_VERSION\"/" pyproject.toml
ok "pyproject.toml updated"

# ── Check git is clean (warn, don't fail) ─────────────────
if ! git diff --quiet; then
    log "Uncommitted changes detected — committing version bump..."
    git add pyproject.toml
    git commit -m "chore: bump version to $PEP_VERSION"
fi

# ── Build ─────────────────────────────────────────────────
log "Building distribution..."
rm -rf dist/ build/ *.egg-info
python -m build
ok "Build complete"

# ── Create git tag ────────────────────────────────────────
TAG="v$VERSION"
if git rev-parse "$TAG" >/dev/null 2>&1; then
    log "Tag $TAG already exists — skipping tag creation"
else
    git tag -a "$TAG" -m "Release $TAG"
    git push origin "$TAG"
    ok "Tag $TAG pushed"
fi

# ── Create GitHub Release ─────────────────────────────────
log "Creating GitHub release..."

# Extract release notes for this version from CHANGELOG
NOTES=$(awk "/^## \[$VERSION\]/,/^## \[/" CHANGELOG.md | head -n -1 | tail -n +2)

gh release create "$TAG" \
    dist/*.whl dist/*.tar.gz \
    --title "GraphRuntime $TAG" \
    --notes "$NOTES" \
    --latest

ok "GitHub release $TAG created"

# ── Publish to PyPI ───────────────────────────────────────
bash scripts/publish_pypi.sh

ok "Release $TAG complete 🚀"
