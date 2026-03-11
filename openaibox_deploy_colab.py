#!/usr/bin/env python3
"""
openaibox_deploy_colab.py — Colab Deployment Script
====================================================
1. Upload ZIP → 2. Build → 3. Test on SmolLM → 4. GitHub → 5. PyPI
"""

import os, sys, json, subprocess, urllib.request, base64

# ════════════════════════════════════════════════════════════
# TOKENS — paste yours here
# ════════════════════════════════════════════════════════════
GITHUB_TOKEN = "PASTE_YOUR_GITHUB_TOKEN_HERE"
PYPI_TOKEN   = "PASTE_YOUR_PYPI_TOKEN_HERE"
GITHUB_USER  = "tryboy869"
REPO_NAME    = "openaibox"
# ════════════════════════════════════════════════════════════

def run(cmd, env=None, check=True):
    full_env = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=full_env)
    if r.stdout.strip(): print(r.stdout)
    if r.returncode != 0:
        if r.stderr.strip(): print(r.stderr[-2000:])
        if check: raise SystemExit(f"Command failed: {cmd}")
    return r

def section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")
def ok(m):  print(f"  ✅ {m}")
def log(m): print(f"  ›  {m}")

# ── Section 1 : Dependencies ──────────────────────────────
section("SECTION 1 — Installing dependencies")
# Upgrade numpy first to avoid scipy/sklearn version conflicts
run("pip install -q --upgrade numpy")
run("pip install -q transformers torch build twine setuptools wheel")
ok("Dependencies installed")

# ── Section 2 : Upload ZIP ────────────────────────────────
section("SECTION 2 — Upload repo ZIP")
try:
    from google.colab import files as colab_files
    log("A file picker will appear below.")
    log("Upload the openaibox-v1.0.0-beta.zip file.")
    uploaded = colab_files.upload()
    if not uploaded: raise SystemExit("No file uploaded.")
    zip_filename = list(uploaded.keys())[0]
except ImportError:
    zips = [f for f in os.listdir(".") if f.endswith(".zip") and "openaibox" in f]
    if not zips: raise SystemExit("No openaibox zip found.")
    zip_filename = zips[0]
ok(f"Uploaded: {zip_filename}")

# ── Section 3 : Extract + Build ───────────────────────────
section("SECTION 3 — Extracting and building")
WORK_DIR = "/content/openaibox-workspace"
os.makedirs(WORK_DIR, exist_ok=True)
run(f"unzip -o '{zip_filename}' -d {WORK_DIR}")
ok("Extracted")

entries = sorted([e for e in os.listdir(WORK_DIR)
                  if os.path.isdir(os.path.join(WORK_DIR, e))
                  and not e.startswith(".")])
log(f"Found dirs in workspace: {entries}")
REPO_DIR = os.path.join(WORK_DIR, entries[0])
log(f"Repo root: {REPO_DIR}")
os.chdir(REPO_DIR)

run("python -m build")
ok("Package built")
dist_files = os.listdir("dist")
log(f"dist/ contents: {dist_files}")

# ── Section 4 : Install + Test ────────────────────────────
section("SECTION 4 — Testing on SmolLM-360M")

whl = next(f for f in dist_files if f.endswith(".whl"))
run(f"pip install dist/{whl} -q --force-reinstall")
ok(f"openaibox installed from {whl}")

# Run test inline so we see ALL output
log("Running validation tests in subprocess (isolated process)...")

# Write a self-contained test runner that avoids numpy kernel conflicts
test_runner = """
import sys, os, json
sys.path.insert(0, os.getcwd())

from openaibox import OpenAIBox

RESULTS_DIR = "tests/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, "smollm_graph.json")

print("  › Loading SmolLM-360M...")
oaib = OpenAIBox("HuggingFaceTB/SmolLM-360M")

print("  › Running discover()...")
oaib.discover()
g = oaib._graph_result
print(f"  ✅ Architecture : {g.architecture}")
print(f"  ✅ Hidden dim   : {g.hidden_dim}")
print(f"  ✅ Layers       : {g.num_layers}")
print(f"  ✅ Params       : {g.total_params:,}")

assert g.architecture == "LlamaForCausalLM", f"Wrong arch: {g.architecture}"
assert g.hidden_dim == 960, f"Wrong hidden_dim: {g.hidden_dim}"
assert any(p.role == "decision" for p in g.injection_points), "No decision point"
print("  ✅ discover() assertions passed")

print("  › Running map_dimensions()...")
oaib.map_dimensions()
m = oaib._mapping_result
print(f"  ✅ Groups analyzed  : {len(m.groups)}")
print(f"  ✅ Multi-role dims  : {len(m.multi_role_dims)}")
print(f"  ✅ Specialist roles : {list(m.specialist_dims.keys())}")

print("  › Exporting graph.json...")
doc = oaib.export(OUTPUT_PATH)
print(f"  ✅ graph.json written ({os.path.getsize(OUTPUT_PATH):,} bytes)")

oaib.print_summary()
print("  ✅ ALL TESTS PASSED")
"""

runner_path = "/tmp/oaib_test_runner.py"
with open(runner_path, "w") as f:
    f.write(test_runner)

result = subprocess.run(
    [sys.executable, runner_path],
    capture_output=True, text=True,
    cwd=REPO_DIR
)
print(result.stdout)
if result.returncode != 0:
    print("  ❌ STDERR:")
    print(result.stderr[-3000:])
    raise SystemExit(1)

ok("ALL TESTS PASSED ✅")
OUTPUT_PATH = os.path.join(REPO_DIR, "tests", "results", "smollm_graph.json")

# ── Section 5 : Git + GitHub ──────────────────────────────
section("SECTION 5 — Git setup + GitHub repo creation")

run('git config --global user.email "anzize.contact@proton.me"')
run('git config --global user.name "Daouda Abdoul Anzize"')
if not os.path.exists(".git"):
    run("git init")
run("git add .")
run('git commit -m "feat: initial release — Open AI Box v1.0.0-beta" --allow-empty')
ok("Git commit ready")

# Create GitHub repo
api_data = json.dumps({
    "name": REPO_NAME,
    "description": "Universal LLM introspection. Open AI Box — understand any model.",
    "homepage": f"https://pypi.org/project/{REPO_NAME}/",
    "has_issues": True, "private": False, "auto_init": False,
}).encode()
req = urllib.request.Request(
    "https://api.github.com/user/repos", data=api_data,
    headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
             "Content-Type": "application/json",
             "Accept": "application/vnd.github+json"},
    method="POST")
try:
    with urllib.request.urlopen(req) as resp:
        repo_info = json.loads(resp.read())
    REPO_URL = repo_info["clone_url"]
    ok(f"Repo created: {repo_info['html_url']}")
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    if "already exists" in str(body.get("errors", "")):
        REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"
        log("Repo already exists — using existing")
    else:
        raise SystemExit(f"GitHub API error: {body}")

AUTH_URL = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")
run(f"git remote remove origin 2>/dev/null || true", check=False)
run(f"git remote add origin '{AUTH_URL}'")
run("git branch -M main")
run("git push -u origin main --force")
ok("Code pushed to GitHub")

# ── Section 6 : GitHub Release ────────────────────────────
section("SECTION 6 — GitHub Release")
TAG = "v1.0.0-beta"
run(f"git tag -a {TAG} -m 'Release {TAG}' 2>/dev/null || true", check=False)
run(f"git push origin {TAG}", check=False)

with open("CHANGELOG.md") as f:
    changelog = f.read()
notes = changelog.split("## [")[1].split("\n", 1)[1].split("## [")[0].strip() if "## [" in changelog else "Initial release"

rel_data = json.dumps({"tag_name": TAG, "name": f"Open AI Box {TAG}",
                        "body": notes, "draft": False, "prerelease": True}).encode()
rel_req = urllib.request.Request(
    f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases",
    data=rel_data,
    headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
             "Content-Type": "application/json",
             "Accept": "application/vnd.github+json"},
    method="POST")
try:
    with urllib.request.urlopen(rel_req) as resp:
        rel_info = json.loads(resp.read())
    RELEASE_URL = rel_info["html_url"]
    ok(f"Release: {RELEASE_URL}")
except urllib.error.HTTPError as e:
    RELEASE_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}/releases"
    log(f"Release may exist already: {e}")

# ── Section 7 : PyPI ──────────────────────────────────────
section("SECTION 7 — Publishing to PyPI")
run("pip install twine -q")
run("twine upload dist/* --non-interactive --skip-existing",
    env={"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": PYPI_TOKEN})
ok("Published to PyPI")

# ── Final Summary ─────────────────────────────────────────
GITHUB_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"
PYPI_URL   = f"https://pypi.org/project/{REPO_NAME}/"

print(f"\n{'█'*60}")
print(f"  🚀 DEPLOYMENT COMPLETE — Open AI Box v1.0.0-beta")
print(f"{'█'*60}")
print(f"  📦 PyPI    → {PYPI_URL}")
print(f"  🐙 GitHub  → {GITHUB_URL}")
print(f"  🏷  Release → {RELEASE_URL}")
print(f"\n  pip install openaibox")
print(f"{'█'*60}\n")
