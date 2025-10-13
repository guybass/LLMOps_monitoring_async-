# Pre-Publication Checklist for PyPI

Complete this checklist before publishing llamonitor-async to PyPI.

## ✅ Required Steps

### 1. Git Configuration

Before committing, ensure git is configured:

```bash
# Set your identity (if not already set)
git config --global user.name "Guy Bass"
git config --global user.email "your-email@example.com"

# Verify configuration
git config --global user.name
git config --global user.email
```

### 2. Commit All Changes

You have two important commits to make:

#### Commit 1: Package Preparation (if not done)
```bash
# Check status
git status

# Unstage artifact file if present
git restore --staged compass_artifact_wf-72e7c58a-746c-4853-84b9-1312f7293f93_text_markdown.md

# Commit package prep changes
git commit -m "Prepare llamonitor-async for PyPI publication

Major updates to prepare package for public release:

Package Updates:
- Rename package from \"llmops-monitoring\" to \"llamonitor-async\"
- Update pyproject.toml with new package metadata
- Update README.md with new installation instructions and badges
- Add MANIFEST.in for proper package distribution

Documentation:
- Add PUBLISH.md: Comprehensive PyPI publishing guide
- Add UPLOAD_GUIDE.md: Quick upload reference
- Add TEST_GUIDE.md: Testing documentation and instructions

Testing Infrastructure:
- Add test_basic_monitoring.py: Smoke tests without API requirements
- Add test_agent_graph_real.py: Real OpenAI API integration test
- Add analyze_results.py: Results analysis and visualization tool

Infrastructure:
- Update .gitignore to production-grade standards
  - Add security patterns (secrets, credentials, API keys)
  - Add cross-platform IDE support
  - Add type checker caches (ruff, pyright, pytype)
  - Add comprehensive OS-specific patterns
  - Organize into clear sections

Ready for: pip install llamonitor-async

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Commit 2: Download Tracking System
```bash
# Stage all new files
git add .github/workflows/collect_download_stats.yml
git add scripts/fetch_download_stats.py
git add scripts/README.md
git add stats/README.md
git add DOWNLOAD_TRACKING.md
git add .gitignore
git add README.md
git add PRE_PUBLISH_CHECKLIST.md

# Create commit
git commit -m "Add comprehensive download tracking system

Implement a complete download tracking infrastructure to monitor
llamonitor-async adoption and usage across the Python ecosystem.

Features:
- Real-time download badges in README (total, monthly, weekly)
- Automated daily statistics collection via GitHub Actions
- Manual collection script with pypistats API integration
- Comprehensive analytics and visualization capabilities

Components Added:

Download Collection:
- scripts/fetch_download_stats.py: CLI tool for fetching PyPI stats
  - Supports 1-180 day ranges
  - Outputs JSON and CSV formats
  - Includes summary statistics
  - Tracks Python versions and OS distribution

Automation:
- .github/workflows/collect_download_stats.yml: Daily GitHub Action
  - Runs at 00:00 UTC
  - Commits stats to repository
  - Manual trigger support
  - Workflow summary generation

Documentation:
- DOWNLOAD_TRACKING.md: Complete tracking system guide
  - API access instructions
  - Data analysis examples
  - Visualization tutorials
  - Best practices and troubleshooting
- scripts/README.md: Script usage documentation
- stats/README.md: Stats directory guide

Infrastructure:
- Updated .gitignore for stats file handling
  - Ignores data files (*.json, *.csv in stats/)
  - Preserves directory structure
  - Keeps README files

Tracked Metrics:
- Total and time-series download counts
- Downloads by Python version (3.8, 3.9, 3.10, 3.11, 3.12+)
- Downloads by operating system (Linux, Windows, macOS)
- Daily, weekly, and monthly aggregations
- Growth rates and trend analysis

Use Cases:
- Monitor package adoption after releases
- Make data-driven Python version support decisions
- Track platform distribution
- Generate reports for stakeholders
- Measure community growth

Dependencies:
- pypistats: PyPI statistics API client
- pandas: Data analysis (optional, for CSV)
- requests: HTTP client

All statistics are aggregate-only with no PII collection.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 3. Push to GitHub

```bash
# Push commits to remote
git push origin main
```

### 4. Build Distribution Packages

```bash
# Install/upgrade build tools
python -m pip install --upgrade pip build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build
```

**Verify output:**
- `dist/llamonitor_async-0.1.0.tar.gz` (source distribution)
- `dist/llamonitor_async-0.1.0-py3-none-any.whl` (wheel)

### 5. Verify Package Contents

```bash
# Check package contents
tar -tzf dist/llamonitor_async-0.1.0.tar.gz | head -20
unzip -l dist/llamonitor_async-0.1.0-py3-none-any.whl | head -20

# Verify with twine
twine check dist/*
```

**Expected output:** `Checking dist/...: PASSED`

### 6. Test Installation Locally

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/llamonitor_async-0.1.0-py3-none-any.whl

# Test import
python -c "from llmops_monitoring import monitor_llm, MonitorConfig; print('✓ Import successful')"

# Deactivate and cleanup
deactivate
rm -rf test_env
```

### 7. Create PyPI Account (if needed)

1. Go to https://pypi.org/account/register/
2. Verify email address
3. Enable 2FA (required for uploading)
4. Create API token:
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope: "Entire account"
   - Copy token (starts with `pypi-`)
   - **Save it securely** - you can't view it again!

### 8. Upload to PyPI

```bash
# Upload using twine
python -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: <paste your PyPI token here>
```

**Expected output:**
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading llamonitor_async-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading llamonitor_async-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://pypi.org/project/llamonitor-async/0.1.0/
```

### 9. Verify Publication

```bash
# Wait 1-2 minutes, then test installation
pip install llamonitor-async

# Verify version
python -c "import llmops_monitoring; print('Installed successfully!')"
```

Visit your package page: https://pypi.org/project/llamonitor-async/

### 10. Create GitHub Release

```bash
# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release"
git push origin v0.1.0
```

On GitHub:
1. Go to **Releases** → **Create a new release**
2. Choose tag: `v0.1.0`
3. Release title: `v0.1.0 - Initial Public Release`
4. Description:
   ```markdown
   # llamonitor-async v0.1.0 🦙📊

   Initial public release of llamonitor-async - lightweight async monitoring for LLM applications.

   ## Installation

   ```bash
   pip install llamonitor-async
   ```

   ## Key Features

   ✨ **Capacity-Based Measurement** - Track text/image amounts, not tokens
   ⚡ **Async-First Architecture** - <1% overhead with non-blocking writes
   🔌 **Pluggable Storage** - Parquet, PostgreSQL, easily extensible
   🌳 **Hierarchical Tracking** - Multi-agent workflow support
   📊 **Download Tracking** - Built-in statistics collection
   🔓 **100% Open Source** - Apache 2.0 license

   ## Documentation

   - [Quick Start](https://github.com/yourusername/LLMOps_monitoring_async/blob/main/README.md)
   - [Architecture Guide](https://github.com/yourusername/LLMOps_monitoring_async/blob/main/ARCHITECTURE.md)
   - [Download Tracking](https://github.com/yourusername/LLMOps_monitoring_async/blob/main/DOWNLOAD_TRACKING.md)

   ## What's Included

   - Core monitoring framework with decorator API
   - Parquet and PostgreSQL storage backends
   - Docker + Grafana setup for visualization
   - Comprehensive test suite
   - Real OpenAI API examples
   - Download statistics automation

   ## Quick Example

   ```python
   from llmops_monitoring import monitor_llm, MonitorConfig

   @monitor_llm("my_function", measure_text=True)
   async def my_llm_call(prompt: str):
       return {"text": "response..."}
   ```

   ## Links

   - 📦 [PyPI](https://pypi.org/project/llamonitor-async/)
   - 📖 [Documentation](https://github.com/yourusername/LLMOps_monitoring_async)
   - 🐛 [Issues](https://github.com/yourusername/LLMOps_monitoring_async/issues)
   - 💬 [Discussions](https://github.com/yourusername/LLMOps_monitoring_async/discussions)

   **Built with "leave space for air conditioning" philosophy** - designed for tomorrow's features! 🚀
   ```

5. Attach distribution files from `dist/` folder
6. Click **Publish release**

### 11. Post-Publication

#### Update Social Media

Post your LinkedIn announcement (see `LINKEDIN_POST.md`)

#### Monitor Initial Adoption

```bash
# Wait 24 hours, then check stats
python scripts/fetch_download_stats.py
```

#### Watch for Issues

- Monitor GitHub Issues for bug reports
- Check PyPI project page for feedback
- Respond to community questions

#### Enable GitHub Features

- **Discussions**: Settings → Features → Enable Discussions
- **Projects**: Create project board for roadmap
- **Wiki**: Add additional documentation

## ⚠️ Common Issues

### Issue: "User not found"
**Solution:** Check username/token. Username should be `__token__`.

### Issue: "403 Forbidden"
**Solution:** Verify token is for PyPI (not TestPyPI) and has correct permissions.

### Issue: "400 Bad Request"
**Solution:** Package name might be taken. Check https://pypi.org/project/llamonitor-async/

### Issue: Git commit fails
**Solution:** Configure git user identity (see step 1).

### Issue: Build fails
**Solution:** Check `pyproject.toml` for syntax errors.

### Issue: Package import fails after installation
**Solution:** Check package structure and `__init__.py` exports.

## 📋 Final Checklist

Before uploading, confirm:

- [ ] Git user configured
- [ ] All changes committed
- [ ] Pushed to GitHub
- [ ] Built distributions successfully
- [ ] Verified package contents
- [ ] Tested local installation
- [ ] PyPI account created with 2FA
- [ ] API token ready
- [ ] Package name available on PyPI
- [ ] README looks good on GitHub
- [ ] LICENSE file present
- [ ] Version number correct in `pyproject.toml`

After uploading, confirm:

- [ ] Package visible on PyPI
- [ ] Installation works: `pip install llamonitor-async`
- [ ] GitHub release created with tag
- [ ] LinkedIn post published
- [ ] GitHub Discussions enabled
- [ ] Download tracking workflow active

## 🎉 You're Ready!

Once all boxes are checked, run:

```bash
python -m twine upload dist/*
```

**Congratulations on publishing llamonitor-async! 🦙📊**

---

Questions? Check [PUBLISH.md](PUBLISH.md) and [UPLOAD_GUIDE.md](UPLOAD_GUIDE.md).
