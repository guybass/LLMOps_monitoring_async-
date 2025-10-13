# Publishing llamonitor to PyPI

Step-by-step guide to publish your package.

## Prerequisites

```bash
# Install build tools
pip install build twine
```

## Step 1: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create account and verify email
3. Enable 2FA (required for publishing)
4. Create API token:
   - Go to https://pypi.org/manage/account/
   - Scroll to "API tokens"
   - Click "Add API token"
   - Name: "llamonitor-upload"
   - Scope: "Entire account" (or specific project after first upload)
   - **SAVE THE TOKEN** - you won't see it again!

## Step 2: Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution files
python -m build
```

This creates:
- `dist/llamonitor-0.1.0.tar.gz` (source distribution)
- `dist/llamonitor-0.1.0-py3-none-any.whl` (wheel)

## Step 3: Test on TestPyPI (Optional but Recommended)

```bash
# Upload to test server
python -m twine upload --repository testpypi dist/*

# You'll be prompted for username and password:
# Username: __token__
# Password: <your-test-pypi-token>

# Test installation
pip install --index-url https://test.pypi.org/simple/ llamonitor
```

## Step 4: Upload to PyPI

```bash
# Upload to real PyPI
python -m twine upload dist/*

# You'll be prompted for username and password:
# Username: __token__
# Password: <your-pypi-token>
```

## Step 5: Verify

```bash
# Check it's live
pip search llamonitor  # (if search is enabled)

# Or visit:
# https://pypi.org/project/llamonitor/

# Test installation
pip install llamonitor
```

## Step 6: Create GitHub Release

```bash
# Tag the release
git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release"
git push origin v0.1.0

# Or create release on GitHub UI
```

## Using API Token in CI/CD

Create `.pypirc` file (DON'T commit this!):

```ini
[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
username = __token__
password = pypi-your-test-token-here
```

Or use environment variables:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-token-here
python -m twine upload dist/*
```

## Updating the Package

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Rebuild and upload:

```bash
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine upload dist/*
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

## Troubleshooting

### "Package already exists"
You can't re-upload the same version. Bump version in `pyproject.toml`.

### "Invalid token"
Make sure you're using `__token__` as username and the full token (including `pypi-`) as password.

### "File already exists"
Someone else has this package name. Choose a different name.

### Build errors
Check `pyproject.toml` syntax and ensure all dependencies are listed.

## Post-Publication Checklist

- [ ] Package appears on PyPI
- [ ] Installation works: `pip install llamonitor`
- [ ] README displays correctly on PyPI
- [ ] Update GitHub repo description to mention PyPI
- [ ] Post announcement on:
  - [ ] Twitter/X
  - [ ] Reddit r/Python
  - [ ] HackerNews
  - [ ] LinkedIn
- [ ] Add PyPI badge to README (already included)

## Marketing Copy

**For social media:**
```
🦙📊 Just published llamonitor - a lightweight async monitoring framework for LLM apps!

✨ Measures text/image CAPACITY (not tokens)
⚡ Async-first, non-blocking
🔌 Pluggable storage (Parquet, PostgreSQL)
🌳 Hierarchical tracking for agent workflows
🔓 100% open source (Apache 2.0)

pip install llamonitor

https://github.com/guybass/LLMOps_monitoring_async-
https://pypi.org/project/llamonitor/

Alternative to Langfuse/LangSmith with focus on extensibility!

#Python #LLM #AI #OpenSource
```

Good luck! 🚀
