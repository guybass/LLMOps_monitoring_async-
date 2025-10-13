# Quick Upload Guide - llamonitor to PyPI

## Step 1: Get Your PyPI Token

1. **Go to PyPI**: https://pypi.org/manage/account/token/

2. **Click "Add API token"**

3. **Fill in**:
   - Token name: `llamonitor-upload`
   - Scope: **"Entire account"** (required for first upload)

4. **COPY THE TOKEN** - You'll only see it once!
   - Format: `pypi-AgEIcHlwaS5vcmcC...` (long string)

## Step 2: Upload to PyPI

```bash
# In your project directory
python -m twine upload dist/*
```

**When prompted:**
- Username: `__token__`
- Password: `<paste-your-token-here>`

## Step 3: Success!

You should see:
```
Uploading llamonitor-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.9/54.9 kB • 00:00 • ?

View at:
https://pypi.org/project/llamonitor/0.1.0/
```

## Step 4: Test Installation

```bash
# In a new terminal/environment
pip install llamonitor

# Test it works
python -c "from llamonitor import monitor_llm; print('✓ llamonitor works!')"
```

## Step 5: Create GitHub Release

```bash
git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release"
git push origin v0.1.0
```

Then create release on GitHub:
- Go to https://github.com/guybass/LLMOps_monitoring_async-/releases/new
- Tag: v0.1.0
- Title: "llamonitor v0.1.0 - Initial Release"
- Description: Copy from PUBLISH.md marketing copy

---

## Troubleshooting

### "Invalid token"
- Make sure you're using `__token__` as username (exactly like that)
- Password should start with `pypi-`

### "Package already exists"
- Someone else has the name
- Choose a different name: `llamonitor-async`, `llama-monitor`, etc.

### "403 Forbidden"
- Wrong token scope - use "Entire account" for first upload
- Or wrong platform token (PyPI vs TestPyPI)

### "File already exists"
- You tried uploading same version twice
- Bump version in `pyproject.toml` and rebuild

---

## After Publishing

✅ Visit: https://pypi.org/project/llamonitor/

✅ Badge will work: `[![PyPI](https://img.shields.io/pypi/v/llamonitor.svg)](https://pypi.org/project/llamonitor/)`

✅ Anyone can: `pip install llamonitor`

✅ Post announcements! (See PUBLISH.md for copy)

---

🎉 **Congratulations on publishing llamonitor!** 🦙📊
