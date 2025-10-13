# Runtime Test Checklist

Quick checklist for testing llamonitor-async after cleanup.

## Prerequisites

Install dependencies first:
```bash
pip install pydantic aiofiles python-dotenv pandas pyarrow
```

---

## Test 1: Package Import ✓

**Test the main package imports correctly:**

```bash
python -c "from llmops_monitoring import monitor_llm, MonitorConfig, MonitoringWriter; print('✓ Imports successful')"
```

**Expected Output:**
```
✓ Imports successful
```

**If it fails:** Check if dependencies are installed

---

## Test 2: Logging Module ✓

**Test the new logging configuration:**

```bash
python -c "from llmops_monitoring.utils.logging_config import get_logger, configure_logging; logger = get_logger(__name__); logger.info('✓ Logging works'); print('✓ Logging module functional')"
```

**Expected Output:**
```
INFO: ✓ Logging works
✓ Logging module functional
```

---

## Test 3: Basic Monitoring Test ✓

**Run the basic monitoring test:**

```bash
python tests/test_basic_monitoring.py
```

**Expected Output:**
```
INFO: Flushing events to storage...
INFO: ✓ All events written to Parquet files
INFO: ✓ Created X Parquet file(s) in ./test_monitoring_data/
```

**Check:**
- [ ] Creates `test_monitoring_data/` directory
- [ ] Contains `.parquet` files
- [ ] No errors or exceptions

---

## Test 4: Simple Example ✓

**Run the simple monitoring example:**

```bash
python llmops_monitoring/examples/01_simple_example.py
```

**Expected Output:**
```
INFO: Running simple example...
Calling function 1...
Calling function 2...
...
INFO: ✓ Created X Parquet file(s)
```

**Check:**
- [ ] No import errors
- [ ] Uses logger.info() (not print for system messages)
- [ ] Creates monitoring data

---

## Test 5: Fetch Download Stats Script ✓

**Test the download statistics script:**

```bash
python scripts/fetch_download_stats.py --help
```

**Expected Output:**
```
usage: fetch_download_stats.py [-h] [--output-dir OUTPUT_DIR] [--days DAYS]
...
```

**Check:**
- [ ] Script runs without errors
- [ ] Uses logger instead of print for status messages
- [ ] Help text displays correctly

---

## Test 6: Analyze Results Script ✓

**Test the analysis script:**

```bash
python scripts/analyze_results.py --help
```

**Expected Output:**
```
usage: analyze_results.py [-h] [--data-dir DATA_DIR]
...
```

**Check:**
- [ ] Script runs without errors
- [ ] Uses logger for system messages
- [ ] Help text displays correctly

---

## Test 7: Build Package ✓

**Test package building:**

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build
```

**Expected Output:**
```
* Creating virtualenv isolated environment...
* Installing packages in isolated environment...
* Getting build dependencies for sdist...
* Building sdist...
* Building wheel from sdist
...
Successfully built llamonitor_async-0.1.0.tar.gz and llamonitor_async-0.1.0-py3-none-any.whl
```

**Check:**
- [ ] Creates `dist/` directory
- [ ] Contains `.tar.gz` file (source distribution)
- [ ] Contains `.whl` file (wheel)
- [ ] No errors during build

---

## Test 8: Verify Package Contents ✓

**Check what's in the package:**

```bash
# Verify with twine
python -m twine check dist/*

# List wheel contents
unzip -l dist/llamonitor_async-0.1.0-py3-none-any.whl | head -30
```

**Expected Output:**
```
Checking dist/llamonitor_async-0.1.0-py3-none-any.whl: PASSED
Checking dist/llamonitor_async-0.1.0.tar.gz: PASSED
```

**Check:**
- [ ] Both distributions PASS twine check
- [ ] Wheel contains all necessary files
- [ ] No personal files included (no .qodo/, compass_artifact_*, etc.)

---

## Test 9: Test Installation ✓

**Install the package locally:**

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/llamonitor_async-0.1.0-py3-none-any.whl

# Test import
python -c "from llmops_monitoring import monitor_llm, MonitorConfig; print('✓ Package installed and imports work')"

# Cleanup
deactivate
rm -rf test_env
```

**Expected Output:**
```
✓ Package installed and imports work
```

**Check:**
- [ ] Package installs without errors
- [ ] Imports work in fresh environment
- [ ] No missing dependencies

---

## Test 10: Documentation Links ✓

**Verify all documentation links work:**

Open README.md and click each link:

- [ ] [Complete Documentation](docs/README.md)
- [ ] [Quick Start Guide](docs/getting-started/QUICKSTART.md)
- [ ] [Testing Guide](docs/guides/TEST_GUIDE.md)
- [ ] [Download Tracking](docs/guides/DOWNLOAD_TRACKING.md)
- [ ] [Publishing to PyPI](docs/publishing/PUBLISH.md)
- [ ] [Upload Guide](docs/publishing/UPLOAD_GUIDE.md)
- [ ] [Pre-Publish Checklist](docs/publishing/PRE_PUBLISH_CHECKLIST.md)
- [ ] [CONTRIBUTING.md](CONTRIBUTING.md)

**All links should open the correct files.**

---

## Test 11: OpenAI Integration Test (Optional) ✓

**If you have an OpenAI API key:**

```bash
# Set API key in .env
echo "OPENAI_API_KEY=your_key_here" > .env

# Run OpenAI integration test
python tests/test_agent_graph_real.py
```

**Expected Output:**
```
INFO: OpenAI API key found
INFO: Initializing monitoring...
...
INFO: ✓ Test completed successfully
```

**Check:**
- [ ] Connects to OpenAI API
- [ ] Creates monitoring data
- [ ] Tracks hierarchical agent calls
- [ ] Uses logger instead of print

---

## Summary Checklist

After running all tests, verify:

### Code Quality:
- [ ] All imports work
- [ ] Logging module functional
- [ ] No print() in library code (only logger calls)
- [ ] User-facing output still uses print()

### Project Structure:
- [ ] Tests in `tests/` directory
- [ ] Scripts in `scripts/` directory
- [ ] Documentation in `docs/` directory
- [ ] Root directory is clean (10 essential files only)

### Functionality:
- [ ] Basic monitoring test passes
- [ ] Examples run without errors
- [ ] Scripts work correctly
- [ ] Package builds successfully

### Documentation:
- [ ] All documentation links work
- [ ] README is clear and organized
- [ ] Guides are accessible

### Package Quality:
- [ ] Package builds without errors
- [ ] Twine check passes
- [ ] Installation works in fresh environment
- [ ] No personal files in distribution

---

## If Any Test Fails

### Import Errors:
```bash
# Install missing dependencies
pip install pydantic aiofiles python-dotenv pandas pyarrow
```

### Syntax Errors:
```bash
# Check syntax
python -m compileall llmops_monitoring/ scripts/ tests/
```

### Documentation Links:
```bash
# Verify files exist
ls -la docs/README.md
ls -la docs/getting-started/QUICKSTART.md
# etc.
```

### Build Errors:
```bash
# Clean and retry
rm -rf dist/ build/ *.egg-info/
python -m pip install --upgrade build twine
python -m build
```

---

## Quick Test Command

Run all basic tests in one go:

```bash
# Install dependencies
pip install pydantic aiofiles python-dotenv pandas pyarrow

# Run tests
echo "Test 1: Imports"
python -c "from llmops_monitoring import monitor_llm; print('✓')"

echo "Test 2: Logging"
python -c "from llmops_monitoring.utils.logging_config import get_logger; print('✓')"

echo "Test 3: Basic monitoring"
python tests/test_basic_monitoring.py

echo "Test 4: Simple example"
python llmops_monitoring/examples/01_simple_example.py

echo "Test 5: Build package"
python -m build

echo "✅ All tests completed!"
```

---

## Success Criteria

All tests should:
- ✅ Run without errors
- ✅ Use logging instead of print (for system messages)
- ✅ Create proper output files
- ✅ Have clear, professional output

If all tests pass: **Project is production-ready! 🎉**

---

**For detailed test results, see TEST_REPORT.md**
