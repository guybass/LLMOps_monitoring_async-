# Contributing to LLMOps Monitoring

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Design Philosophy

**"Leave Space for Air Conditioning"**

Every component should have clear extension points. When adding features, consider:
- Will this be easy to extend later?
- Does it follow the existing patterns?
- Are the interfaces clear?

## Areas for Contribution

### 1. Storage Backends

The most impactful contributions! Implement `StorageBackend` for:
- MySQL/MariaDB (similar to PostgreSQL backend)
- ClickHouse (for analytics workloads)
- MongoDB (document-based storage)
- S3 (for archival storage)
- TimescaleDB (optimized time-series)
- Redis (for real-time streaming)

See `llmops_monitoring/transport/backends/postgres.py` as reference.

### 2. Metric Collectors

Add specialized collectors by implementing `MetricCollector`:
- Cost tracking (with model pricing)
- Token counting (actual tokenizer-based)
- Cache hit rates
- Latency patterns
- Audio/video metrics

See `llmops_monitoring/instrumentation/collectors/text.py` as reference.

### 3. Documentation

- Tutorials for specific use cases
- Integration guides (LangChain, LlamaIndex, etc.)
- Architecture deep-dives
- Performance optimization guides

### 4. Visualization

- New Grafana dashboards
- Pre-built queries for common analyses
- Integration with other visualization tools

## Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/guybass/LLMOps_monitoring_async-.git
cd LLMOps_monitoring_async-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linting
black llmops_monitoring/
ruff check llmops_monitoring/
mypy llmops_monitoring/
```

## Code Style

- **Formatting**: Use Black with default settings
- **Linting**: Follow Ruff recommendations
- **Type hints**: Required for all public APIs
- **Docstrings**: Google-style for all public classes/functions

Example:

```python
async def my_function(arg: str, optional: int = 0) -> Dict[str, Any]:
    """
    Brief description of function.

    Args:
        arg: Description of arg
        optional: Description of optional parameter

    Returns:
        Dictionary containing results

    Raises:
        ValueError: When arg is invalid
    """
    pass
```

## Testing

- Write tests for all new features
- Maintain >90% code coverage
- Use pytest fixtures for common setup
- Test both success and failure cases

```python
import pytest
from llmops_monitoring import monitor_llm

@pytest.mark.asyncio
async def test_my_feature():
    # Arrange
    ...

    # Act
    result = await my_function()

    # Assert
    assert result is not None
```

## Pull Request Process

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests
   - Update documentation

3. **Run the full test suite**
   ```bash
   pytest
   black llmops_monitoring/
   ruff check llmops_monitoring/
   mypy llmops_monitoring/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add MySQL storage backend"
   ```

   Use conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring
   - `chore:` for maintenance

5. **Push and create pull request**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **PR checklist**:
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] Example added (if applicable)
   - [ ] CHANGELOG.md updated
   - [ ] Type hints present
   - [ ] Docstrings complete

## Extension Point Guidelines

When adding new extension points:

1. **Define clear interfaces**
   ```python
   from abc import ABC, abstractmethod

   class MyInterface(ABC):
       @abstractmethod
       async def required_method(self) -> None:
           """Clear description of what this must do."""
           pass
   ```

2. **Provide reference implementation**
   - Show the pattern clearly
   - Document all extension points
   - Add to registry if applicable

3. **Document extension**
   - Add to README "Extension Points" section
   - Create example in `examples/`
   - Update architecture documentation

## Community

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Discord**: [Coming soon]

## Questions?

Open an issue or discussion - we're here to help!

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
