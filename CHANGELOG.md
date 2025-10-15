# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-10-15

### Added
- **MySQL Backend**: Complete MySQL storage backend with connection pooling, automatic schema creation, and optimized indexes
  - Async connection pooling with aiomysql
  - InnoDB engine with UTF8MB4 charset
  - 6 optimized indexes for hierarchical queries
  - Smart connection string parsing
  - Comprehensive test suite (tests/test_mysql_backend.py)
  - Example: llmops_monitoring/examples/04_mysql_backend.py

- **Built-in Cost Calculation**: Automatic cost tracking for 18+ major LLM models
  - Pricing database for OpenAI, Anthropic, Google, Meta, and Mistral models
  - Support for exact token counts (input_tokens/output_tokens)
  - Estimation from char_count using 4:1 char-to-token ratio
  - Custom pricing overrides
  - Model name matching (exact + prefix)
  - Cost breakdown (text + image)
  - Example: llmops_monitoring/examples/05_cost_calculation.py

### Changed
- Updated .gitignore for production-ready repository
- Enhanced documentation with cost tracking examples
- Improved QUICKSTART guide with Step 5: Built-in Cost Tracking

### Documentation
- Added MYSQL_BACKEND_COMPLETED.md feature documentation
- Updated README.md with cost tracking examples
- Updated roadmap marking MySQL and cost calculation as complete

## [0.1.1] - 2025-10-14

### Added
- Project restructuring and cleanup
- Centralized logging with llmops_monitoring/utils/logging_config.py
- Professional .gitignore with 400+ patterns
- Comprehensive documentation organization

### Changed
- Refactored all print statements to use proper logging
- Moved documentation to docs/ directory structure
- Enhanced README with better examples

## [0.1.0] - 2025-10-13

### Added
- Initial release
- Async-first monitoring architecture
- Hierarchical tracking with automatic parent-child relationships
- Parquet and PostgreSQL storage backends
- Text and image metric collectors
- Decorator-based API (@monitor_llm)
- Docker Compose stack with Grafana
- Comprehensive documentation

### Features
- Non-blocking metric collection with buffered batch writes
- Flexible metrics (text: chars, words, bytes; images: count, pixels, file size)
- Pluggable storage backends
- Custom collector support
- Production-ready error handling and graceful shutdown
