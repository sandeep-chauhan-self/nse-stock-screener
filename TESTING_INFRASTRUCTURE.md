# Testing & CI Infrastructure Documentation

## Overview

This document describes the comprehensive testing and CI infrastructure implemented for the NSE Stock Screener project as part of **Requirement 3.11: Tests, CI, reproducibility and dependency management**.

## 🏗️ Architecture

### Dependencies Management
- **requirements.txt**: Pinned production dependencies with specific versions
- **pyproject.toml**: Modern Python project configuration with build system, optional dependencies, and tool configurations

### Testing Framework
- **pytest**: Primary testing framework with extensive plugin ecosystem
- **Test Fixtures**: Deterministic test data in `tests/fixtures/test_data.py`
- **Coverage**: Code coverage measurement with pytest-cov
- **Mocking**: Test isolation with pytest-mock

### Containerization
- **Dockerfile**: Multi-stage build with security best practices
- **docker-compose.yml**: Development and production environments
- **Health checks**: Container health monitoring

### CI/CD Pipeline
- **GitHub Actions**: Automated testing, linting, security scanning
- **Multi-platform**: Testing across Python 3.9, 3.10, 3.11
- **Release automation**: Automated package building and publishing

## 📂 Project Structure

```
nse-stock-screener/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Main CI pipeline
│       └── release.yml         # Release automation
├── tests/
│   ├── conftest.py            # pytest configuration
│   ├── fixtures/
│   │   └── test_data.py       # Test data fixtures
│   ├── test_indicators.py     # Technical indicator tests
│   ├── test_scoring.py        # Scoring system tests
│   ├── test_risk_management.py # Risk management tests
│   └── test_backtesting.py    # Backtesting tests
├── scripts/
│   └── check_deps.py          # Enhanced dependency checker
├── requirements.txt           # Production dependencies
├── pyproject.toml            # Project configuration
├── Dockerfile                # Container definition
└── docker-compose.yml        # Multi-container setup
```

## 🔧 Dependencies

### Core Dependencies (requirements.txt)
```
pandas>=2.0.0          # Data manipulation and analysis
numpy>=1.24.0          # Numerical computing
matplotlib>=3.6.0      # Plotting and visualization
yfinance>=0.2.18       # Financial data retrieval
requests>=2.28.0       # HTTP requests
scipy>=1.10.0          # Scientific computing
scikit-learn>=1.2.0    # Machine learning
plotly>=5.17.0         # Interactive plotting
ta-lib>=0.4.25         # Technical analysis (optional)
```

### Development Dependencies (pyproject.toml)
```
pytest>=7.0.0          # Testing framework
pytest-cov>=4.0.0      # Coverage plugin
pytest-mock>=3.10.0    # Mocking support
pytest-xdist>=3.0.0    # Parallel testing
ruff>=0.1.0            # Fast Python linter
mypy>=1.5.0            # Static type checking
black>=23.0.0          # Code formatting
isort>=5.12.0          # Import sorting
bandit>=1.7.0          # Security linting
pre-commit>=3.0.0      # Git hooks
```

## 🧪 Testing Strategy

### Test Categories

1. **Unit Tests**: Individual component testing
   - Technical indicators (RSI, ATR, MACD, ADX)
   - Volume indicators and Z-score calculations
   - Scoring algorithms and regime adjustments
   - Risk management functions
   - Position sizing calculations

2. **Integration Tests**: Component interaction testing
   - Data pipeline integration
   - Scoring system integration
   - Backtesting workflow

3. **Performance Tests**: Performance and resource testing
   - Large dataset processing
   - Memory usage validation
   - Execution time benchmarks

### Test Data Strategy

**Deterministic Test Data** (`tests/fixtures/test_data.py`):
- Fixed OHLCV data for consistent results
- Expected indicator values for validation
- Edge cases and boundary conditions
- Market regime scenarios

**Example Test Data**:
```python
def create_test_ohlcv_data(length: int = 100) -> pd.DataFrame:
    """Create deterministic OHLCV data for testing"""
    base_price = 100.0
    dates = pd.date_range('2023-01-01', periods=length, freq='D')
    
    # Create predictable price movements
    trend = np.linspace(0, 10, length)  # Upward trend
    noise = np.sin(np.linspace(0, 4*np.pi, length)) * 2  # Cyclical noise
    
    close_prices = base_price + trend + noise
    # ... rest of OHLCV calculation
```

### Test Execution

**Local Testing**:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/test_indicators.py -v

# Run performance tests
pytest tests/ -m "performance" --timeout=60
```

**Parallel Testing**:
```bash
# Run tests in parallel
pytest tests/ -n auto
```

## 🐳 Docker Setup

### Multi-Stage Dockerfile

**Builder Stage**:
- Installs build dependencies
- Compiles Python packages
- Optimizes for build speed

**Runtime Stage**:
- Minimal runtime environment
- Non-root user for security
- Health checks included

**Usage**:
```bash
# Build image
docker build -t nse-screener .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output nse-screener

# Development with docker-compose
docker-compose up --build
```

### Security Features
- Non-root user execution
- Minimal attack surface
- No unnecessary packages
- Security scanning in CI

## 🚀 CI/CD Pipeline

### Main CI Pipeline (.github/workflows/ci.yml)

**9 Parallel Jobs**:

1. **code-quality**: Linting, formatting, type checking, security scanning
2. **unit-tests**: Multi-version testing (Python 3.9, 3.10, 3.11)
3. **integration-tests**: Import validation and basic functionality
4. **docker-build**: Container build testing
5. **performance-tests**: Resource-intensive test execution
6. **security-scan**: Dependency vulnerability scanning
7. **docs-build**: Documentation generation
8. **deploy-prep**: Docker image publishing preparation
9. **notify**: Results aggregation and notification

**Triggers**:
- Push to main branch
- Pull requests
- Manual workflow dispatch

### Release Pipeline (.github/workflows/release.yml)

**Automated Release Process**:
- Version extraction from git tags
- Full test suite execution
- Python package building (wheel + source)
- Docker image building (multi-platform)
- GitHub release creation
- Asset upload (packages, checksums)
- Optional PyPI publishing

**Triggers**:
- Git tag push (v*.*.*)
- Release publication

## 📋 Quality Gates

### Code Quality Checks
- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Import sorting**: Consistent import organization

### Test Requirements
- **Minimum coverage**: 80% code coverage
- **All tests pass**: Zero test failures allowed
- **Performance bounds**: Tests must complete within time limits
- **Security checks**: No high-severity vulnerabilities

### Documentation Requirements
- **API documentation**: All public functions documented
- **Type hints**: Full type annotation coverage
- **Examples**: Usage examples for key functions

## 🔍 Dependency Management

### Enhanced Dependency Checker

**Comprehensive Validation**:
```bash
python scripts/check_deps.py
```

**Features**:
- Python version validation (3.9+)
- Core dependency verification
- Testing framework validation
- Development tool checks
- Docker environment validation
- Project structure verification
- Quick functionality tests

**Output Example**:
```
==============================================================
                    Python Version Check                     
==============================================================
✓ Python 3.11.5 ✓

==============================================================
                  Core Dependencies Check                    
==============================================================
  ✓ pandas 2.0.3 ✓
  ✓ numpy 1.24.3 ✓
  ✓ matplotlib 3.7.1 ✓
  ✓ yfinance 0.2.18 ✓
```

### Reproducible Environments

**Requirements Management**:
- Pinned versions for production stability
- Version ranges for development flexibility
- Optional dependencies for advanced features
- Clear dependency groups (core, dev, test, production)

**Environment Options**:
1. **Local development**: pip install -r requirements.txt
2. **Development with extras**: pip install -e ".[dev]"
3. **Docker development**: docker-compose up
4. **Production deployment**: Docker image with pinned dependencies

## 📊 Monitoring & Reporting

### Test Reports
- **Coverage reports**: HTML and XML formats
- **Performance metrics**: Execution time tracking
- **Dependency analysis**: Security and update notifications

### CI Artifacts
- Test results and coverage reports
- Built packages and Docker images
- Security scan results
- Documentation builds

### Notifications
- GitHub commit status checks
- Pull request status updates
- Release notifications
- Failure alerts

## 🛠️ Development Workflow

### Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd nse-stock-screener

# 2. Check dependencies
python scripts/check_deps.py

# 3. Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. Run tests
pytest tests/ -v

# 5. Check code quality
ruff check .
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Docker Development
```bash
# Development environment
docker-compose up --build

# Production testing
docker build -t nse-screener .
docker run -v $(pwd)/data:/app/data nse-screener
```

## 🚨 Troubleshooting

### Common Issues

**1. Dependency Installation Failures**:
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Check dependency conflicts
pip check
```

**2. Test Failures**:
```bash
# Run specific failing test
pytest tests/test_indicators.py::test_rsi_calculation -v

# Debug with print statements
pytest tests/test_indicators.py -s

# Run with coverage to see untested code
pytest tests/ --cov=src --cov-report=html
```

**3. Docker Issues**:
```bash
# Check Docker daemon
docker info

# Rebuild without cache
docker build --no-cache -t nse-screener .

# Check container logs
docker logs <container-id>
```

### Performance Optimization

**Large Dataset Testing**:
- Use data sampling for unit tests
- Implement performance benchmarks
- Monitor memory usage
- Profile critical functions

**CI Performance**:
- Parallel test execution
- Dependency caching
- Incremental testing
- Docker layer caching

## 📈 Future Enhancements

### Planned Improvements
1. **Advanced Testing**:
   - Property-based testing with Hypothesis
   - Mutation testing for test quality
   - Load testing for production scenarios

2. **Enhanced CI/CD**:
   - Deployment automation
   - A/B testing infrastructure
   - Canary deployments

3. **Monitoring**:
   - Application performance monitoring
   - Error tracking and alerting
   - Usage analytics

4. **Security**:
   - SAST/DAST integration
   - Dependency update automation
   - Security policy enforcement

## 📚 References

- [pytest Documentation](https://docs.pytest.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)
- [Code Coverage Best Practices](https://coverage.readthedocs.io/)

---

**Note**: This infrastructure provides a solid foundation for production-ready development with comprehensive testing, automated quality checks, and reliable deployment processes.