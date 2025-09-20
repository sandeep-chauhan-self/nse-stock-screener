# Test Data and Fixtures

This directory contains comprehensive test data and fixtures for the NSE Stock Screener system, ensuring reproducible and reliable testing across all environments.

## Directory Structure

```
data/test/
├── README.md                     # This file
├── *.csv                        # Stock price data for 20 symbols (1 year)
├── *_quick_test.csv             # Quick test data (30 days)
├── test_symbols.txt             # List of test symbols
├── .env.test                    # Test environment configuration
├── configs/                     # Test configurations
│   ├── conservative_test.json   # Conservative trading parameters
│   ├── aggressive_test.json     # Aggressive trading parameters
│   └── quick_test.json         # Quick test parameters
├── scenarios/                   # Market scenario definitions
│   ├── bullish_market.json     # Bull market conditions
│   ├── bearish_market.json     # Bear market conditions
│   ├── sideways_market.json    # Sideways market conditions
│   └── high_volatility.json    # High volatility conditions
└── sample_results/             # Expected test outputs
    ├── sample_analysis.json    # Sample analysis results
    └── sample_analysis.csv     # Sample CSV report
```

## Test Data Generation

### Stock Data
- **Symbols**: 20 major NSE stocks (RELIANCE, TCS, INFY, etc.)
- **Period**: 1 year of synthetic daily OHLCV data
- **Quick Tests**: 30-day datasets for faster testing
- **Realistic**: Price movements follow realistic statistical distributions

### Data Characteristics
- **Deterministic**: Same symbol always generates same data (for reproducibility)
- **Realistic**: Proper OHLC relationships, volume correlations
- **Complete**: No missing data or gaps
- **Varied**: Different price ranges and volatilities per symbol

## Usage in Tests

### Loading Test Data
```python
import pandas as pd
from tests.test_fixtures import TestDataManager

# Using test data manager
test_manager = TestDataManager()
data = test_manager.get_test_stock_data("RELIANCE")
quick_data = test_manager.get_test_stock_data("RELIANCE", period="quick")

# Direct loading
data = pd.read_csv("data/test/RELIANCE_test_data.csv", index_col="Date", parse_dates=True)
```

### Using Test Configurations
```python
import json
from tests.test_fixtures import TestDataManager

test_manager = TestDataManager()
config = test_manager.get_test_config("quick_test")

# Or direct loading
with open("data/test/configs/quick_test.json") as f:
    config = json.load(f)
```

### Testing Market Scenarios
```python
from tests.test_fixtures import market_data_mocker

# Generate bullish market data
bullish_data = market_data_mocker.generate_scenario_data("bullish", days=30)

# Test with specific scenario
scenario = test_manager.get_test_scenario("bullish_market")
symbols = scenario["test_symbols"]
```

## Test Fixtures

### Key Fixtures Available
- `test_data_manager`: Session-scoped data manager
- `sample_stock_data`: In-memory sample data
- `mock_yfinance_data`: Mocked yfinance API
- `test_config`: Default test configuration
- `temp_output_dir`: Temporary output directory
- `mock_redis`: Mocked Redis for caching tests
- `performance_monitor`: Performance monitoring
- `market_data_mocker`: Market scenario data generator

### Using Fixtures in Tests
```python
import pytest

def test_stock_analysis(test_data_manager, test_config, temp_output_dir):
    """Test stock analysis with fixtures."""
    # Get test data
    data = test_data_manager.get_test_stock_data("RELIANCE", period="quick")
    
    # Use test configuration
    analysis = run_analysis(data, test_config)
    
    # Save to temporary directory
    output_file = temp_output_dir / "test_results.json"
    save_results(analysis, output_file)
    
    # Assertions
    assert analysis["score"] > 0
    assert output_file.exists()
```

## Performance Testing

### Memory and Time Constraints
```python
def test_performance(performance_monitor):
    """Test with performance monitoring."""
    performance_monitor.start()
    
    # Run your test
    result = heavy_computation()
    
    # Assert performance constraints
    performance_monitor.assert_performance(max_time=10, max_memory_mb=100)
```

## Integration Testing

### Mock Environment
```python
def test_integration(mock_file_system, mock_yfinance_data):
    """Integration test with mocked dependencies."""
    # File operations go to temporary directory
    # yfinance calls return test data
    
    result = run_full_analysis()
    
    assert_analysis_results(result, min_stocks=5)
```

## Environment Configuration

### Test Environment Variables
The `.env.test` file contains test-specific configurations:
- `TEST_MODE=true`: Enables test mode
- `MOCK_API_CALLS=true`: Uses mock data instead of live APIs
- `LOG_LEVEL=DEBUG`: Verbose logging for debugging
- `ENABLE_CACHE=false`: Disables caching for predictable tests

### Loading Test Environment
```python
import os
from pathlib import Path

# Load test environment
test_env_path = Path("data/test/.env.test")
if test_env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(test_env_path)
```

## Creating New Test Data

### Generate Fresh Test Data
```python
# Run the test data generator
python scripts/create_test_data.py
```

### Custom Test Scenarios
1. Add new scenario to `scenarios/` directory
2. Follow the JSON format of existing scenarios
3. Use in tests via `test_manager.get_test_scenario("your_scenario")`

### Adding New Stock Symbols
1. Edit `scripts/create_test_data.py`
2. Add symbols to `create_sample_stock_symbols()`
3. Regenerate test data

## Test Categories

### Pytest Markers
- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, with dependencies)
- `@pytest.mark.slow`: Slow tests (may fetch live data)
- `@pytest.mark.requires_network`: Tests needing network access
- `@pytest.mark.requires_talib`: Tests needing TA-Lib library

### Running Specific Test Categories
```bash
# Unit tests only (fast)
pytest tests/ -m "unit" -v

# Integration tests with test data
pytest tests/ -m "integration" -v

# All tests except slow ones
pytest tests/ -m "not slow" -v

# Performance tests
pytest tests/ -m "performance" -v
```

## Continuous Integration

### CI/CD Usage
The test data is used in GitHub Actions workflows:
1. **Quick Tests**: Use `*_quick_test.csv` files for fast CI feedback
2. **Full Tests**: Use complete datasets for thorough testing
3. **Scenario Tests**: Test different market conditions
4. **Performance Tests**: Ensure code performance standards

### Test Data Validation
The CI pipeline validates:
- Test data integrity (no missing values, proper formats)
- Configuration validity (JSON schema validation)
- Performance baselines (execution time, memory usage)
- Expected results consistency

## Best Practices

### Test Data Management
1. **Deterministic**: Always use the same test data for reproducible results
2. **Realistic**: Test data should reflect real market conditions
3. **Complete**: Cover edge cases and different market scenarios
4. **Isolated**: Each test should be independent
5. **Clean**: Clean up temporary files and state after tests

### Performance Considerations
- Use quick test data for unit tests
- Use full datasets only for integration tests
- Monitor memory usage in performance tests
- Set reasonable time limits for test execution

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Add docstrings explaining test purpose
- Use appropriate pytest markers
- Keep tests focused and atomic

## Troubleshooting

### Common Issues
1. **Missing Test Data**: Run `python scripts/create_test_data.py`
2. **Import Errors**: Ensure PYTHONPATH includes `src/`
3. **Permission Errors**: Check write permissions on test output directories
4. **Memory Issues**: Use quick test data or reduce dataset size

### Debugging Tests
```bash
# Verbose output with test details
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Debug specific test
pytest tests/test_specific.py::test_function -v -s

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

When adding new tests:
1. Use existing fixtures when possible
2. Create new fixtures for reusable test data
3. Follow naming conventions
4. Add appropriate pytest markers
5. Document complex test scenarios
6. Ensure tests are reproducible and isolated