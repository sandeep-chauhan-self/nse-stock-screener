# Create script to add missing return type hints in bulk
$content = Get-Content "examples\fs4_complete_example.py" -Raw

# Add return type hints to functions
$content = $content -replace 'def demonstrate_scoring_engine\(\):', 'def demonstrate_scoring_engine() -> None:'
$content = $content -replace 'def demonstrate_parameter_persistence\(\):', 'def demonstrate_parameter_persistence() -> None:'
$content = $content -replace 'def demonstrate_calibration_harness\(\):', 'def demonstrate_calibration_harness() -> None:'
$content = $content -replace 'def demonstrate_yaml_configuration\(\):', 'def demonstrate_yaml_configuration() -> None:'
$content = $content -replace 'def main\(\):', 'def main() -> None:'

# Fix the generate_sample_indicators return type
$content = $content -replace 'def generate_sample_indicators\(symbol: str, data: pd\.DataFrame\) -> Dict\[str, float\]:', 'def generate_sample_indicators(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:'

Set-Content -Path "examples\fs4_complete_example.py" -Value $content -NoNewline
Write-Host "Added return type hints to fs4_complete_example.py"
