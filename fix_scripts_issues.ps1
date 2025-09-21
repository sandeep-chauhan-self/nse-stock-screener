# Comprehensive script to fix remaining issues in scripts files
$ErrorActionPreference = "Continue"

Write-Host "Fixing remaining issues in scripts files..."

# Fix check_deps.py
$content = Get-Content "scripts\check_deps.py" -Raw -ErrorAction SilentlyContinue
if ($content) {
    # Add return type hints to methods
    $content = $content -replace '(def print_header\(self, title: str\)):', '$1 -> None:'
    $content = $content -replace '(def print_status\(self, message: str, status: str = "info", indent: int = 0\)):', '$1 -> None:'
    $content = $content -replace '(def _create_backup\(self\)):', '$1 -> None:'
    $content = $content -replace '(def show_instructions\(self\)):', '$1 -> None:'
    $content = $content -replace '(def main\(\)):', '$1 -> None:'
    
    Set-Content -Path "scripts\check_deps.py" -Value $content -NoNewline -ErrorAction SilentlyContinue
    Write-Host "Fixed check_deps.py"
}

# Fix code_cleanup.py
$content = Get-Content "scripts\code_cleanup.py" -Raw -ErrorAction SilentlyContinue
if ($content) {
    # Add class inheritance
    $content = $content -replace '^class (\w+):', 'class $1(object):'
    
    # Add return type hints
    $content = $content -replace '(def _create_backup\(self\)):', '$1 -> None:'
    $content = $content -replace '(def main\(\)):', '$1 -> None:'
    
    # Fix type instantiation issues
    $content = $content -replace 'Set\(\)', 'set()'
    $content = $content -replace 'List\(\)', 'list()'
    $content = $content -replace 'Dict\(\)', 'dict()'
    
    Set-Content -Path "scripts\code_cleanup.py" -Value $content -NoNewline -ErrorAction SilentlyContinue
    Write-Host "Fixed code_cleanup.py"
}

# Fix code_hygiene_analyzer.py
$content = Get-Content "scripts\code_hygiene_analyzer.py" -Raw -ErrorAction SilentlyContinue
if ($content) {
    # Add class inheritance to all classes
    $content = $content -replace '^class (\w+):', 'class $1(object):'
    
    # Fix field default factories
    $content = $content -replace 'field\(default_factory=List\[str\]\)', 'field(default_factory=list)'
    $content = $content -replace 'field\(default_factory=List\[FunctionInfo\]\)', 'field(default_factory=list)'
    $content = $content -replace 'field\(default_factory=List\[ImportInfo\]\)', 'field(default_factory=list)'
    $content = $content -replace 'field\(default_factory=List\[ClassInfo\]\)', 'field(default_factory=list)'
    $content = $content -replace 'field\(default_factory=List\[HygieneIssue\]\)', 'field(default_factory=list)'
    
    # Fix type instantiation
    $content = $content -replace 'Set\(\)', 'set()'
    $content = $content -replace 'List\(\)', 'list()'
    $content = $content -replace 'Dict\(\)', 'dict()'
    
    # Add return type hints to methods
    $content = $content -replace '(def visit_Import\(self, node: ast\.Import\)):', '$1 -> None:'
    $content = $content -replace '(def visit_ImportFrom\(self, node: ast\.ImportFrom\)):', '$1 -> None:'
    $content = $content -replace '(def visit_FunctionDef\(self, node: ast\.FunctionDef\)):', '$1 -> None:'
    $content = $content -replace '(def visit_ClassDef\(self, node: ast\.ClassDef\)):', '$1 -> None:'
    
    Set-Content -Path "scripts\code_hygiene_analyzer.py" -Value $content -NoNewline -ErrorAction SilentlyContinue
    Write-Host "Fixed code_hygiene_analyzer.py"
}

Write-Host "Completed fixing scripts files"
