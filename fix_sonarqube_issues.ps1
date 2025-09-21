# PowerShell script to fix common SonarQube issues project-wide
$ErrorActionPreference = "Continue"
$count = 0

Write-Host "Fixing SonarQube issues across the project..."

# Function to fix specific patterns
function Fix-SonarQubeIssues {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    
    $changed = $false
    $originalContent = $content
    
    # Fix trailing comments - move to previous line
    if ($content -match '([^#\n]*[^\s#])(\s+)(#[^\n]*?)(\n)') {
        $content = $content -replace '([^#\n]*[^\s#])(\s+)(#[^\n]*?)(\n)', '$1$4$2$3$4'
        $changed = $true
    }
    
    # Fix class inheritance - add (object)
    if ($content -match '^class\s+(\w+)\s*:' -and $content -notmatch '^class\s+\w+\s*\([^)]*\)\s*:') {
        $content = $content -replace '^(class\s+\w+)\s*:', '$1(object):'
        $changed = $true
    }
    
    # Fix constructor return type hints
    if ($content -match 'def __init__\(.*?\):(?!\s*->\s*None)') {
        $content = $content -replace '(def __init__\([^)]*\)):', '$1 -> None:'
        $changed = $true
    }
    
    # Fix main function return type
    if ($content -match '^def main\(\)\s*:' -and $content -notmatch '^def main\(\)\s*->\s*None\s*:') {
        $content = $content -replace '^(def main\(\))\s*:', '$1 -> None:'
        $changed = $true
    }
    
    # Fix f-strings with no variables
    if ($content -match 'f"([^"]*?)"' -and $content -match 'f"[^{]*?"') {
        $content = $content -replace 'f"([^{}"]*?)"', '"$1"'
        $changed = $true
    }
    
    # Fix built-in generic types (use lowercase)
    $content = $content -replace '\bDict\[', 'dict['
    $content = $content -replace '\bList\[', 'list['
    $content = $content -replace '\bSet\[', 'set['
    $content = $content -replace '\bTuple\[', 'tuple['
    
    if ($content -ne $originalContent) {
        $changed = $true
    }
    
    if ($changed) {
        Set-Content -Path $FilePath -Value $content -NoNewline
        return $true
    }
    return $false
}

# Process all Python files
Get-ChildItem -Path "src", "scripts", "docker", "examples" -Include "*.py" -Recurse | ForEach-Object {
    if (Fix-SonarQubeIssues $_.FullName) {
        $count++
        Write-Host "Fixed SonarQube issues in: $($_.Name)"
    }
}

Write-Host "Total files with SonarQube fixes applied: $count"
