# Final comprehensive syntax fix script
$ErrorActionPreference = "Continue"
$fixedCount = 0

Write-Host "Running final comprehensive syntax fixes..."

function Fix-SyntaxIssues {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    
    $changed = $false
    $originalContent = $content
    
    # Fix regex replacement artifacts
    $content = $content -replace '\$1', 'data'
    $content = $content -replace '\$2', 'value'
    
    # Fix broken strings
    $content = $content -replace 'print\("([^"]*)\n([^"]*)"', 'print("$1$2"'
    
    # Fix Dict vs dict consistency
    $content = $content -replace '\bdict\[', 'Dict['
    $content = $content -replace '\blist\[', 'List['
    $content = $content -replace '\bset\[', 'Set['
    $content = $content -replace '\btuple\[', 'Tuple['
    
    if ($content -ne $originalContent) {
        $changed = $true
    }
    
    if ($changed) {
        try {
            Set-Content -Path $FilePath -Value $content -NoNewline
            return $true
        } catch {
            Write-Host "Warning: Could not write to $FilePath - file in use"
            return $false
        }
    }
    return $false
}

# Process files with errors
$errorFiles = @(
    "src\backtest\persistence.py",
    "src\backtest\report_generator.py", 
    "src\backtest\walk_forward.py",
    "src\common\config.py",
    "src\common\enums.py",
    "src\common\interfaces.py",
    "src\common\paths.py",
    "src\common\volume_thresholds.py",
    "src\data\cache.py",
    "src\data\compat.py",
    "src\data\connectors.py",
    "src\data\corporate_actions.py"
)

foreach ($file in $errorFiles) {
    if (Test-Path $file) {
        if (Fix-SyntaxIssues $file) {
            $fixedCount++
            Write-Host "Fixed: $file"
        }
    }
}

Write-Host "Fixed $fixedCount files"
