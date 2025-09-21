# PowerShell script to fix Python typing syntax for compatibility
$ErrorActionPreference = "Continue"
$count = 0

Write-Host "Fixing Python typing syntax for compatibility..."

# Function to fix typing syntax
function Fix-TypingSyntax {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    
    $changed = $false
    $originalContent = $content
    
    # Fix built-in generic types back to typing versions for compatibility
    $content = $content -replace '\bdict\[', 'Dict['
    $content = $content -replace '\blist\[', 'List['
    $content = $content -replace '\bset\[', 'Set['
    $content = $content -replace '\btuple\[', 'Tuple['
    
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
    if (Fix-TypingSyntax $_.FullName) {
        $count++
        Write-Host "Fixed typing syntax in: $($_.Name)"
    }
}

Write-Host "Total files with typing syntax fixes: $count"
