# PowerShell script to fix all remaining syntax issues
$ErrorActionPreference = "Continue"
$count = 0

Write-Host "Fixing remaining syntax issues..."

function Fix-AllSyntaxIssues {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    
    $changed = $false
    $originalContent = $content
    
    # Fix dict[...] -> Dict[...]
    $content = $content -replace '\bdict\[([^\]]+)\]', 'Dict[$1]'
    
    # Fix list[...] -> List[...]
    $content = $content -replace '\blist\[([^\]]+)\]', 'List[$1]'
    
    # Fix set[...] -> Set[...]
    $content = $content -replace '\bset\[([^\]]+)\]', 'Set[$1]'
    
    # Fix tuple[...] -> Tuple[...]
    $content = $content -replace '\btuple\[([^\]]+)\]', 'Tuple[$1]'
    
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
    if (Fix-AllSyntaxIssues $_.FullName) {
        $count++
        Write-Host "Fixed syntax in: $($_.Name)"
    }
}

Write-Host "Total files with syntax fixes: $count"
