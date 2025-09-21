# PowerShell script to fix all new-style typing syntax
$ErrorActionPreference = "Continue"
$count = 0

Write-Host "Fixing all new-style typing syntax..."

function Fix-NewStyleTyping {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    
    $changed = $false
    $originalContent = $content
    
    # Fix new-style typing to old-style for compatibility
    if ($content -match '\bdict\[|\blist\[|\bset\[|\btuple\[') {
        # Fix dict[...] -> Dict[...]
        $content = $content -replace '\bdict\[', 'Dict['
        
        # Fix list[...] -> List[...]
        $content = $content -replace '\blist\[', 'List['
        
        # Fix set[...] -> Set[...]
        $content = $content -replace '\bset\[', 'Set['
        
        # Fix tuple[...] -> Tuple[...]
        $content = $content -replace '\btuple\[', 'Tuple['
        
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
    if (Fix-NewStyleTyping $_.FullName) {
        $count++
        Write-Host "Fixed typing in: $($_.Name)"
    }
}

Write-Host "Total files with typing fixes: $count"
