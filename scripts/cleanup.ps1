##############################################################
# StocksTool Repository Cleanup Script
# Created: September 18, 2025
# 
# This script cleans up temporary files, build artifacts, and
# cache files while preserving the core functionality.
##############################################################

# Display header
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "       StocksTool Repository Cleanup          " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Define directories and files to clean (relative to parent directory)
$dirsToRemove = @(
    "..\\.venv",            # Python virtual environment
    "..\\build",            # Build directory
    "..\\dist",             # Distribution directory
    "..\\_pycache__",      # Python cache
    "..\\**\\__pycache__"    # Python cache in subdirectories
)

$filesToRemove = @(
    "..\*.pyc",            # Compiled Python files
    "..\*.pyo",            # Optimized Python files
    "..\*.pyd",            # Python DLL files
    "..\*.so",             # Shared object files
    "..\*.exe",            # Executable files (except in dist directory)
    "..\*.spec",           # PyInstaller spec files
    "..\*.log",            # Log files
    "..\*.tmp",            # Temporary files
    "..\*~",               # Backup files
    "..\*.bak"             # Backup files
)

# Check if we're in the correct directory (should be in scripts folder)
$expectedFiles = @("..\src\early_warning_system.py", "..\README.md")
$foundExpectedFiles = $true

foreach ($file in $expectedFiles) {
    if (-not (Test-Path $file)) {
        $foundExpectedFiles = $false
        break
    }
}

if (-not $foundExpectedFiles) {
    Write-Host "Error: This script must be run from the scripts directory!" -ForegroundColor Red
    Write-Host "Expected files not found: $($expectedFiles -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Function to calculate size
function Get-DirectorySize {
    param (
        [string]$Path
    )
    
    $size = 0
    if (Test-Path $Path) {
        $size = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    }
    return $size
}

# Calculate initial size (from parent directory)
$initialSize = Get-DirectorySize -Path ".."
$initialSizeMB = [math]::Round($initialSize / 1MB, 2)

Write-Host "Initial repository size: $initialSizeMB MB" -ForegroundColor Yellow
Write-Host ""

# Ask for confirmation
Write-Host "This script will remove the following:" -ForegroundColor Yellow
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        $dirSize = Get-DirectorySize -Path $dir
        $dirSizeMB = [math]::Round($dirSize / 1MB, 2)
        Write-Host "- $dir ($dirSizeMB MB)" -ForegroundColor Gray
    }
}

$filesToRemoveExpanded = @()
foreach ($pattern in $filesToRemove) {
    $matchingFiles = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    if ($matchingFiles.Count -gt 0) {
        $filesToRemoveExpanded += $matchingFiles
        $totalSize = ($matchingFiles | Measure-Object -Property Length -Sum).Sum
        $totalSizeMB = [math]::Round($totalSize / 1MB, 2)
        Write-Host "- $pattern ($totalSizeMB MB, $($matchingFiles.Count) files)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Proceeding with cleanup..." -ForegroundColor Green

# Perform cleanup
Write-Host ""
Write-Host "Cleaning up repository..." -ForegroundColor Green

# Remove directories
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Removing directory: $dir" -ForegroundColor Cyan
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
        if (-not $?) {
            Write-Host "  Warning: Could not completely remove $dir" -ForegroundColor Yellow
        }
    }
}

# Remove files
foreach ($pattern in $filesToRemove) {
    Write-Host "Removing files matching: $pattern" -ForegroundColor Cyan
    $filesToDelete = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $filesToDelete) {
        # Skip files in the dist directory
        if ($file.FullName -like "*\dist\*") {
            continue
        }
        Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
        if ($?) {
            Write-Host "  Removed: $($file.FullName)" -ForegroundColor Gray
        } else {
            Write-Host "  Warning: Could not remove $($file.FullName)" -ForegroundColor Yellow
        }
    }
}

# Calculate final size
$finalSize = Get-DirectorySize -Path ".."
$finalSizeMB = [math]::Round($finalSize / 1MB, 2)
$savedSizeMB = [math]::Round(($initialSize - $finalSize) / 1MB, 2)
$percentSaved = [math]::Round((($initialSize - $finalSize) / $initialSize) * 100, 1)

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "              Cleanup Summary                 " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Initial Size: $initialSizeMB MB" -ForegroundColor White
Write-Host "Final Size: $finalSizeMB MB" -ForegroundColor White
Write-Host "Space Saved: $savedSizeMB MB ($percentSaved%)" -ForegroundColor Green
Write-Host ""

Write-Host "Repository cleaned successfully!" -ForegroundColor Green