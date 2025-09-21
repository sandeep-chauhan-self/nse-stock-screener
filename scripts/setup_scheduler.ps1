# ================================================================
# NSE Stock Screener - Task Scheduler Setup
# Creates automated tasks for daily and weekly analysis
# ================================================================

param(
    [string]$ProjectPath = (Get-Location).Path,
    [switch]$Remove,
    [switch]$Test
)

# Ensure running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script requires Administrator privileges" -ForegroundColor Red
    Write-Host "💡 Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🚀 NSE Stock Screener - Task Scheduler Setup" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan

$ProjectPath = Resolve-Path $ProjectPath
Write-Host "📁 Project Path: $ProjectPath" -ForegroundColor Yellow

# Task names
$DailyTaskName = "NSE_Daily_Morning_Scan"
$WeeklyTaskName = "NSE_Weekly_Deep_Analysis"

# Script paths
$DailyScript = Join-Path $ProjectPath "scripts\automated_launcher.bat"
$WeeklyScript = Join-Path $ProjectPath "scripts\weekly_analysis.bat"

if ($Remove) {
    Write-Host "🗑️ Removing existing tasks..." -ForegroundColor Yellow
    
    try {
        Unregister-ScheduledTask -TaskName $DailyTaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "✅ Removed daily task: $DailyTaskName" -ForegroundColor Green
    } catch {
        Write-Host "ℹ️ Daily task not found: $DailyTaskName" -ForegroundColor Gray
    }
    
    try {
        Unregister-ScheduledTask -TaskName $WeeklyTaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "✅ Removed weekly task: $WeeklyTaskName" -ForegroundColor Green
    } catch {
        Write-Host "ℹ️ Weekly task not found: $WeeklyTaskName" -ForegroundColor Gray
    }
    
    Write-Host "✅ Task removal completed!" -ForegroundColor Green
    exit 0
}

# Verify script files exist
if (-not (Test-Path $DailyScript)) {
    Write-Host "❌ Daily script not found: $DailyScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $WeeklyScript)) {
    Write-Host "❌ Weekly script not found: $WeeklyScript" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found automation scripts" -ForegroundColor Green

# Create Daily Morning Scan Task
Write-Host "📅 Creating daily morning scan task..." -ForegroundColor Yellow

$DailyAction = New-ScheduledTaskAction -Execute $DailyScript -WorkingDirectory $ProjectPath
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At "09:00AM"
$DailySettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$DailyPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

$DailyTask = New-ScheduledTask -Action $DailyAction -Trigger $DailyTrigger -Settings $DailySettings -Principal $DailyPrincipal -Description "NSE Stock Screener - Daily Morning Market Scan"

try {
    Register-ScheduledTask -TaskName $DailyTaskName -InputObject $DailyTask -Force | Out-Null
    Write-Host "✅ Created daily task: $DailyTaskName" -ForegroundColor Green
    Write-Host "   ⏰ Runs daily at 9:00 AM" -ForegroundColor Gray
    Write-Host "   🎯 Executes: $DailyScript" -ForegroundColor Gray
} catch {
    Write-Host "❌ Failed to create daily task: $($_.Exception.Message)" -ForegroundColor Red
}

# Create Weekly Deep Analysis Task
Write-Host "📅 Creating weekly deep analysis task..." -ForegroundColor Yellow

$WeeklyAction = New-ScheduledTaskAction -Execute $WeeklyScript -WorkingDirectory $ProjectPath
$WeeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "08:00PM"
$WeeklySettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$WeeklyPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

$WeeklyTask = New-ScheduledTask -Action $WeeklyAction -Trigger $WeeklyTrigger -Settings $WeeklySettings -Principal $WeeklyPrincipal -Description "NSE Stock Screener - Weekly Deep Market Analysis"

try {
    Register-ScheduledTask -TaskName $WeeklyTaskName -InputObject $WeeklyTask -Force | Out-Null
    Write-Host "✅ Created weekly task: $WeeklyTaskName" -ForegroundColor Green
    Write-Host "   ⏰ Runs every Sunday at 8:00 PM" -ForegroundColor Gray
    Write-Host "   🎯 Executes: $WeeklyScript" -ForegroundColor Gray
} catch {
    Write-Host "❌ Failed to create weekly task: $($_.Exception.Message)" -ForegroundColor Red
}

if ($Test) {
    Write-Host "🧪 Testing tasks..." -ForegroundColor Yellow
    
    Write-Host "🔍 Testing daily task..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $DailyTaskName
    Start-Sleep -Seconds 5
    $DailyState = (Get-ScheduledTask -TaskName $DailyTaskName).State
    Write-Host "   Status: $DailyState" -ForegroundColor Gray
    
    Write-Host "🔍 Testing weekly task..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $WeeklyTaskName  
    Start-Sleep -Seconds 5
    $WeeklyState = (Get-ScheduledTask -TaskName $WeeklyTaskName).State
    Write-Host "   Status: $WeeklyState" -ForegroundColor Gray
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🎉 Task Scheduler Setup COMPLETED!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "📋 Created Tasks:" -ForegroundColor Yellow
Write-Host "   • $DailyTaskName (Daily 9:00 AM)" -ForegroundColor White
Write-Host "   • $WeeklyTaskName (Sunday 8:00 PM)" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "🛠️ Management Commands:" -ForegroundColor Yellow
Write-Host "   • View tasks: Get-ScheduledTask | Where-Object {`$_.TaskName -like 'NSE_*'}" -ForegroundColor White
Write-Host "   • Remove tasks: .\setup_scheduler.ps1 -Remove" -ForegroundColor White
Write-Host "   • Test tasks: .\setup_scheduler.ps1 -Test" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan