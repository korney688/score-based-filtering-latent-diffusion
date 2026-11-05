[CmdletBinding()]
param(
    [ValidateSet('Audit','Preflight','DryRun','Start','Resume','Status','Logs')]
    [string]$Action = 'Preflight',
    [ValidateSet(43,44)][int]$Seed = 43,
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing' }
$driverPath = Join-Path $PSScriptRoot 'run_cifar10_noise_seed_replicates.py'
if ($Action -eq 'Logs') {
    $logPath = Join-Path $projectPath "experiments/cifar10/cifar10_001_final_20260901_seed$Seed/00_protocol/logs/cifar10_seed$Seed.log"
    if (-not (Test-Path -LiteralPath $logPath)) { Write-Output "Seed $Seed has not started yet"; return }
    if ($Follow) { Get-Content -LiteralPath $logPath -Encoding UTF8 -Tail 40 -Wait }
    else { Get-Content -LiteralPath $logPath -Encoding UTF8 -Tail 40 }
    return
}
if ($Action -in @('Start','Resume')) {
    $logsPath = Join-Path $projectPath 'logs'
    New-Item -ItemType Directory -Path $logsPath -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdoutPath = Join-Path $logsPath "cifar10_noise_replicates_${stamp}.log"
    $stderrPath = Join-Path $logsPath "cifar10_noise_replicates_${stamp}.stderr.log"
    $arguments = @('-u', ('"' + $driverPath + '"'), 'run')
    if ($Action -eq 'Resume') { $arguments += '--resume' }
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Write-Output "CIFAR launcher PID=$($process.Id) stdout=$stdoutPath stderr=$stderrPath"
    return
}
$mode = switch ($Action) { 'DryRun' {'dry-run'} default {$Action.ToLowerInvariant()} }
& $pythonPath -u $driverPath $mode
if ($LASTEXITCODE -ne 0) { throw "CIFAR launcher failed (exit=$LASTEXITCODE)" }
