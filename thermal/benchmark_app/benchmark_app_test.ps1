<#
.SYNOPSIS
    OpenVINO Multi-Device & Fixed Precision Benchmark Script
.DESCRIPTION
    Usage: use -r <count> (or -runs <count>) to set benchmark repetitions.

    Maps requested precisions to valid benchmark_app arguments:
    - CPU: F32, F16
    - GPU: F32, F16
    - NPU: F16, i8    
.PARAMETER r
    Number of benchmark repetitions. Default is 1.
    Alias: -runs
#>
param (
    [Alias("runs")]
    [int]$r = 1
)

# Configuration
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$modelCandidates = @(    
    (Join-Path $projectRoot "thermal\yolov8n_openvino_model\yolov8n.xml"),
    (Join-Path $scriptDir "yolov8n.xml")
)
$modelPath = $modelCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$outputCsv = Join-Path $scriptDir "benchmark_precision_comparison.csv"

$pythonCandidates = @()
try {
    $pythonCommands = @(Get-Command python -All -ErrorAction Stop)
    foreach ($pythonCommand in $pythonCommands) {
        if ($pythonCommand.Path) {
            $pythonCandidates += $pythonCommand.Path
        }
    }
} catch {
    throw "python executable not found in PATH."
}

$systemPythonPath = $pythonCandidates | Where-Object { $_ -notmatch "\\\.venv\\" } | Select-Object -First 1
if (-not $systemPythonPath) {
    $systemPythonPath = $pythonCandidates | Select-Object -First 1
}

$systemPythonDir = Split-Path -Parent $systemPythonPath
$benchmarkAppCandidates = @(
    (Join-Path $systemPythonDir "benchmark_app.exe"),
    (Join-Path $systemPythonDir "Scripts\benchmark_app.exe"),
    (Join-Path (Split-Path -Parent $systemPythonDir) "Scripts\benchmark_app.exe")
)
$benchmarkAppPath = $benchmarkAppCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not (Test-Path $modelPath)) {
    $candidateText = ($modelCandidates -join ", ")
    throw "Model file not found. Checked: $candidateText"
}

if (-not (Test-Path $systemPythonPath)) {
    throw "System python not found at: $systemPythonPath"
}

if (-not $benchmarkAppPath) {
    $benchmarkCandidateText = ($benchmarkAppCandidates -join ", ")
    throw "benchmark_app not found. Checked: $benchmarkCandidateText"
}

# Updated test suite based on your version's allowed -ip values
$testSuite = @(
    @{Device="CPU"; Precision="f32"},
    @{Device="CPU"; Precision="f16"},
    @{Device="GPU"; Precision="f32"},
    @{Device="GPU"; Precision="f16"},
    @{Device="NPU"; Precision="f16"},
    @{Device="NPU"; Precision="i8"}
)

$allData = @()

Write-Host ">>> Starting OpenVINO Fixed Batch Test ($r total runs) <<<" -ForegroundColor Cyan

for ($i = 1; $i -le $r; $i++) {
    Write-Host "`n==== Starting Run $i of $r ====" -ForegroundColor White -BackgroundColor Blue
    
    foreach ($config in $testSuite) {
        $dev = $config.Device
        $prec = $config.Precision
        
        Write-Host "  Testing [ $dev | $prec ]..." -NoNewline -ForegroundColor Yellow
        
        # Execute benchmark_app resolved from system Python installation.
        try {
            $output = & $benchmarkAppPath -m $modelPath -d $dev -ip $prec -t 10 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) {
                throw "benchmark_app exited with code $exitCode"
            }
        } catch {
            $outputTail = ""
            if ($output) {
                $outputTail = ($output -split "`r?`n" | Where-Object { $_ -and $_.Trim() -ne "" } | Select-Object -Last 6) -join " | "
            }
            Write-Host " -> Error: benchmark_app failed for $dev / $prec. $($_.Exception.Message)" -ForegroundColor Red
            if ($outputTail) {
                Write-Host "    Details: $outputTail" -ForegroundColor DarkRed
            }
            continue
        }
        
        # Parse Throughput and Latency
        $fps = 0
        $latency = 0
        if ($output -match 'Throughput:\s+([\d\.]+)\s+FPS') { $fps = [double]$matches[1] }
        if ($output -match 'Average:\s+([\d\.]+)\s+ms') { $latency = [double]$matches[1] }

        if ($fps -gt 0) {
            Write-Host " -> Done: $fps FPS / $latency ms" -ForegroundColor Gray
            $allData += [PSCustomObject]@{
                Run        = $i
                Device     = $dev
                Precision  = $prec
                FPS        = $fps
                Latency_ms = $latency
                Status     = "Raw"
            }
        } else {
            Write-Host " -> Error: Check if $dev supports $prec." -ForegroundColor Red
        }
    }
}

# Final Average Calculation
Write-Host "`n>>> Summarizing Results <<<" -ForegroundColor Cyan
foreach ($config in $testSuite) {
    $dev = $config.Device
    $prec = $config.Precision
    $subset = $allData | Where-Object { $_.Device -eq $dev -and $_.Precision -eq $prec -and $_.Status -eq "Raw" }

    $subsetCount = @($subset).Count
    if ($subsetCount -gt 0) {
        $avgFps = [Math]::Round(($subset | Measure-Object FPS -Average).Average, 2)
        $avgLatency = [Math]::Round(($subset | Measure-Object Latency_ms -Average).Average, 2)
        $allData += [PSCustomObject]@{
            Run        = "AVERAGE"
            Device     = $dev
            Precision  = $prec
            FPS        = $avgFps
            Latency_ms = $avgLatency
            Status     = "Summary"
        }
    }
}

$allData | Export-Csv -Path $outputCsv -NoTypeInformation -Encoding UTF8
Write-Host "`nTest Complete! CSV saved to: $outputCsv" -ForegroundColor Green
$allData | Where-Object { $_.Run -eq "AVERAGE" } | Format-Table -Property Device, Precision, FPS, Latency_ms