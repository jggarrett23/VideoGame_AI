param(
    [string]$Model           = "dueling_cnn",
    [int]$Episodes           = 10,
    [float]$Lr               = 1e-3,
    [int]$BatchSize          = 32,
    [int]$NumEnvs            = 2,
    [switch]$LoadCheckpoint,
    [switch]$SaveData
)

$startTime      = Get-Date
$venvActivate   = "D:\VideoGame_AI\DBZ\venv\Scripts\activate.ps1"
$pythonScript   = "D:\VideoGame_AI\DBZ\train_agent.py"

# Activate the virtual environment
Write-Host "Activating the virtual environment..."
& $venvActivate

# Launch PCSX2 and wait for it to reach a playable state
Write-Host "Launching game via MCP server..."
python "$PSScriptRoot\launch_game.py" --num-envs $NumEnvs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Game launch failed. Aborting."
    exit 1
}

# Build argument list
$trainArgs = @("$pythonScript", "--model", $Model, "--episodes", $Episodes, "--lr", $Lr, "--batch-size", $BatchSize, "--num-envs", $NumEnvs)
if ($LoadCheckpoint) { $trainArgs += "--load-checkpoint" }
if ($SaveData)       { $trainArgs += "--save-data" }

Write-Host "Running: python $($trainArgs -join ' ')"

try {
    python @trainArgs
    Write-Host "Python script executed successfully."
} catch {
    Write-Host "An error occurred while executing the Python script."
} finally {
    $endTime     = Get-Date
    $elapsedTime = New-TimeSpan -Start $startTime -End $endTime
    Write-Output "Elapsed time: $($elapsedTime.TotalHours) hours"

    # wait 5 minutes before sleeping
    #Start-Sleep -Seconds 300

    Write-Host "Putting the computer to sleep..."
    #rundll32.exe powrprof.dll,SetSuspendState Sleep
}
