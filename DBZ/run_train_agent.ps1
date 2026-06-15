# Define the Python script to run
$startTime = Get-Date
$venvActivate = "D:\VideoGame_AI\DBZ\venv\Scripts\activate.ps1"
$pythonScript = "D:\VideoGame_AI\DBZ\train_agent.py"

# Activate the virtual environment
Write-Host "Activating the virtual environment..."
& $venvActivate

# Try to execute the Python script
try {
    # Run the Python script
    python $pythonScript
    Write-Host "Python script executed successfully."
} catch {
    # Handle any errors
    Write-Host "An error occurred while executing the Python script."
} finally {
    $endTime = Get-Date
    $elapsedTime = New-TimeSpan -Start $startTime -End $endTime
    Write-Output "Elapsed time: $($elapsedTime.TotalHours) hours"

    # wait 5 minutes
    Start-Sleep -Seconds 300

    # Put the computer to sleep regardless of success or failure
    Write-Host "Putting the computer to sleep..."
    rundll32.exe powrprof.dll,SetSuspendState Sleep
}
