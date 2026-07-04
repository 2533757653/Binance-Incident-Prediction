# Consolidate to ONE desktop shortcut. Remove old ones, create Horizon-Signal. Pure ASCII.
$desk = [Environment]::GetFolderPath('Desktop')
foreach ($old in 'Horizon-1-Start.lnk','Horizon-2-Stop.lnk','Horizon-3-Review.lnk','Horizon-Incident.lnk') {
  $p = Join-Path $desk $old
  if (Test-Path $p) { Remove-Item $p -Force; Write-Host ("removed: " + $old) }
}
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desk 'Horizon-Signal.lnk'))
$lnk.TargetPath = (Join-Path $PSScriptRoot 'horizon.bat')
$lnk.WorkingDirectory = (Split-Path $PSScriptRoot -Parent)
$lnk.IconLocation = 'shell32.dll,137'
$lnk.Description = 'Horizon 事件合约信号：双击开始，关窗即停，内置弹窗+战绩'
$lnk.Save()
Write-Host 'created: Horizon-Signal.lnk'
