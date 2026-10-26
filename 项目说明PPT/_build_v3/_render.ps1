
param([string]$Src, [string]$Out, [int]$W, [int]$H)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Get-ChildItem -Path $Out -Filter *.PNG -ErrorAction SilentlyContinue | Remove-Item -Force
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($Src, $true, $false, $false)
$pres.Export($Out, "PNG", $W, $H)
$n = $pres.Slides.Count
$pres.Close()
$ppt.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
[GC]::Collect()
Write-Host "exported $n slides"
