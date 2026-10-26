# -*- coding: utf-8 -*-
"""render.py — 用 PowerPoint COM 把 pptx 导出为 PNG 以便逐页自检。
用法: python render.py <pptx> <outdir> [width] [height]
"""
import os, sys, subprocess

ps = r'''
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
'''

def main():
    src, out = sys.argv[1], sys.argv[2]
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 720
    src = os.path.abspath(src); out = os.path.abspath(out)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_render.ps1")
    with open(script, "w", encoding="utf-8") as f:
        f.write(ps)
    r = subprocess.run(["pwsh", "-NoProfile", "-File", script, src, out, str(w), str(h)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        print("STDERR:", r.stderr.strip()[:2000])
        sys.exit(r.returncode)
    files = sorted(os.listdir(out))
    print("PNG count:", len([f for f in files if f.upper().endswith(".PNG")]))

if __name__ == "__main__":
    main()
