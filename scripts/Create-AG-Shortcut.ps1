<#
.SYNOPSIS
    Creates a double-click "Apple-Gorilla" shortcut that launches the AG web app.

.DESCRIPTION
    Run this once. It finds AG.bat in the repo, then drops an "Apple-Gorilla.lnk"
    shortcut on your Desktop (and, with -StartMenu, in the Start Menu too). The
    shortcut runs AG.bat, which starts the web app and opens your browser.

    Re-running it just refreshes the shortcut, so it is safe to run again after
    moving the repo.

.PARAMETER StartMenu
    Also create a Start Menu entry (searchable from the Start button).

.PARAMETER IconPath
    Optional path to a .ico file to use as the shortcut icon. If omitted, a stock
    Windows application icon is used.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\Create-AG-Shortcut.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\Create-AG-Shortcut.ps1 -StartMenu
#>
[CmdletBinding()]
param(
    [switch]$StartMenu,
    [string]$IconPath
)

$ErrorActionPreference = 'Stop'

# --- Locate the repo root and AG.bat ----------------------------------------
# This script lives in <repo>\scripts, so the repo root is its parent folder.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir
$agBat     = Join-Path $repoRoot 'AG.bat'

if (-not (Test-Path $agBat)) {
    # Fallback: search a couple of common spots.
    $candidates = @(
        (Join-Path $scriptDir 'AG.bat'),
        (Join-Path (Get-Location) 'AG.bat')
    )
    $agBat = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $agBat) {
        throw "Could not find AG.bat. Run this script from inside the Apple-Gorilla repo (it normally lives in <repo>\scripts)."
    }
    $repoRoot = Split-Path -Parent $agBat
}

Write-Host "Apple-Gorilla repo : $repoRoot"
Write-Host "Launcher           : $agBat"

# --- Choose an icon ---------------------------------------------------------
# Prefer a user-supplied .ico, then a bundled one, else a stock Windows icon.
if (-not $IconPath) {
    $bundled = Join-Path $repoRoot 'ag.ico'
    if (Test-Path $bundled) {
        $IconPath = $bundled
    } else {
        # A generic "application" icon that exists on all modern Windows.
        $IconPath = "$env:SystemRoot\System32\imageres.dll,109"
    }
}
Write-Host "Icon               : $IconPath"

# --- Helper to write one shortcut -------------------------------------------
function New-AGShortcut([string]$LinkPath) {
    $shell    = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath       = $agBat
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description       = 'Apple-Gorilla - self-improving AI (opens the web app in your browser)'
    $shortcut.WindowStyle       = 1           # normal window
    $shortcut.IconLocation      = $IconPath
    $shortcut.Save()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
    Write-Host "Created shortcut   : $LinkPath" -ForegroundColor Green
}

# --- Desktop shortcut (always) ----------------------------------------------
$desktop  = [Environment]::GetFolderPath('Desktop')
New-AGShortcut (Join-Path $desktop 'Apple-Gorilla.lnk')

# --- Start Menu shortcut (optional) -----------------------------------------
if ($StartMenu) {
    $programs = [Environment]::GetFolderPath('Programs')
    New-AGShortcut (Join-Path $programs 'Apple-Gorilla.lnk')
}

Write-Host ""
Write-Host "Done. Double-click 'Apple-Gorilla' on your Desktop to start AG." -ForegroundColor Cyan
