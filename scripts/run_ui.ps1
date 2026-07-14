param(
    [int]$Port = 8501
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Push-Location $root
try {
    py -m streamlit run app.py --server.port $Port
}
finally {
    Pop-Location
}
