param(
    [string]$Data = "data\selected_thermodynamic_properties.csv",
    [string]$Formula,
    [string]$Composition,
    [string]$InputFile,
    [string]$Output,
    [ValidateSet("loo", "kfold")]
    [string]$Validation = "loo",
    [ValidateSet("fast", "expanded", "full")]
    [string]$ModelSet = "expanded",
    [string]$Model,
    [switch]$TrainOnly,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$argsList = @("-m", "src.cli", "--data", $Data, "--validation", $Validation, "--model-set", $ModelSet)

if ($Formula) {
    $argsList += @("--formula", $Formula)
}
if ($Composition) {
    $argsList += @("--composition", $Composition)
}
if ($InputFile) {
    $argsList += @("--input-file", $InputFile)
}
if ($Output) {
    $argsList += @("--output", $Output)
}
if ($Model) {
    $argsList += @("--model", $Model)
}
if ($TrainOnly) {
    $argsList += "--train-only"
}
if ($Json) {
    $argsList += "--json"
}

Push-Location $root
try {
    py @argsList
}
finally {
    Pop-Location
}
