param([switch]$ConfigureOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$dependencyRoot = Join-Path $projectRoot 'build/amrex-reference-deps'
$buildRoot = Join-Path $projectRoot 'build/amrex-reference-clang-codex'
$pins = @(
    @('amrex', 'https://github.com/AMReX-Codes/amrex.git', '015c14b65715b277fbc54706ebe7455b196d2254'),
    @('amrex-hydro', 'https://github.com/AMReX-Fluids/AMReX-Hydro.git', '5829a90a6cd521b378ebb7bce1390937beb49506')
)
New-Item -ItemType Directory -Force $dependencyRoot | Out-Null
foreach ($pin in $pins) {
    $sourceRoot = Join-Path $dependencyRoot $pin[0]
    if (!(Test-Path -LiteralPath $sourceRoot)) {
        git init $sourceRoot
        git -C $sourceRoot remote add origin $pin[1]
        git -C $sourceRoot fetch --depth 1 origin $pin[2]
        git -C $sourceRoot checkout --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw "Failed to fetch $($pin[0])" }
    }
    $actual = git -C $sourceRoot rev-parse HEAD
    if ($actual -ne $pin[2]) { throw "Unexpected source pin in $sourceRoot : $actual" }
}
$amrexSource = Join-Path $dependencyRoot 'amrex'
$portabilityPatch = Join-Path $PSScriptRoot 'amrex-windows-library-copy.patch'
git -C $amrexSource apply --reverse --check $portabilityPatch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $amrexSource apply $portabilityPatch
    if ($LASTEXITCODE -ne 0) { throw 'AMReX Windows library-copy patch failed' }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vsInstall 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
$clangCl = 'C:/Program Files/LLVM/bin/clang-cl.exe'
cmake -S $PSScriptRoot -B $buildRoot -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$clangCl" "-DCMAKE_CXX_COMPILER=$clangCl"
if ($LASTEXITCODE -ne 0) { throw 'Reference configure failed' }
if (!$ConfigureOnly) {
    cmake --build $buildRoot --target FluidAmrexReference FluidAmrGasTests FluidExtractionTests FluidPressure2DTests -j 8
    if ($LASTEXITCODE -ne 0) { throw 'Reference build failed' }
    Copy-Item -LiteralPath 'C:/Program Files/LLVM/bin/libomp.dll' -Destination $buildRoot
}
