[CmdletBinding()]
param(
    [ValidateSet('Build','Gpu','Preflight','DryRun','Start','Resume','Status','Logs')]
    [string]$Action = 'Preflight',
    [string]$Image = 'research-imagenet100:latest',
    [string]$Gpu = $env:RESEARCH_GPU,
    [string[]]$AllowedGpus = @(),
    [string]$ContainerName = 'research-imagenet100-seeds43-44',
    [switch]$Follow,
    [string]$DockerHost = $env:DOCKER_HOST,
    [string]$CaCert = $env:RESEARCH_DOCKER_CA,
    [string]$ClientCert = $env:RESEARCH_DOCKER_CERT,
    [string]$ClientKey = $env:RESEARCH_DOCKER_KEY,
    [string]$BaseImage = $env:RESEARCH_BASE_IMAGE
)
$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $DockerHost) { throw 'Pass -DockerHost or set DOCKER_HOST explicitly.' }
$dockerArgs = @('--tlsverify', "--tlscacert=$CaCert", "--tlscert=$ClientCert", "--tlskey=$ClientKey", "-H=$DockerHost")
foreach ($credentialPath in @($CaCert, $ClientCert, $ClientKey)) {
    if (-not $credentialPath -or -not (Test-Path -LiteralPath $credentialPath -PathType Leaf)) { throw 'Pass existing CA, client certificate and private key paths.' }
}
if (-not $AllowedGpus -and $env:RESEARCH_ALLOWED_GPUS) { $AllowedGpus = @($env:RESEARCH_ALLOWED_GPUS.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
if ($Action -notin @('Build','Logs','Status')) {
    if (-not $Gpu -or -not $AllowedGpus -or $Gpu -notin $AllowedGpus -or $Gpu -eq 'all') { throw 'Pass one GPU and an explicit allowed GPU list.' }
}
function Invoke-RemoteDocker {
    param([string[]]$Arguments, [string]$InputTar = '')
    # Windows PowerShell represents native stderr (including Docker warnings)
    # as ErrorRecords when the caller redirects output; trust the exit code.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($InputTar) {
            & $pythonPath (Join-Path $PSScriptRoot 'make_noise_image_context.py') --send $InputTar @dockerArgs @Arguments
        } else {
            & docker @dockerArgs @Arguments
        }
        $dockerExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($dockerExit -ne 0) { throw "Remote Docker failed (exit=$dockerExit)" }
}
if ($Action -eq 'Build') {
    if (-not $BaseImage) { throw 'Pass -BaseImage or set RESEARCH_BASE_IMAGE.' }
    $baseId = & docker @dockerArgs image inspect $BaseImage --format '{{.Id}}'
    if ($LASTEXITCODE -ne 0) { throw 'Verified base image is missing' }
    $baseTag = 'research-imagenet100-base:' + $baseId.Replace('sha256:','').Substring(0,12)
    Invoke-RemoteDocker -Arguments @('tag',$baseId,$baseTag)
    $pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
    $archivePath = Join-Path $projectPath 'tmp/imagenet100-noise-image-context.tar'
    $sourceHash = & $pythonPath (Join-Path $PSScriptRoot 'make_noise_image_context.py') $archivePath
    if ($LASTEXITCODE -ne 0) { throw 'Context creation failed' }
    $revision = & git -C $projectPath rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot determine git revision' }
    # Legacy builder supports explicit CPU/RAM limits for every RUN step.
    $previousBuildkit = $env:DOCKER_BUILDKIT
    try {
        $env:DOCKER_BUILDKIT = '0'
        Invoke-RemoteDocker -InputTar $archivePath -Arguments @('build','--pull=false','--memory=8g','--memory-swap=8g','--cpu-period=100000','--cpu-quota=200000',
            '--build-arg',"BASE_IMAGE=$baseTag",'--build-arg',"GIT_COMMIT=$revision",'--build-arg',"SOURCE_SHA256=$sourceHash",
            '-f','Dockerfile.noise-replicates','-t',$Image,'-')
    } finally { $env:DOCKER_BUILDKIT = $previousBuildkit }
    Invoke-RemoteDocker -Arguments @('image','inspect',$Image,'--format','{{.Id}} {{json .Config.Labels}}')
    return
}
if ($Action -eq 'Gpu') {
    Invoke-RemoteDocker -Arguments @('run','--rm','--pull=never','--read-only','--network=none','--cpus=2','--memory=2g','--memory-swap=2g',
        '--runtime=nvidia','-e',"NVIDIA_VISIBLE_DEVICES=$Gpu",'-e','NVIDIA_DRIVER_CAPABILITIES=utility',
        'nvidia/cuda:12.4.1-base-ubuntu22.04','nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv')
    return
}
if ($Action -eq 'Logs') {
    if ($Follow) {
        $familyPath = '/workspace/experiments/imagenet100_image_space_variable_awgn_v1'
        $replicateLogs = @(43,44 | ForEach-Object { "$familyPath/imagenet100_image_space_variable_awgn_001_seed$_/00_protocol/logs/imagenet100_seed$_.log" })
        Invoke-RemoteDocker -Arguments (@('exec',$ContainerName,'tail','-n','50','-F') + $replicateLogs)
        return
    }
    Invoke-RemoteDocker -Arguments @('logs','--tail','80',$ContainerName)
    $Action = 'Status'
}
if ($Action -eq 'Status') {
    # Container process state is authoritative; PID numbers in persistent files
    # belong to the training container's PID namespace, not this helper.
    & docker @dockerArgs ps -a --filter "name=^/$ContainerName`$" --format '{{.Names}} {{.Status}}'
}
$isTraining = $Action -in @('Start','Resume')
$runArgs = @('run','--pull=never','--runtime=nvidia','--cpus=2','--memory=8g','--memory-swap=8g',
    '--shm-size=2g','-e',"NVIDIA_VISIBLE_DEVICES=$Gpu",'-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
    '-e','PROJECT_ROOT=/workspace','-e','DATA_ROOT=/workspace/data/imagenet100','-e','OUTPUT_BASE=/workspace/experiments',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
    '-v','imagenet100_data:/workspace/data:ro')
if ($isTraining) {
    Invoke-RemoteDocker -Arguments @('image','inspect',$Image,'--format','{{.Id}}')
    $existing = & docker @dockerArgs ps -a --filter "name=^/$ContainerName`$" --format '{{.ID}} {{.State}}'
    if ($existing) { throw "Container already exists: $existing. Inspect logs/state; preserve it and choose a new -ContainerName for resume." }
    $runArgs = @('run','-d','--init','--stop-timeout=60','--name',$ContainerName,'--pull=never','--runtime=nvidia','--cpus=12','--memory=64g','--memory-swap=64g',
        '--shm-size=2g','-e',"NVIDIA_VISIBLE_DEVICES=$Gpu",'-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
        '-e','PROJECT_ROOT=/workspace','-e','DATA_ROOT=/workspace/data/imagenet100','-e','OUTPUT_BASE=/workspace/experiments',
        '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
        '-v','imagenet100_data:/workspace/data:ro')
    foreach ($name in @('checkpoints','experiments','outputs','logs')) { $runArgs += @('-v',"imagenet100_${name}:/workspace/$name") }
} else {
    $runArgs += @('--rm','--read-only','--network=none','--tmpfs','/tmp:rw,size=512m')
    $expMode = if ($Action -eq 'Preflight') { 'rw' } else { 'ro' }
    $runArgs += @('-v',"imagenet100_experiments:/workspace/experiments:$expMode")
}
# Nested read-only mount prevents even accidental writes into the existing seed42.
$protectedRel = 'imagenet100_image_space_variable_awgn_v1/imagenet100_image_space_variable_awgn_001'
$runArgs += @('--mount',"type=volume,source=imagenet100_experiments,target=/workspace/experiments/$protectedRel,volume-subpath=$protectedRel,readonly")
$mode = switch ($Action) { 'Preflight' {'preflight'} 'DryRun' {'dry-run'} 'Status' {'status'} default {'run'} }
$resumeValue = if ($Action -eq 'Resume') { '1' } else { '0' }
if ($Action -eq 'Status') { $runArgs += @('-e','STATUS_EXTERNAL_CONTAINER=1') }
$runArgs += @('-e',"RESUME=$resumeValue",$Image,'bash','scripts/remote/run_imagenet100_noise_seed_replicates.sh',$mode)
Invoke-RemoteDocker -Arguments $runArgs
