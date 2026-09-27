# Run on your laptop (PowerShell) from D:\Amazon_ml. Creates upload.tgz with the code and the
# challenge data, excluding the venv, caches, artifacts and outputs (~0.6 GB compressed).
#     powershell -ExecutionPolicy Bypass -File aws\pack.ps1
# With -CodeOnly it creates code.tgz (source, requirements, aws scripts; a few KB) for an instance
# that already has the data:  tar -xzf ~/code.tgz -C ~/Amazon_ml
param([switch]$CodeOnly)
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($CodeOnly) {
    tar -czf code.tgz --exclude="__pycache__" `
        code/business_entity_resolution/src code/business_entity_resolution/requirements.txt `
        code/business_entity_resolution/README.md aws/run_on_ec2.sh aws/AWS_GUIDE.md
    Get-Item code.tgz | Select-Object Name, Length
    return
}
tar -czf upload.tgz `
    --exclude=".venv" --exclude="__pycache__" --exclude=".pytest_cache" --exclude=".DS_Store" `
    --exclude="code/business_entity_resolution/cache" `
    --exclude="code/business_entity_resolution/artifacts" `
    code student_resource aws Documentation_template.md
Get-Item upload.tgz | Select-Object Name, @{n='MB';e={[int]($_.Length/1MB)}}
