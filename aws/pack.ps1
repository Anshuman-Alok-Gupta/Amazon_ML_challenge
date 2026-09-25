# Run on your laptop (PowerShell) from D:\Amazon_ml. Creates upload.tgz with the code and the
# challenge data, excluding the venv, caches, artifacts and outputs (~0.6 GB compressed).
#     powershell -ExecutionPolicy Bypass -File aws\pack.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
tar -czf upload.tgz `
    --exclude=".venv" --exclude="__pycache__" --exclude=".pytest_cache" --exclude=".DS_Store" `
    --exclude="code/business_entity_resolution/cache" `
    --exclude="code/business_entity_resolution/artifacts" `
    code student_resource aws Documentation_template.md
Get-Item upload.tgz | Select-Object Name, @{n='MB';e={[int]($_.Length/1MB)}}
