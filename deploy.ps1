# --- SETTINGS (FILL THIS) ---
$ServerUser = "talipzhb"
$ServerIP   = "10.15.159.149"                 # <--- ENTER YOUR SERVER IP HERE
$RemotePath = "/home/talipzhb/prod_rag"  
# ----------------------------

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Building Docker Image (linux/amd64)..." -ForegroundColor Cyan
# Make sure your Dockerfile has the '--trusted-host' and model download steps!
docker build --platform linux/amd64 -t bank-rag-app:latest .

Write-Host "[2/5] Saving Image to file..." -ForegroundColor Cyan
docker save -o app.tar bank-rag-app:latest

Write-Host "[3/5] Updating Configs (docker-compose & env)..." -ForegroundColor Cyan
# Copy configs to server
scp docker-compose.yml .env "$($ServerUser)@$($ServerIP):$($RemotePath)/"

Write-Host "[4/5] Uploading Image to server..." -ForegroundColor Cyan
scp app.tar "$($ServerUser)@$($ServerIP):$($RemotePath)/"

Write-Host "[5/5] Restarting Service on server..." -ForegroundColor Cyan
# Load image and restart
ssh "$($ServerUser)@$($ServerIP)" "cd $RemotePath && sudo docker load -i app.tar && sudo docker compose up -d --force-recreate app && sudo rm app.tar"

Write-Host "DONE! Service updated successfully." -ForegroundColor Green