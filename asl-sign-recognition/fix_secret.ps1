# fix_secret.ps1
# Rewrites the commit that contained the hardcoded Anthropic API key
# Run this from: D:\asl-sign-recognition\asl-sign-recognition

$BAD_COMMIT   = "8beeff3"
$PARENT_COMMIT = "a53d29f"
$APP_PY = "asl-sign-recognition/backend/app.py"
$SECRET_PATTERN = 'sk-ant-api03-[A-Za-z0-9_\-]+'

Write-Host "==> Step 1: Creating sequence editor script..." -ForegroundColor Cyan

# Write a tiny bat file that PowerShell will use as GIT_SEQUENCE_EDITOR
$seqEditorScript = @"
@echo off
powershell -Command "(Get-Content '%1') -replace '^pick $BAD_COMMIT', 'edit $BAD_COMMIT' | Set-Content '%1'"
"@
$seqEditorPath = "$env:TEMP\git_seq_editor.bat"
Set-Content -Path $seqEditorPath -Value $seqEditorScript
Write-Host "    Sequence editor written to: $seqEditorPath" -ForegroundColor Gray

Write-Host "==> Step 2: Starting interactive rebase (automated)..." -ForegroundColor Cyan
$env:GIT_SEQUENCE_EDITOR = $seqEditorPath
$env:GIT_EDITOR = "true"  # skip any editor for commit messages

# Run rebase - it will pause at the bad commit for us to edit
$rebaseOutput = git rebase -i $PARENT_COMMIT 2>&1
Write-Host $rebaseOutput -ForegroundColor Gray

Write-Host "==> Step 3: Scrubbing API key from app.py..." -ForegroundColor Cyan
$filePath = Join-Path (git rev-parse --show-toplevel) $APP_PY

if (Test-Path $filePath) {
    $content = Get-Content $filePath -Raw
    $cleaned = $content -replace $SECRET_PATTERN, 'os.getenv("ANTHROPIC_API_KEY")'
    Set-Content -Path $filePath -Value $cleaned -NoNewline
    Write-Host "    Key replaced in: $filePath" -ForegroundColor Green
} else {
    Write-Host "    ERROR: Could not find $filePath" -ForegroundColor Red
    exit 1
}

Write-Host "==> Step 4: Staging and amending the commit..." -ForegroundColor Cyan
git add $APP_PY
git commit --amend --no-edit

Write-Host "==> Step 5: Continuing rebase..." -ForegroundColor Cyan
git rebase --continue

Write-Host "==> Step 6: Verifying the key is gone from all history..." -ForegroundColor Cyan
$found = git log -p --all | Select-String "sk-ant-api03"
if ($found) {
    Write-Host "    WARNING: Key still found in history!" -ForegroundColor Red
    $found
} else {
    Write-Host "    SUCCESS: No secrets found in git history." -ForegroundColor Green
}

Write-Host ""
Write-Host "==> Ready to push. Run: git push --force-with-lease origin main" -ForegroundColor Yellow
Write-Host "==> IMPORTANT: Regenerate your Anthropic API key at https://console.anthropic.com" -ForegroundColor Red
