param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if ($RemoteUrl -notmatch "^https://github\.com/[^/]+/[^/]+(\.git)?$") {
    throw "RemoteUrl should look like https://github.com/<account>/<repo>.git"
}

git status --short
$dirty = git status --porcelain
if ($dirty) {
    throw "Working tree is not clean. Commit or discard local changes before pushing."
}

$remotes = git remote
if ($remotes -contains "origin") {
    git remote set-url origin $RemoteUrl
}
else {
    git remote add origin $RemoteUrl
}

git branch -M $Branch
git push -u origin $Branch
