# Push Instructions for Anonymous Submission

This repository is already anonymized and committed locally.

## Recommended Submission Link

For double-blind paper submission, the common workflow is:

1. Push this repository to a neutral GitHub repository.
2. Open https://anonymous.4open.science/anonymize
3. Paste the GitHub repository URL.
4. Add any author-specific replacement terms requested by the interface.
5. Use the generated link, usually in the form `https://anonymous.4open.science/r/...`, in the paper.

Anonymous GitHub / 4open.science is a proxy mirror, not a Git remote. It cannot be used directly as `git push` destination.

## Push to an Empty GitHub Repository

Create an empty repository on GitHub first. For double-blind review, use a neutral account or private temporary account if allowed by the venue. Do not add README, license, or `.gitignore` on GitHub because this local repository already contains them.

Then run:

```powershell
cd <ANONYMOUS_RELEASE_DIR>
.\push_to_github.ps1 -RemoteUrl "https://github.com/<anonymous-account>/<repo-name>.git"
```

After pushing, create the anonymous mirror at:

```text
https://anonymous.4open.science/anonymize
```

Recommended anonymization terms should include any author name, username, institution, personal machine path, or private project name that may remain in the repository. Example placeholders:

```text
<author-name>
<username>
<institution>
<local-user-path>
<private-project-name>
```

The local anonymization report currently shows no sensitive string matches.
