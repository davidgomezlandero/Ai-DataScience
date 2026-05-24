````md
# remote-repo-guide.md

# Remote Repository Guide

This guide explains how to:

- Keep an official repository inside a GitHub organization
- Keep a personal repository synchronized with it
- Push changes to both repositories at the same time
- Show your project development publicly to recruiters

This workflow is perfect for:
- Ecole 42 projects
- Portfolio projects
- Team repositories
- Public progress tracking

---

# Final Structure

You will have:

```text
Organization Repository (official)
        ↓
Personal Repository (portfolio/back-up)
````

Both repositories will receive the same commits.

---

# 1. Create the Personal Repository

Go to GitHub and create a new repository in your personal account.

Example:

```text
Organization repo:
42-org/leaffliction

Personal repo:
your-user/leaffliction
```

IMPORTANT:

When creating the personal repository:

* DO NOT add a README
* DO NOT add a .gitignore
* DO NOT add a license

The repository must be empty.

---

# 2. Clone the Organization Repository

Clone the organization repository to your computer.

```bash
git clone git@github.com:42-org/leaffliction.git
cd leaffliction
```

Replace:

* `42-org` with your organization name
* `leaffliction` with your repository name

---

# 3. Check Current Remotes

Run:

```bash
git remote -v
```

You should see:

```text
origin  git@github.com:42-org/leaffliction.git (fetch)
origin  git@github.com:42-org/leaffliction.git (push)
```

`origin` currently points only to the organization repository.

---

# 4. Add Your Personal Repository as a Second Remote

Add the personal repository:

```bash
git remote add personal git@github.com:your-user/leaffliction.git
```

Now check remotes again:

```bash
git remote -v
```

You should see something similar to:

```text
origin    git@github.com:42-org/leaffliction.git (fetch)
origin    git@github.com:42-org/leaffliction.git (push)

personal  git@github.com:your-user/leaffliction.git (fetch)
personal  git@github.com:your-user/leaffliction.git (push)
```

---

# 5. Push to Both Repositories

Push to the organization repository:

```bash
git push origin main
```

Push to the personal repository:

```bash
git push personal main
```

Now both repositories contain the same code.

---

# 6. Daily Workflow

After coding:

## Add files

```bash
git add .
```

## Create commit

```bash
git commit -m "implemented parser"
```

## Push to organization repository

```bash
git push origin main
```

## Push to personal repository

```bash
git push personal main
```

---

# 7. OPTIONAL — Push to Both with One Command

This is the best setup if you always want both repositories synchronized automatically.

Instead of pushing twice, configure Git so `origin` pushes to BOTH repositories.

Run:

```bash
git remote set-url --add --push origin git@github.com:42-org/leaffliction.git
```

Then add the personal repository:

```bash
git remote set-url --add --push origin git@github.com:your-user/leaffliction.git
```

Now verify:

```bash
git remote -v
```

You should see:

```text
origin  git@github.com:42-org/leaffliction.git (fetch)

origin  git@github.com:42-org/leaffliction.git (push)
origin  git@github.com:your-user/leaffliction.git (push)
```

---

# 8. Push Once to Update Both Repositories

Now you only need:

```bash
git push origin main
```

Git will automatically push to:

* organization repository
* personal repository

at the same time.

---

# 9. Recommended Workflow

## Organization Repository

Use it for:

* official project
* collaboration
* pull requests
* project management

## Personal Repository

Use it for:

* portfolio
* recruiter visibility
* GitHub contribution graph
* backup

---

# 10. Verify Everything Works

Make a small test:

```bash
touch test.txt
git add .
git commit -m "test sync"
git push origin main
```

Then verify:

* the commit appears in the organization repository
* the commit appears in the personal repository

If both updated correctly, the setup is complete.

---

# 11. Useful Commands

## Show remotes

```bash
git remote -v
```

## Remove a remote

```bash
git remote remove personal
```

## Rename a remote

```bash
git remote rename personal backup
```

## Show current branch

```bash
git branch
```

---

# 12. Common Problems

## ERROR: repository not found

Check:

* repository name
* organization name
* SSH permissions
* repository visibility

---

## ERROR: permission denied (publickey)

Your SSH key is probably not configured.

Generate one:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add it to GitHub:

* GitHub Settings
* SSH and GPG keys
* New SSH key

---

## ERROR: failed to push some refs

Usually means:

* remote repository contains commits you do not have locally
* someone pushed changes before you

Fix with:

```bash
git pull origin main --rebase
```

Then push again.

---

# 13. Final Recommended Setup

Recommended configuration:

```text
Official development:
Organization repository

Public visibility:
Personal repository

Synchronization:
Single git push command
```

This gives:

* clean workflow
* professional setup
* recruiter visibility
* backup safety
* contribution tracking

---

# Example Final Workflow

```bash
git add .
git commit -m "implemented enemy AI"
git push origin main
```

Both repositories update automatically.

```
```
