# Cheatsheet
Quick commands for Python, MuJoCo and Linux setup.
> [!NOTE] Use Python 3.12.3 for this project


# Open mujoco drag and drop
> [!NOTE]
> Modern MuJoCo no longer uses `./simulate`.
> Use `python -m mujoco.viewer` instead.

# Activate virtual environment
> cd python
> source .bach_env/bin/activate

# Git commands
<git_commands>
```

# Make new branch
> git switch -c <branch_name>

# Connect branch to main
> git push --set-upstream origin <branch_name>

# Delete remote branch
> git push origin --delete <branch_name>

# Delete local branch
> git branch -d <branch_name>

# Update branch list
> git fetch --prune

</git_commands>```