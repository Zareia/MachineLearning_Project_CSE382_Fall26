# How to Contribute (Beginner-Friendly Guide)

New to Git and GitHub? Start here. Follow the steps in order and copy the commands exactly.

> **Why this matters:** each person is graded on **their own commits** on GitHub. If your work is pushed from someone else's account, it doesn't count for you. So everyone needs to set this up themselves.

---

## Two words you need to know

- **Git** is a program on your computer that saves "snapshots" of the project. Each snapshot is called a **commit**.
- **GitHub** is the website that stores the shared copy of the project, so the whole team can see each other's commits.

You work on your laptop, then **push** (upload) your commits to GitHub, and **pull** (download) everyone else's.

---

## Part 1: One-time setup (about 20 minutes)

Do this once, on your own laptop.

### Step 1: Install the tools

1. **Git:** download from https://git-scm.com/downloads and install it. The default options are fine; just keep clicking *Next*.
2. **Python 3.12:** download from https://www.python.org/downloads/release/python-3120/ (scroll to *Files* and pick the Windows installer).
   During install, **tick "Add Python to PATH"**.
   ⚠️ Use 3.12, not the newest Python. Some of our libraries don't work on the newest version yet.
3. **A code editor:** [VS Code](https://code.visualstudio.com/) is recommended.

### Step 2: Accept the GitHub invite

1. Make a GitHub account at https://github.com if you don't have one.
2. Send your **GitHub username** to Zareia (the repo owner).
3. Check your email for an invite to `MachineLearning_Project_CSE382_Fall26` and click **Accept**.
   You can't upload anything until you accept it.

### Step 3: Open a terminal

- **Windows:** open the Start menu, type **Git Bash**, and open it.
  (You can also use the terminal inside VS Code: *Terminal → New Terminal*.)
- **Mac:** open the **Terminal** app.

Every command below is typed into this terminal. Press **Enter** after each one.

### Step 4: Tell Git who you are

Replace the name and email with yours. **The email must be the same one you used for your GitHub account**, or your commits won't show up as yours.

```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

### Step 5: Download the project (called "cloning")

First, move to the folder where you want the project, for example your Documents folder:

```bash
cd ~/Documents
```

Then download it:

```bash
git clone https://github.com/Zareia/MachineLearning_Project_CSE382_Fall26.git
cd MachineLearning_Project_CSE382_Fall26
```

You now have a folder called `MachineLearning_Project_CSE382_Fall26` with all the code inside.

### Step 6: Set up Python for the project

We use a **virtual environment** (`.venv`), a private box of Python libraries just for this project.

**Windows (Git Bash):**
```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

**Mac:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal line. That means it's active.

Install the libraries (takes a few minutes):
```bash
pip install -r requirements.txt
```

Check that everything works:
```bash
pytest
```
If you see **`1 passed`**, your setup is done. 🎉

### Step 7: Your first contribution

Open `README.md`, find the **Team** table, and fill in your name, student ID and GitHub username. Then follow **Part 2** below to upload it. It's a good practice run.

---

## Part 2: Every time you work on the project

Follow this same routine every time. It looks long the first time, but it becomes muscle memory.

### ① Open the project and activate Python

```bash
cd ~/Documents/MachineLearning_Project_CSE382_Fall26
source .venv/Scripts/activate      # Mac: source .venv/bin/activate
```

### ② Get the latest code from the team

```bash
git checkout main
git pull
```

**Always do this before starting.** It downloads what your teammates pushed.

### ③ Make a branch for your task

A **branch** is your own separate copy to work on, so you don't break anyone else's work. Give it a short name with dashes describing what you're doing:

```bash
git checkout -b eda-missing-values
```

(Use your own name for the branch, e.g. `regression-baseline` or `add-my-name-readme`.)

### ④ Do your work

Edit files in VS Code as normal. Put code in the right folder:

| What you're working on | Folder |
|---|---|
| Loading data | `src/data/` |
| Cleaning, encoding, scaling | `src/preprocessing/` |
| Choosing / creating features | `src/features/` |
| ADR prediction | `src/regression/` |
| Cancellation prediction | `src/classification/` |
| Clustering | `src/clustering/` |
| PCA / LDA | `src/dimensionality/` |
| Hyperparameter tuning | `src/tuning/` |
| Metrics, tables, plots | `src/evaluation/` |
| Experiments and scratch work | `notebooks/` (not the final solution!) |

### ⑤ Save a snapshot (commit)

See what you changed:
```bash
git status
```

Add your changes and commit them with a short message saying **what you did**:
```bash
git add .
git commit -m "Add missing-value summary for EDA"
```

💡 **Commit small and often**: every time you finish a small piece, not once at the end. The graders look at your commit history over time.

Good messages: `Add KNN regression baseline`, `Fix Children column NaN handling`
Bad messages: `update`, `stuff`, `asdf`

### ⑥ Upload your branch (push)

The first time you push a new branch:
```bash
git push -u origin eda-missing-values
```
(Use your branch name.) After that, just `git push` is enough on the same branch.

The **first time ever**, a browser window may pop up asking you to log in to GitHub. Log in and approve it.

### ⑦ Ask to merge it (Pull Request)

1. Go to https://github.com/Zareia/MachineLearning_Project_CSE382_Fall26
2. You'll see a yellow banner saying your branch had recent pushes. Click **Compare & pull request**.
3. Write a short title and description of what you did, then click **Create pull request**.
4. Tell the team in the group chat. A teammate takes a quick look and clicks **Merge pull request**.

Your work is now part of `main`. 🎉 For your next task, go back to step ② and start a **new** branch.

---

## Quick cheat sheet

```bash
git checkout main && git pull         # get the latest code
git checkout -b my-task-name          # start a new task
git status                            # what did I change?
git add .                             # stage everything
git commit -m "Describe what you did" # save a snapshot
git push -u origin my-task-name       # upload (first time on this branch)
git push                              # upload (after that)
```

---

## Team rules

- ✅ **Pull before you start working**, every time.
- ✅ **Work on a branch**, never directly on `main`.
- ✅ **Commit from your own laptop and your own account.**
- ✅ Use settings from `config.yaml` (random seed, file paths) instead of typing numbers into your code.
- ✅ Load the data with `load_raw()` from `src/data/load.py`.
- ❌ **Don't commit trained models** or big generated files. They go in `artifacts/`, which Git ignores automatically.
- ❌ **Don't look at the test set** while choosing models or settings. Use the validation set or cross-validation. The test set is only for the final results.
- ❌ Don't commit the `.venv` folder. It's ignored already, so just don't force it.

---

## Help! Something went wrong

**`(.venv)` isn't showing / `ModuleNotFoundError: No module named 'pandas'`**
You forgot to activate the environment. Run the `source .venv/...` command from step ①.

**`py: command not found` or `python3.12: command not found`**
Python 3.12 isn't installed, or "Add to PATH" wasn't ticked. Reinstall it and tick the box.

**`Permission denied` / `403` when pushing**
You haven't accepted the GitHub invite (Part 1, Step 2), or you're logged in to a different GitHub account.

**`git push` says `rejected ... fetch first`**
Someone pushed before you. Run:
```bash
git pull
git push
```

**`CONFLICT` / "merge conflict"**
You and a teammate changed the same lines. Don't panic and don't delete anything. Open the file in VS Code: it highlights the conflict and gives you buttons (*Accept Current*, *Accept Incoming*, *Accept Both*). Pick the right version, save, then:
```bash
git add .
git commit -m "Resolve merge conflict"
```
If you're unsure, ask in the group chat **before** pushing.

**A text editor opened and I'm stuck (it says VIM)**
You ran `git commit` without `-m "message"`. Type `:q!` and press Enter to escape, then commit again with `-m`.

**I committed to `main` by mistake (and haven't pushed)**
Ask for help in the group chat. It's easy to fix, so don't try random commands.

Still stuck? Copy the **exact** error message into the group chat. That's the fastest way to get help.
