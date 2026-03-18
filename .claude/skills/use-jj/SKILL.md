---
name: use-jj
description: Reference for using jj (Jujutsu) VCS correctly in this repo. Invoke when you need to do any version control operation — committing, branching, pushing, undoing, diffing, etc. Never use raw git commands in a jj repo.
---

# jj (Jujutsu) Usage Reference

## Core Mental Model

| Concept | jj | Git |
|---|---|---|
| Working state | Always a real commit (`@`) | Unstaged/staged files |
| Staging area | **None** — all changes auto-tracked | `git add` required |
| Branch | Bookmark (named pointer, optional) | Branch (required for pushes) |
| History rewrite | Change ID stays stable across rewrites | Commit hash changes = new ref |
| Safety net | Operation log (`jj op log`) | `git reflog` |

**`@`** = working copy commit. **`@-`** = parent. **`@--`** = grandparent.

---

## REPO RULES (CLAUDE.md constraints)

These operations are **banned in this repo**:
- `jj edit` — mutates history
- `jj abandon` — mutates history
- `jj squash` — mutates history

Use `jj new` + `jj describe` instead. Always move forward, never rewrite.

---

## Daily Operations

### Status & Inspection
```bash
jj st                          # status (short alias)
jj log                         # commit graph
jj log -r @                    # just the working copy commit
jj diff                        # changes in working copy
jj diff -r <id>                # changes in a specific commit
jj show <id>                   # full details of a commit
```

### Making Commits

jj tracks all file changes automatically — no `git add` needed.

```bash
# Describe current commit and start a new empty one (most common flow)
jj commit -m "your message"    # sugar for: jj describe -m "..." && jj new

# Or separately:
jj describe -m "your message"  # set message on current (@) commit
jj new                         # create new empty commit on top
```

**`jj new` creates an empty commit and makes it the working copy.** Your previous commit is now `@-`.

### Viewing History
```bash
jj log                         # graph view (default)
jj log -r 'trunk()..@'        # commits since main/trunk
jj log -r <bookmark>           # commits at a bookmark
```

### Bookmarks (= Git Branches)
```bash
jj bookmark list               # list all bookmarks
jj bookmark create <name>      # create bookmark at @
jj bookmark create <name> -r <id>  # create at specific commit
jj bookmark set <name>         # move bookmark to @
jj bookmark set <name> -r <id> # move to specific commit
jj bookmark delete <name>      # delete bookmark
```

### Remote Operations
```bash
jj git fetch                   # fetch from remote (updates remote-tracking bookmarks)
jj git fetch --remote origin   # explicit remote
jj git push                    # push all tracked bookmarks
jj git push --bookmark <name>  # push a specific bookmark
jj git push --all              # push all bookmarks
```

### Undoing Mistakes
```bash
jj undo                        # undo the last operation (SAFE — uses operation log)
jj redo                        # redo the last undo
jj op log                      # view all operations with IDs
jj op restore <op-id>          # restore repo to any past operation state
```

**Nothing is lost.** The operation log records every state. `jj undo` is always safe.

---

## Git → jj Translation Table

| You want to... | Git | jj |
|---|---|---|
| See status | `git status` | `jj st` |
| See changes | `git diff` | `jj diff` |
| Commit all changes | `git add -A && git commit -m` | `jj commit -m "..."` |
| See log | `git log --oneline` | `jj log` |
| Create branch | `git checkout -b name` | `jj new && jj bookmark create name` |
| Switch to branch | `git checkout name` | `jj new name` (new commit on top of that bookmark) |
| Push branch | `git push -u origin name` | `jj git push --bookmark name` |
| Fetch | `git fetch` | `jj git fetch` |
| Undo last commit | `git reset --soft HEAD~1` | `jj undo` |
| Hard reset | `git reset --hard` | `jj undo` or `jj op restore <id>` |
| Stash | `git stash` | Not needed — `@` IS your stash; just `jj new` to start fresh work |
| Amend | `git commit --amend` | **Not allowed here** — use `jj new` + describe new commit |
| Rebase | `git rebase <target>` | `jj rebase -d <target>` |
| See a file at a commit | `git show <id>:path` | `jj file show -r <id> path` |
| Blame | `git blame` | `jj file annotate` |

---

## Revset Syntax (selecting commits)

```
@              working copy
@-             parent of working copy
@--            grandparent
<id>-          parent of <id>
trunk()        main/master branch tip
<name>         bookmark named <name>
<prefix>       unique short prefix of any change ID or commit ID
```

Examples:
```bash
jj log -r '@-'                  # show parent
jj diff -r 'abc12'              # diff by short ID
jj show trunk()                 # show tip of main
jj log -r 'trunk()..@'         # all commits since trunk
```

---

## Conflict Resolution

jj records conflicts in commits (they don't block you from continuing work).

```bash
jj rebase -d <target>          # rebase; conflicts are recorded, not fatal
jj log                         # conflicted commits are marked with "conflict"
# Edit the conflicted files to resolve them
jj new                         # move to next commit once resolved
```

---

## NEVER DO (in any jj repo)

```bash
# NEVER — these corrupt jj's state:
git add
git commit
git reset
git checkout
git rebase
git merge
git push / git push --force  # use jj git push instead
```

---

## Quick Reference Card

```
LOOK:    jj st | jj log | jj diff | jj show <id>
COMMIT:  jj commit -m "msg"          (describe + new in one shot)
PUSH:    jj git push --bookmark name
FETCH:   jj git fetch
UNDO:    jj undo                     (always safe)
HISTORY: jj op log                   (operation log = full undo stack)
RECOVER: jj op restore <op-id>       (restore any past state)
```
