# Releasing openDVP

Maintainer notes. Contributors do not need this — see [CONTRIBUTING.md](../CONTRIBUTING.md).

## In short

| Where | What you do |
| --- | --- |
| CLI, on a branch | Bump `version` in `pyproject.toml`. Rename `## [Unreleased]` in `CHANGELOG.md` to the version and date, leaving a fresh empty `[Unreleased]` above it. |
| CLI | `uv run --all-extras --group docs python scripts/check_tutorials.py` — nothing else executes the notebooks. |
| GitHub | Merge that PR into `main`. |
| GitHub | Releases → Draft a new release → **type a new tag name** → Target `main` → Generate notes → trim → Publish. |
| — | `publish.yml` builds and uploads to PyPI on its own. Nothing else to do. |

The only two things to remember: **never `git tag` yourself**, and **the version that reaches
PyPI is the one in `pyproject.toml`, not the tag name**. The rest of this file explains why.

## The three things that are easy to confuse

They are separate, and nothing keeps them in sync automatically.

| Thing | What it is | Created by |
| --- | --- | --- |
| **Tag** | A git label pointing at one commit. Inert — pushing a tag does nothing on its own. | the Release UI (**not** `git tag`) |
| **Release** | A GitHub object attached to a tag, with a title and notes. Creating one fires the `release: published` event. | GitHub UI or `gh release create` |
| **PyPI version** | The `version = "..."` string in `pyproject.toml`. **The tag name has no effect on it.** | `publish.yml`, on `release: published` |

The chain is:

```
edit pyproject.toml version  ->  merge to main  ->  publish a GitHub Release
                                                            |
                                                            v
                                             .github/workflows/publish.yml
                                              uv build -> twine check -> PyPI
```

Three consequences worth internalising:

1. **A tag alone publishes nothing.** `git tag v1.0.0 && git push --tags` creates a tag on GitHub and stops there. This is the single most common way a release silently doesn't happen.
2. **The version on PyPI comes from `pyproject.toml`, not from the tag name.** If the tag says `v0.8.0` and `pyproject.toml` says `0.7.9`, PyPI gets `0.7.9` and no error is raised.
3. **A PyPI version can never be reused.** Publish `0.8.0` once and that number is spent forever, even if you delete it. A broken release costs you a version number — always go forwards to `0.8.1`.

## The release checklist

Do this on a branch, merge it, then release from `main`.

### 1. Bump the version

Edit `version` in `pyproject.toml`. [Semantic versioning](https://semver.org/):

| Change | Bump | Example |
| --- | --- | --- |
| Breaking API change — renamed or removed function, changed defaults | MAJOR | `0.7.4` → `1.0.0` |
| New feature, backwards compatible | MINOR | `0.7.4` → `0.8.0` |
| Bug fix, docs, dependency bump | PATCH | `0.7.4` → `0.7.5` |

While on `0.x`, a breaking change conventionally bumps MINOR rather than MAJOR — `0.7.4` → `0.8.0`.

### 2. Write the changelog entry

Get the raw material first — this is the list of everything that merged, so nothing is forgotten:

```bash
gh api repos/CosciaLab/openDVP/releases/generate-notes \
  -f tag_name=v0.8.0 -f target_commitish=main --jq '.body'
```

This is read-only — it creates no tag and no release.

> **Gotcha:** if `tag_name` is a tag that *already exists*, GitHub compares against that tag and
> silently ignores `target_commitish`. You get a short, wrong list. Only use a tag name you have
> not created yet.

Then write the entry **by hand** in `CHANGELOG.md`. The generated list is PR titles; it tells a
reader what merged, not what changed for them. Three to five user-facing lines under
`### Added` / `### Changed` / `### Fixed` is plenty.

Most entries should already be there, added by the PRs that made the changes. So this step is
usually just closing the section off:

```diff
-## [Unreleased]
+## [Unreleased]
+
+---
+
+## [0.8.0] - 2026-09-14
```

That is: rename `[Unreleased]` to the version and today's date, and leave a fresh empty
`[Unreleased]` above it for the next cycle.

Two habits that make the generated list actually useful: give PRs descriptive titles (`Dev` tells
nobody anything), and squash-merge so one PR is one line.

### 3. Check the tutorials still run

Nothing else does. `nb_execution_mode` is `"off"`, so the docs build renders the notebooks'
committed outputs without executing a line of them, and the test suite does not touch them.

```bash
uv run --all-extras --group docs python scripts/check_tutorials.py
```

It runs T1, T2 and T3 in order — T3 reads a checkpoint T2 writes — and exits non-zero on any
failing cell. The two interactive napari cells are tagged `skip-execution` and reported as
skipped; they need a real display. Expect 10-20 minutes and a 133 MB download on first run; set
`OPENDVP_DATA_DIR` to reuse a cache.

This checks that the *code* still runs. It does not refresh the *outputs* the docs site shows —
for that, run the notebooks in Jupyter with the viewer cells live and commit the result. Worth
doing whenever their output has visibly drifted from the code.

### 4. Merge to `main`

Open a PR from your branch to `main`, let CI pass, merge it.

### 5. Publish the Release

**Do this in the GitHub UI, and let it create the tag.**

> [!IMPORTANT]
> **Never run `git tag` for a release.** The tag must not exist before you start this step.
>
> The button is labelled *"Choose a tag"*, which makes it look like you pick an existing one.
> You do not. You **type a tag name that does not exist yet**, and GitHub creates it — pointing
> at the target branch — at the moment you publish.
>
> Creating the tag yourself beforehand is how `v0.7.4` ended up frozen at a commit six behind
> `main`: the tag was made in the CLI, `main` moved on, and the tag stayed put.

> Releases → **Draft a new release**
> - **Choose a tag** → type `v0.8.0` → a dropdown appears: click **"＋ Create new tag: v0.8.0 on publish"**
>   - if it offers to *select* `v0.8.0` instead of *create* it, the tag already exists — stop and
>     delete it first
> - **Target** → `main`  ← confirm this, it is the whole point
> - **Release title** → `v0.8.0`
> - **Generate release notes** → gives you the PR list; replace or trim it with your changelog entry
> - **Publish release**

Equivalently, from the terminal:

```bash
gh release create v0.8.0 --repo CosciaLab/openDVP --target main \
  --title "v0.8.0" --notes-file <(sed -n '/## \[0.8.0\]/,/^---$/p' CHANGELOG.md)
```

### 6. Verify

```bash
# did the publish workflow succeed?
gh run list --repo CosciaLab/openDVP --workflow publish.yml --limit 1

# what does PyPI say the latest version is?
curl -s https://pypi.org/pypi/openDVP/json | jq -r '.info.version'
```

The publish workflow takes a couple of minutes. If it fails, fix the cause, bump to the next
patch version and release again — you cannot retry the same version number.

## Rules of thumb

- **Always** create the tag from the Release UI, targeting `main`.
- **Never** `git tag` locally for a release.
- Bump the version and update the changelog in the *same* PR, before releasing.
- One release per version number, forwards only.
- If a release fails to publish, go to the next patch number rather than trying to repair it.

## History, and why the tag list looks messy

For reference, the state as of 2026-09-14 — 28 tags, 4 Releases, 24 PyPI versions:

- **v0.1.1 – v0.6.5** were tagged and uploaded to PyPI manually. Tags exist, Releases do not.
  That is why the Releases page looks nearly empty despite 24 published versions.
- **v0.7.0 – v0.7.3** are the first four done the automated way, and are consistent across all
  three lists. This is the pattern to keep.
- **v0.2.5, v0.6.0, v0.6.1, v0.6.2** are tagged but never reached PyPI — tags created, no
  Release, so `publish.yml` never fired.
- **v0.7.4** was in the same state, and additionally pointed six commits behind `main`.

Old tags are harmless; leave them. Nothing needs cleaning up retroactively.
