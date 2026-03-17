# Branch Protection Setup

After pushing to GitHub, configure these branch protection rules via
**Settings > Branches > Branch protection rules**.

## `main` branch

- [x] **Require a pull request before merging**
  - Required approving reviews: 1
  - Dismiss stale pull request approvals when new commits are pushed
- [x] **Require status checks to pass before merging**
  - Required checks:
    - `Linux (Debug)`
    - `Linux (Release)`
    - `Windows (Debug)`
    - `Windows (Release)`
- [x] **Require branches to be up to date before merging**
- [x] **Do not allow bypassing the above settings**
- [ ] Require signed commits (optional)
- [x] **Restrict who can push** - No direct pushes, only merges via PR

## `dev` branch

- [x] **Require a pull request before merging**
  - Required approving reviews: 1
- [x] **Require status checks to pass before merging**
  - Required checks:
    - `Linux (Debug)`
    - `Linux (Release)`

## Workflow

```
feature/xxx ──PR──> dev ──PR──> main ──tag──> Release
```

1. Create feature branches from `dev`
2. Open PR to `dev`, get 1 review, CI must pass
3. When `dev` is stable, open PR to `main`, all CI must pass
4. Merging to `main` triggers release workflow when tagged
5. Tag with `git tag v0.1.0 && git push --tags` to create a release
