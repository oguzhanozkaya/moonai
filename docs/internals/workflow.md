---
description: Guide to set up, build, test, and deploy the project.
---

# Workflow

## Use `justfile` Commands

Always reach for a `just` recipe before running any raw CLI command. If a `just` command exists for the task. Do not run raw commands or tooling directly if any just recipe exists. The `justfile` ensures consistent behavior across development and CI environments.

## Development Commands

## Quality Gate

Before any commit or pull request:

```bash
just gate
```

This runs `just check` followed by `just test`.
