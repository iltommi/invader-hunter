# Contributing

## Git hooks

A pre-commit hook stamps `docs/index.html` with the current build date/time on every commit, so the loader screen always shows which version is deployed.

Install it once after cloning:

```bash
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
