# Engineering Workflow

Use for repo changes.

## Loop

inspect context -> classify owner -> make smallest slice -> run tests -> repeat.

## TDD shape

one behavior -> one change -> green -> refactor.
No bulk speculative rewrites.
No tests that protect old file shape.

## Collapse rule

can delete? -> delete.
can internalize? -> internalize.
must expose? -> package root only.
hard lesson? -> guardrail/test/classifier.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Commit report

files deleted -> files changed -> public exports before/after -> tests run -> known remaining bloat.
