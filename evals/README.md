# Evals

Golden datasets and evaluation suites (grounding, routing, trajectory) live here.

## Long-Horizon Project Memory (`longmemeval`)

Runs the 12-month *Helios-AI* benchmark across the 5 LongMemEval axes (`IE`, `MSR`, `TR`, `KU`, `ABS`) against a live `ai/` server:

```powershell
Set-Location d:\project\flux\ai; $env:PYTHONPATH="src"; & '.\.venv\Scripts\python.exe' evals/longmemeval/runner.py --base-url http://127.0.0.1:18420
```

Stress test with 60 distractor notes alongside the 12-month golden corpus:

```powershell
Set-Location d:\project\flux\ai; $env:PYTHONPATH="src"; & '.\.venv\Scripts\python.exe' evals/longmemeval/runner.py --base-url http://127.0.0.1:18420 --noise-docs 60
```
