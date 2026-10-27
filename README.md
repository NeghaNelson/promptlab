# PromptLab

PromptLab is a Streamlit application for analyzing, optimizing, executing, and comparing prompts. It is designed as a complete demonstration project for prompt-engineering coursework.

## Features

- Prompt analysis for objective, context, ambiguity, constraints, and output format.
- Prompt optimization using role-based, context-injection, and structured-output techniques.
- Selectable response tone (Professional, Friendly, Academic, or Humorous) and side-by-side execution of original and optimized prompts.
- Five-criterion rubric: relevance, completeness, instruction compliance, output format compliance, and factual support.
- Interactive score chart, percentage change, SQLite persistence, and JSON report export.
- Local demo mode works without an API key; Gemini integration is enabled with `GEMINI_API_KEY`.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add GEMINI_API_KEY to .env for live model responses (optional)
streamlit run app.py
```

Run tests with `pytest`.

## Evaluation limitations

The local evaluator is intentionally transparent and heuristic. It is useful for demonstrating the workflow, but it does not prove factual correctness or replace human review. For a course evaluation, create a fixed dataset of 30–50 tasks, run both prompt versions with the same model settings, and manually validate a sample of scores.
