import html
import json
import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analyzer import analyze_prompt, optimize_prompt
from src.database import get_all_experiments, init_db, save_experiment
from src.evaluator import evaluate_responses
from src.llm import generate_pair

init_db()
st.set_page_config(page_title="PromptLab Pro", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

TEMPLATES = {
    "Machine learning explainer": {
        "prompt": "Explain machine learning and give three practical examples.",
        "outcome": "A clear introduction with practical examples and a short glossary.",
        "audience": "First-year computer science students",
    },
    "Product launch brief": {
        "prompt": "Write a launch announcement for a new productivity app.",
        "outcome": "A concise launch brief with a headline, key benefits, and call to action.",
        "audience": "Busy professionals evaluating productivity tools",
    },
    "Research summary": {
        "prompt": "Summarize the main arguments and limitations of the supplied research.",
        "outcome": "A balanced summary that separates findings, evidence, and limitations.",
        "audience": "Graduate students outside the research specialty",
    },
}

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        color-scheme: dark;
        --bg: #0c0b12;
        --panel: rgba(22, 20, 33, 0.72);
        --line: rgba(210, 192, 255, 0.13);
        --muted: #9c96ad;
        --text: #f3efff;
        --purple: #a879ff;
        --cyan: #62e9f4;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(ellipse at 12% -10%, rgba(120, 69, 202, 0.22), transparent 38%),
            radial-gradient(ellipse at 90% 0%, rgba(41, 143, 166, 0.11), transparent 34%),
            var(--bg);
    }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(14, 12, 21, 0.95);
        border-right: 1px solid rgba(168, 121, 255, 0.55);
        box-shadow: 8px 0 32px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #f7f3ff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] input {
        color: #f7f3ff !important;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: #eee8ff;
        letter-spacing: -0.035em;
    }
    .stApp h1 {
        background: linear-gradient(100deg, #d7b6ff 0%, #a879ff 48%, #62e9f4 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .stApp p, .stApp li { color: #c9c3d6; }
    .stApp label, [data-testid="stWidgetLabel"] p {
        color: var(--muted) !important;
        font-size: .67rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.025);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    .stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        color: var(--text) !important;
        background: rgba(255,255,255,.035) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .stTextArea textarea:focus, .stTextInput input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: rgba(98,233,244,.72) !important;
        box-shadow: 0 0 0 2px rgba(98,233,244,.12), 0 0 20px rgba(168,121,255,.16) !important;
    }
    .stButton > button {
        width: 100%;
        min-height: 42px;
        color: white !important;
        background: linear-gradient(100deg, #7548c8, #496fd1 58%, #2499aa);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 7px 22px rgba(92,68,170,.2);
        transition: .18s ease;
    }
    .stButton > button:hover {
        color: white !important;
        filter: brightness(1.16);
        border-color: rgba(98,233,244,.55);
        box-shadow: 0 0 24px rgba(98,233,244,.2), 0 8px 28px rgba(125,75,204,.26);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"], .glass-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: inset 0 0 0 1px rgba(180,145,255,.035), 0 12px 30px rgba(0,0,0,.18);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    [data-testid="stMetric"] { padding: 14px 16px; }
    .glass-card {
        padding: 16px;
        margin: 0 0 12px;
        color: var(--text);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .response-copy {
        font-family: 'Fira Code', monospace;
        font-size: .82rem;
        line-height: 1.7;
        color: #e0d9ef;
    }
    [data-testid="stCode"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(13,12,20,.78);
    }
    [data-testid="stTabs"] button { font-weight: 600; }
    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.045);
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("prompt_input", TEMPLATES["Machine learning explainer"]["prompt"])
st.session_state.setdefault("outcome_input", TEMPLATES["Machine learning explainer"]["outcome"])
st.session_state.setdefault("audience_input", TEMPLATES["Machine learning explainer"]["audience"])
st.session_state.setdefault("few_shot_input", "")
st.session_state.setdefault("lab_results", None)
st.session_state.setdefault("battle_results", None)
st.session_state.setdefault("latest_result_key", None)


def _build_optimized_prompt(
    prompt: str,
    outcome: str,
    audience: str,
    toolkit: dict[str, bool],
) -> str:
    analysis = analyze_prompt(prompt, outcome, audience)
    base_prompt = optimize_prompt(prompt, analysis, outcome, audience)
    instructions = []
    if toolkit["role_playing"]:
        instructions.append("Act as an expert subject-matter specialist for this task.")
    if toolkit["chain_of_thought"]:
        instructions.append(
            "Think step-by-step through the task before producing a concise final answer."
        )
    if toolkit["strict_formatting"]:
        instructions.append(
            "Respond ONLY in valid JSON. Do not include markdown fences or text outside the JSON object."
        )
    instructions.append(base_prompt)
    return "\n\n".join(instructions)


def _execute_models(
    models: list[str],
    prompt: str,
    outcome: str,
    audience: str,
    few_shot_examples: str,
    tone: str,
    temperature: float,
    max_tokens: int,
    toolkit: dict[str, bool],
) -> dict:
    analysis = analyze_prompt(prompt, outcome, audience)
    optimized_prompt = _build_optimized_prompt(prompt, outcome, audience, toolkit)
    runs = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model_name in models:
        original_response, optimized_response, model_used = generate_pair(
            prompt,
            analysis,
            outcome,
            audience,
            model_name,
            tone,
            temperature=temperature,
            few_shot_examples=few_shot_examples,
            optimized_prompt_override=optimized_prompt,
            max_tokens=max_tokens,
        )
        evaluation = evaluate_responses(
            prompt,
            optimized_prompt,
            original_response,
            optimized_response,
        )
        run = {
            "selected_model": model_name,
            "model_used": model_used,
            "original_response": original_response,
            "optimized_response": optimized_response,
            "evaluation": evaluation,
        }
        runs.append(run)
        save_experiment(
            timestamp,
            model_used,
            tone,
            prompt,
            optimized_prompt,
            original_response,
            optimized_response,
            evaluation["original_avg"],
            evaluation["optimized_avg"],
        )

    return {
        "timestamp": timestamp,
        "prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "outcome": outcome,
        "audience": audience,
        "few_shot_examples": few_shot_examples,
        "tone": tone,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "toolkit": toolkit.copy(),
        "runs": runs,
    }


def _render_card(title: str, content: str) -> None:
    st.markdown(
        f'<div class="glass-card"><strong>{html.escape(title)}</strong><br><br>{html.escape(content)}</div>',
        unsafe_allow_html=True,
    )


def _render_result(run: dict) -> None:
    evaluation = run["evaluation"]
    st.markdown(f"#### {run['selected_model']} · {run['model_used']}")
    response_columns = st.columns(2)
    with response_columns[0]:
        with st.expander("Original response", expanded=False):
            st.code(run["original_response"], language="markdown", width="stretch")
    with response_columns[1]:
        with st.expander("Optimized response", expanded=True):
            st.code(run["optimized_response"], language="markdown", width="stretch")

    score_columns = st.columns(3)
    score_columns[0].metric("Original score", f"{evaluation['original_avg']:.2f} / 5")
    score_columns[1].metric("Optimized score", f"{evaluation['optimized_avg']:.2f} / 5")
    score_delta = evaluation["optimized_avg"] - evaluation["original_avg"]
    score_columns[2].metric(
        "Score delta",
        f"{score_delta:+.2f} points",
        delta=f"{score_delta:+.2f}",
    )
    with st.expander("View Raw JSON Evaluation", expanded=False):
        st.code(
            json.dumps(
                {
                    "model": run["selected_model"],
                    "runtime_model": run["model_used"],
                    "original_scores": evaluation["original_scores"],
                    "optimized_scores": evaluation["optimized_scores"],
                    "original_average": evaluation["original_avg"],
                    "optimized_average": evaluation["optimized_avg"],
                    "feedback": evaluation["feedback"],
                },
                indent=2,
            ),
            language="json",
            line_numbers=True,
            width="stretch",
        )


def _build_score_delta_figure(runs: list[dict]) -> go.Figure:
    criteria = list(runs[0]["evaluation"]["original_scores"].keys())
    deltas = [
        sum(
            run["evaluation"]["optimized_scores"][criterion]
            - run["evaluation"]["original_scores"][criterion]
            for run in runs
        )
        / len(runs)
        for criterion in criteria
    ]
    figure = go.Figure(
        go.Bar(
            x=deltas,
            y=criteria,
            orientation="h",
            marker_color=["#4ed5d2" if delta >= 0 else "#f080a2" for delta in deltas],
            text=[f"{delta:+.2f}" for delta in deltas],
            textposition="outside",
            cliponaxis=False,
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=12, r=36, t=18, b=35),
        xaxis=dict(
            title="Optimized − original score (points)",
            zeroline=True,
            zerolinecolor="#8c819f",
            gridcolor="rgba(255,255,255,.08)",
        ),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee8ff"),
    )
    return figure


with st.sidebar:
    st.markdown("## 🧪 PromptLab")
    st.caption("EXPERIMENT CONTROL PLANE")
    if st.button("Go to Lab", width="stretch", key="go_to_lab"):
        st.rerun()

    with st.expander("Model Config", expanded=True):
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro"]
        primary_model = st.selectbox("Primary model", model_options, key="primary_model")
        opponent_options = [model for model in model_options if model != primary_model]
        opponent_model = st.selectbox("Battle opponent", opponent_options, key="opponent_model")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=128,
            max_value=8192,
            value=1024,
            step=128,
        )
        tone = st.selectbox(
            "Response tone",
            ["Professional", "Friendly", "Academic", "Humorous"],
        )

    with st.expander("Prompt Engineering Toolkit", expanded=False):
        toolkit = {
            "role_playing": st.checkbox("Role-Playing", value=False),
            "chain_of_thought": st.checkbox("Chain of Thought", value=False),
            "strict_formatting": st.checkbox("Strict Formatting", value=False),
        }

    with st.expander("Prompt Templates", expanded=False):
        selected_template = st.selectbox("Load a starting point", list(TEMPLATES))
        if st.button("Apply template", width="stretch"):
            template = TEMPLATES[selected_template]
            st.session_state["prompt_input"] = template["prompt"]
            st.session_state["outcome_input"] = template["outcome"]
            st.session_state["audience_input"] = template["audience"]

    st.caption("Experiment runs are stored in the local SQLite database.")

st.title("PromptLab · AI Evaluation Workspace")
st.caption("Design, compare, and inspect prompt experiments.")
lab_tab, battle_tab, analytics_tab, history_tab = st.tabs(
    ["🧪 Lab", "⚔️ Battle", "📊 Analytics", "📜 History"]
)

with lab_tab:
    st.subheader("Experiment")
    st.caption(
        f"PRIMARY MODEL / {primary_model}  ·  "
        f"TEMPERATURE / {temperature:.2f}  ·  TONE / {tone}"
    )

    prompt_columns = st.columns([3, 1])
    with prompt_columns[0]:
        prompt = st.text_area("Task prompt", height=152, key="prompt_input")
    with prompt_columns[1]:
        estimated_tokens = max(1, math.ceil(len(prompt) / 4))
        st.metric("Estimated prompt tokens", f"~{estimated_tokens}")
        st.caption("Approx. 1 token per 4 characters")

    context_columns = st.columns(2)
    with context_columns[0]:
        outcome = st.text_area("Expected outcome", height=85, key="outcome_input")
    with context_columns[1]:
        audience = st.text_area("Target audience", height=85, key="audience_input")
    few_shot_examples = st.text_area(
        "Few-Shot Examples",
        placeholder="Add example input/output pairs to guide response style and structure.",
        height=110,
        key="few_shot_input",
    )

    if st.button("▶  Analyze & Run Experiment", type="primary", width="stretch"):
        if not prompt.strip():
            st.warning("Enter a task prompt before running an experiment.")
        else:
            try:
                with st.spinner("Generating responses and evaluating rubric criteria..."):
                    result = _execute_models(
                        [primary_model],
                        prompt,
                        outcome,
                        audience,
                        few_shot_examples,
                        tone,
                        temperature,
                        int(max_tokens),
                        toolkit,
                    )
                    st.session_state["lab_results"] = result
                    st.session_state["latest_result_key"] = "lab_results"
                st.success("Experiment saved to history.")
            except Exception as error:
                st.error(f"Experiment failed: {error}")

    lab_results = st.session_state["lab_results"]
    if lab_results:
        st.divider()
        st.subheader("Run output")
        _render_card("Optimized prompt", lab_results["optimized_prompt"])
        _render_result(lab_results["runs"][0])

with battle_tab:
    st.subheader("Model Battle")
    st.caption(f"OPTIMIZED PROMPT / {primary_model} vs {opponent_model}")
    st.markdown(
        "Run the same task and few-shot examples against both configured models. "
        "Each model's original and optimized response scores are recorded separately."
    )

    if st.button("⚔️ Run model battle", type="primary", width="stretch"):
        if not prompt.strip():
            st.warning("Enter a task prompt in the Lab tab before starting a battle.")
        else:
            try:
                with st.spinner("Running both models and scoring their responses..."):
                    result = _execute_models(
                        [primary_model, opponent_model],
                        prompt,
                        outcome,
                        audience,
                        few_shot_examples,
                        tone,
                        temperature,
                        int(max_tokens),
                        toolkit,
                    )
                    st.session_state["battle_results"] = result
                    st.session_state["latest_result_key"] = "battle_results"
                st.success("Both model runs saved to history.")
            except Exception as error:
                st.error(f"Battle failed: {error}")

    battle_results = st.session_state["battle_results"]
    if battle_results:
        st.divider()
        st.subheader("Model response comparison")
        battle_columns = st.columns(2)
        for column, run in zip(battle_columns, battle_results["runs"]):
            with column:
                _render_result(run)

with analytics_tab:
    st.subheader("Experiment analytics")
    history = get_all_experiments()
    if history.empty:
        original_average = optimized_average = None
        score_delta = None
        improvement = None
    else:
        original_average = float(history["avg_original_score"].mean())
        optimized_average = float(history["avg_optimized_score"].mean())
        score_delta = optimized_average - original_average
        improvement = score_delta / original_average * 100 if original_average else 0.0

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Original Avg Score",
        f"{original_average:.2f} / 5" if original_average is not None else "—",
    )
    metric_columns[1].metric(
        "Optimized Avg Score",
        f"{optimized_average:.2f} / 5" if optimized_average is not None else "—",
    )
    metric_columns[2].metric(
        "Score Delta",
        f"{score_delta:+.2f} points" if score_delta is not None else "—",
        delta=f"{improvement:+.1f}% improvement" if improvement is not None else None,
        delta_color="normal",
    )

    latest_key = st.session_state["latest_result_key"]
    latest_results = st.session_state[latest_key] if latest_key else None
    if latest_results:
        st.markdown("#### Criterion score differences · latest run")
        chart_runs = latest_results["runs"]
        criteria = list(chart_runs[0]["evaluation"]["original_scores"].keys())
        deltas = [
            sum(
                run["evaluation"]["optimized_scores"][criterion]
                - run["evaluation"]["original_scores"][criterion]
                for run in chart_runs
            )
            / len(chart_runs)
            for criterion in criteria
        ]
        delta_chart = go.Figure(
            go.Bar(
                x=deltas,
                y=criteria,
                orientation="h",
                marker_color=["#4ed5d2" if value >= 0 else "#f080a2" for value in deltas],
                text=[f"{value:+.2f}" for value in deltas],
                textposition="outside",
                cliponaxis=False,
            )
        )
        delta_chart.update_layout(
            height=340,
            margin=dict(l=12, r=40, t=18, b=40),
            xaxis=dict(
                title="Optimized − original score (points)",
                zeroline=True,
                zerolinecolor="#a99dbb",
                gridcolor="rgba(255,255,255,.08)",
            ),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee8ff"),
        )
        st.plotly_chart(delta_chart, width="stretch")
    else:
        st.info("Run an experiment or model battle to see criterion-level score differences.")

with history_tab:
    st.subheader("Experiment history")
    history = get_all_experiments()
    if history.empty:
        st.info("No experiments have been logged yet.")
    else:
        st.dataframe(history, hide_index=True, width="stretch")
        st.download_button(
            "Download history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="promptlab_history.csv",
            mime="text/csv",
            width="stretch",
        )
