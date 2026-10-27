from src.analyzer import analyze_prompt, optimize_prompt
from src.evaluator import evaluate_response
from src.llm import generate_pair


def test_analysis_identifies_missing_context():
    analysis = analyze_prompt("Explain machine learning.")
    assert analysis.weaknesses
    assert "Output format compliance" not in analysis.detected_techniques


def test_optimizer_preserves_original_task():
    analysis = analyze_prompt("Explain machine learning.")
    optimized = optimize_prompt("Explain machine learning.", analysis)
    assert "Explain machine learning." in optimized
    assert "Requirements" in optimized


def test_evaluator_returns_all_rubric_criteria():
    evaluation = evaluate_response("Explain this with headings.", "# Answer\n\nA complete explanation.")
    assert set(evaluation.scores) == {
        "Relevance",
        "Completeness",
        "Instruction compliance",
        "Output format compliance",
        "Factual support",
    }
    assert 5 <= evaluation.total <= 25


def test_generate_pair_includes_selected_tone_in_both_model_prompts(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.prompts = []

        def generate_content(self, prompt):
            self.prompts.append(prompt)
            return type("Result", (), {"text": "response"})()

    model = FakeModel()
    monkeypatch.setattr("src.llm._get_model", lambda model_name: model)

    generate_pair(
        "Explain machine learning.",
        analyze_prompt("Explain machine learning."),
        "",
        "",
        "test-model",
        "Academic",
    )

    assert len(model.prompts) == 2
    assert all(prompt.startswith("Use the Academic tone in your response.\n\n") for prompt in model.prompts)


def test_generate_pair_uses_optimized_override_and_generation_settings(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return type("Result", (), {"text": "response"})()

    model = FakeModel()
    monkeypatch.setattr("src.llm._get_model", lambda model_name: model)

    generate_pair(
        "Explain machine learning.",
        analyze_prompt("Explain machine learning."),
        "",
        "",
        "test-model",
        "Professional",
        temperature=0.25,
        max_tokens=256,
        few_shot_examples="Example input and output",
        optimized_prompt_override="Act as an expert.\n\nOptimized task.",
    )

    assert len(model.calls) == 2
    assert all(
        kwargs["generation_config"] == {
            "temperature": 0.25,
            "max_output_tokens": 256,
        }
        for _, kwargs in model.calls
    )
    assert all("Example input and output" in prompt for prompt, _ in model.calls)
    assert "Act as an expert.\n\nOptimized task." in model.calls[1][0]
