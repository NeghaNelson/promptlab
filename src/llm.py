import os

from .analyzer import optimize_prompt
from .models import PromptAnalysis


def _demo_response(prompt: str, tone: str = "") -> str:
    tone_text = f" in a {tone} tone" if tone else ""
    return (
        "## Response\n\n"
        f"This is a local demonstration response for the task: **{prompt[:180]}**{tone_text}\n\n"
        "- The task has been interpreted using the supplied objective.\n"
        "- Add a Gemini API key in `.env` to generate a real model response.\n\n"
        "### Assumptions\n"
        "This demo uses deterministic placeholder content, so its score is illustrative rather than an LLM result."
    )


def _get_model(model_name: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_pair(
    original: str,
    analysis: PromptAnalysis,
    expected_outcome: str,
    audience: str,
    model_name: str,
    tone: str = "Professional",
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    few_shot_examples: str = "",
    optimized_prompt_override: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str, str]:
    optimized = optimized_prompt_override or optimize_prompt(
        original, analysis, expected_outcome, audience
    )
    model = _get_model(model_name)

    if model is None:
        return (
            _demo_response(original, tone),
            _demo_response(optimized, tone),
            "Demo mode",
        )

    instructions = [f"Use the {tone} tone in your response."]
    if few_shot_examples.strip():
        instructions.extend(
            [
                "Use the following examples as guidance for style and format:",
                few_shot_examples.strip(),
            ]
        )
    prompt_prefix = "\n\n".join(instructions) + "\n\n"
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    token_limit = max_tokens if max_tokens is not None else max_output_tokens
    if token_limit is not None:
        generation_config["max_output_tokens"] = token_limit

    def generate(prompt: str):
        if generation_config:
            return model.generate_content(
                prompt_prefix + prompt,
                generation_config=generation_config,
            )
        return model.generate_content(prompt_prefix + prompt)

    original_result = generate(original)
    optimized_result = generate(optimized)

    return original_result.text, optimized_result.text, model_name


def generate_optimized(
    original: str,
    analysis: PromptAnalysis,
    expected_outcome: str,
    audience: str,
    model_name: str,
) -> tuple[str, str]:
    optimized = optimize_prompt(original, analysis, expected_outcome, audience)
    model = _get_model(model_name)
    if model is None:
        return optimized, "Demo mode"
    instruction = (
        "Rewrite the following optimized prompt to be more precise while preserving user intent. "
        "Return only the rewritten prompt.\n\n" + optimized
    )
    return model.generate_content(instruction).text, model_name