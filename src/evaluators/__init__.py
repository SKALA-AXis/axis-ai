"""Rule-based evaluator package.

Import submodules directly (e.g. ``from src.evaluators.evaluator import Evaluator``).
Avoid eager imports here — axis-ai-cron imports ``llm_judge_prompts`` only and must
not pull ``src.analysis`` (not shipped in the slim cron image).
"""
