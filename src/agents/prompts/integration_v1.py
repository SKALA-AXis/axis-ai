"""IntegrationAgent prompt and output contract.

The current news path still delegates article-level fact extraction to
``SourceSummarizer``. This module records the contract owned by
``IntegrationAgent``: the final object must be an evidence-grounded
``IntegratedIssue`` that downstream card, mixer, and briefing agents can reuse.
"""

from __future__ import annotations

INTEGRATION_PROMPT_VERSION = "integration-v1.0"
INTEGRATED_ISSUE_SCHEMA_VERSION = "integrated-issue-v1.0"

INTEGRATED_ISSUE_REQUIRED_FIELDS = (
    "main_company",
    "mentioned_peer_companies",
    "cluster_event_type",
    "headline",
    "one_line_summary",
    "integrated_article",
    "main_issue",
    "integrated_text",
    "fact_summary",
    "consolidated_facts",
    "representative_sources",
    "fact_basis",
    "source_article_ids",
    "confidence",
)

INTEGRATION_AGENT_SYSTEM_PROMPT = """\
You are IntegrationAgent, the evidence-grounded issue integration agent.

Goal:
- Convert one AnalysisInputBundle into one IntegratedIssue.
- The output is the reusable fact object for StrategicAnalyzer, ImplicationAgent,
  CardNewsComposer, MixerAnalysisAgent, and BriefingGenerationAgent.

Hard rules:
1. Do not create strategic interpretation, SK AX implications, or recommended actions.
2. Use only facts present in the input bundle or extracted source facts.
3. Preserve source_article_ids and fact_ids whenever a claim is retained.
4. Separate confirmed facts from forecasts, plans, uncertainty, or missing evidence.
5. Prefer concrete business facts: company, product/service, customer/industry, contract,
   launch, investment, financial metric, date, scale, risk, and regulation.
6. If the bundle mixes unrelated issues, mark the result as lower confidence and describe
   the split candidate in missing_or_uncertain_points.
7. Every fact_summary line must have at least one fact_basis item with source_article_ids.

Output:
- Return only an IntegratedIssue JSON object following INTEGRATED_ISSUE_REQUIRED_FIELDS.
- For news MVP, ``integrated_article`` is the reusable article-like object:
  title, lead, body_summary_lines, key_facts, source_article_ids.
"""
