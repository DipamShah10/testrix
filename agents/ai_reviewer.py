from qa.models import FigmaBaseline, QAEvaluation, RequirementModel


class AIReviewer:
    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def review(
        self,
        requirements: RequirementModel,
        figma: FigmaBaseline | None,
        qa_result: QAEvaluation,
        strict_accessibility: bool = False,
    ) -> dict:
        MAX_FINDINGS = 40
        MAX_GAPS = 30
        MAX_RESPONSIVE = 20

        findings_text = [f"{item.severity}|{item.title}|{item.evidence}" for item in qa_result.findings]
        unique_gaps = list(dict.fromkeys(qa_result.accessibility_gaps))
        unique_responsive = list(dict.fromkeys(qa_result.responsive_issues))

        prompt = f"""
You are a Senior QA Lead and release reviewer.
Generate JSON keys:
- qa_observations: list[str]
- bug_summary: list[str]
- ux_analysis: list[str]
- recommendation: "GO" or "NO-GO"
- rationale: string

Inputs:
requirements_features={requirements.features}
acceptance_criteria={requirements.acceptance_criteria}
figma_components={(figma.components if figma else [])}
findings (showing {min(len(findings_text), MAX_FINDINGS)} of {len(findings_text)})={findings_text[:MAX_FINDINGS]}
accessibility_gaps (unique, showing {min(len(unique_gaps), MAX_GAPS)} of {len(unique_gaps)} unique / {len(qa_result.accessibility_gaps)} total)={unique_gaps[:MAX_GAPS]}
responsive_issues (showing {min(len(unique_responsive), MAX_RESPONSIVE)} of {len(unique_responsive)})={unique_responsive[:MAX_RESPONSIVE]}
missing_features={qa_result.missing_features}
accessibility_blocker_count={qa_result.accessibility_blocker_count}
"""
        result = await self.llm_client.complete_json(
            system_prompt="Use GStack-style review rigor: clear, critical, and evidence-driven.",
            user_prompt=prompt,
        )
        if "recommendation" not in result:
            result["recommendation"] = "NO-GO" if any(f.severity in {"critical", "high"} for f in qa_result.findings) else "GO"
        if strict_accessibility and qa_result.accessibility_blocker_count > 0:
            result["recommendation"] = "NO-GO"
            existing_rationale = result.get("rationale", "")
            result["rationale"] = (
                f"{existing_rationale} Strict accessibility mode is enabled and blocker-level accessibility issues were found."
            ).strip()
        return result