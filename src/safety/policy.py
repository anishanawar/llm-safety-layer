class SafetyPolicy:
    """
    combines multiple signals into a single decision:
      - prompt risk (prompt injection / harmful intent)
      - hallucination risk (self-consistency entropy)
      - optional confidence signals (future)
    decisions:
      - "refuse": block unsafe requests
      - "abstain": uncertain; request clarification or human review
      - "allow": safe enough to proceed
    """

    def __init__(
        self,
        refuse_risk_score_threshold=1,         # any injection/harm hit => refuse
        abstain_entropy_threshold=0.55,        # high disagreement => abstain
        allow_entropy_threshold=0.30           # low disagreement => allow
    ):
        self.refuse_risk_score_threshold = refuse_risk_score_threshold
        self.abstain_entropy_threshold = abstain_entropy_threshold
        self.allow_entropy_threshold = allow_entropy_threshold

    def decide(self, prompt_risk: dict, hallucination: dict):
        """
        inputs:
          prompt_risk: {"risk_score": int, "injection_hits": int, "harmful_hits": int}
          hallucination: {"entropy": float, "normalized_entropy": float, "samples": [...]}

        returns:
          {"decision": "...", "reason": "...", "signals": {...}}
        """
        risk_score = int(prompt_risk.get("risk_score", 0))
        norm_entropy = float(hallucination.get("normalized_entropy", 0.0))

        # hard block if prompt risk is non-zero (simple but strong baseline)
        if risk_score >= self.refuse_risk_score_threshold:
            return {
                "decision": "refuse",
                "reason": "Prompt flagged as potential injection/harmful intent.",
                "signals": {
                    "risk_score": risk_score,
                    "normalized_entropy": norm_entropy
                }
            }

        # abstain if model outputs disagree heavily (uncertainty)
        if norm_entropy >= self.abstain_entropy_threshold:
            return {
                "decision": "abstain",
                "reason": "High output disagreement (hallucination risk).",
                "signals": {
                    "risk_score": risk_score,
                    "normalized_entropy": norm_entropy
                }
            }

        # allow if low disagreement
        if norm_entropy <= self.allow_entropy_threshold:
            return {
                "decision": "allow",
                "reason": "Low disagreement and no prompt risk flags.",
                "signals": {
                    "risk_score": risk_score,
                    "normalized_entropy": norm_entropy
                }
            }

        # middle zone => abstain (conservative)
        return {
            "decision": "abstain",
            "reason": "Uncertainty in middle zone; safer to abstain.",
            "signals": {
                "risk_score": risk_score,
                "normalized_entropy": norm_entropy
            }
        }