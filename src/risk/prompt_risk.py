import re


class PromptRiskScorer:
    def __init__(self):
        # simple heuristic patterns (can expand later)
        self.injection_patterns = [
            r"ignore previous instructions",
            r"disregard system",
            r"override instructions",
            r"act as",
            r"bypass",
            r"jailbreak",
            r"simulate",
            r"pretend you are"
        ]

        self.harmful_patterns = [
            r"hack",
            r"exploit",
            r"steal",
            r"attack",
            r"fraud",
            r"bypass security"
        ]

    def score(self, prompt: str):
        prompt_lower = prompt.lower()

        injection_hits = sum(
            1 for pattern in self.injection_patterns
            if re.search(pattern, prompt_lower)
        )

        harmful_hits = sum(
            1 for pattern in self.harmful_patterns
            if re.search(pattern, prompt_lower)
        )

        risk_score = injection_hits + harmful_hits

        return {
            "risk_score": risk_score,
            "injection_hits": injection_hits,
            "harmful_hits": harmful_hits
        }