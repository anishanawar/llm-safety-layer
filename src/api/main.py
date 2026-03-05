from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.retriever import SimpleRetriever
from src.risk.prompt_risk import PromptRiskScorer
from src.risk.hallucination_score import HallucinationScorer
from src.safety.policy import SafetyPolicy


app = FastAPI(title="LLM Safety Layer")

# initialize components once at startup
retriever = SimpleRetriever()
prompt_risk_scorer = PromptRiskScorer()
hallucination_scorer = HallucinationScorer()
policy = SafetyPolicy()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/analyze")
def analyze_prompt(request: PromptRequest):
    prompt = request.prompt

    # retrieve relevant knowledge
    retrieved_context = retriever.retrieve(prompt)

    # prompt risk scoring
    prompt_risk = prompt_risk_scorer.score(prompt)

    # hallucination scoring
    hallucination = hallucination_scorer.score(prompt)

    # safety decision
    decision = policy.decide(prompt_risk, hallucination)

    return {
        "prompt": prompt,
        "retrieved_context": retrieved_context,
        "prompt_risk": prompt_risk,
        "hallucination": {
            "entropy": hallucination["entropy"],
            "normalized_entropy": hallucination["normalized_entropy"]
        },
        "decision": decision,
        "samples": hallucination["samples"]
    }