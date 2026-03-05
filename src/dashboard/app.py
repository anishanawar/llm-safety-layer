import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="LLM Safety Layer", layout="wide")

st.title("LLM Safety Layer Dashboard")
st.markdown("Evaluate prompt risk, hallucination uncertainty, and safety decisions.")

prompt = st.text_area("Enter Prompt:", height=150)

if st.button("Analyze"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json={"prompt": prompt})

        if response.status_code == 200:
            result = response.json()

            # Decision Display
            decision = result["decision"]["decision"]
            reason = result["decision"]["reason"]

            if decision == "allow":
                st.success(f"Decision: {decision.upper()}")
            elif decision == "abstain":
                st.warning(f"Decision: {decision.upper()}")
            else:
                st.error(f"Decision: {decision.upper()}")

            st.markdown(f"**Reason:** {reason}")

            # prompt risk
            st.subheader("Prompt Risk Analysis")
            st.json(result["prompt_risk"])

            # hallucination metrics
            st.subheader("📊 Hallucination Metrics")
            entropy = result["hallucination"]["entropy"]
            norm_entropy = result["hallucination"]["normalized_entropy"]

            st.metric("Entropy", round(entropy, 4))
            st.metric("Normalized Entropy", round(norm_entropy, 4))

            # retrieved context
            st.subheader("Retrieved Context")
            for doc in result["retrieved_context"]:
                st.write(f"- {doc}")

            # generated samples
            st.subheader("Generated Samples")
            for i, sample in enumerate(result["samples"]):
                st.markdown(f"**Sample {i+1}:**")
                st.write(sample)
                st.divider()

        else:
            st.error("Error contacting API.")