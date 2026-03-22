"""Fourth-Down Decision Calculator — interactive prediction UI."""

import os

import streamlit as st
import requests

st.set_page_config(page_title="4th Down Calculator", page_icon="🏈", layout="wide")

st.title("🏈 4th Down Decision Calculator")
st.caption("Enter a game scenario to get the model's recommendation.")

API_URL = st.sidebar.text_input("API URL", value=os.environ.get("API_URL", "http://localhost:8000"))

# --- Game State Input ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Situation")
    qtr = st.selectbox("Quarter", [1, 2, 3, 4, 5], index=2, format_func=lambda x: f"Q{x}" if x <= 4 else "OT")
    ydstogo = st.number_input("Yards to Go", min_value=1, max_value=99, value=3)
    yardline_100 = st.number_input("Yards from End Zone", min_value=1, max_value=99, value=35)
    goal_to_go = st.checkbox("Goal to Go")

with col2:
    st.subheader("Score & Time")
    score_differential = st.number_input("Score Differential", min_value=-50, max_value=50, value=-7,
                                         help="Positive = your team leads")
    game_seconds = st.number_input("Game Seconds Remaining", min_value=0, max_value=3600, value=900)
    half_seconds = st.number_input("Half Seconds Remaining", min_value=0, max_value=1800, value=900)
    quarter_seconds = st.number_input("Quarter Seconds Remaining", min_value=0, max_value=900, value=900)

with col3:
    st.subheader("Context")
    wp = st.slider("Win Probability", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

# --- Predict ---
if st.button("Get Recommendation", type="primary", use_container_width=True):
    payload = {
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "score_differential": score_differential,
        "half_seconds_remaining": half_seconds,
        "game_seconds_remaining": game_seconds,
        "quarter_seconds_remaining": quarter_seconds,
        "qtr": qtr,
        "goal_to_go": int(goal_to_go),
        "wp": wp,
    }
    try:
        resp = requests.post(f"{API_URL}/fourth-down/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        rec = result["recommendation"]
        probs = result["probabilities"]

        emoji = {"go_for_it": "🏃", "punt": "🦵", "field_goal": "🥅"}.get(rec, "❓")
        st.success(f"{emoji} **Recommendation: {rec.replace('_', ' ').title()}**")

        st.subheader("Probabilities")
        for decision, prob in sorted(probs.items(), key=lambda x: -x[1]):
            label = decision.replace("_", " ").title()
            st.progress(prob, text=f"{label}: {prob:.1%}")

        with st.expander("Raw Response"):
            st.json(result)

    except requests.ConnectionError:
        st.error(f"Could not connect to API at {API_URL}. Is the server running?")
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
