import streamlit as st

from src.predict import predict

MAX_CHARS = 1000

EXAMPLES = {
    "Spam: prize": "WINNER!! You have won a $1000 prize. Call now to claim.",
    "Spam: competition": "Free entry in a weekly competition, text WIN to 80086",
    "Ham: meeting": "Ok, see you at 6 at the cafe.",
    "Ham: errand": "Can you pick up some milk on your way home?",
}

st.set_page_config(page_title="SMS Spam Detector", page_icon=None)
st.title("SMS Spam Detector")
st.caption(
    "BiLSTM trained on the UCI SMS Spam Collection (English SMS only). "
    "Not suitable for email, and short or non-English messages may be unreliable."
)

choice = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
default_text = EXAMPLES.get(choice, "")

message = st.text_area("SMS message", value=default_text, height=120, max_chars=MAX_CHARS)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        result = predict([message.strip()])[0]
        probability = result["spam_probability"]
        if result["label"] == "spam":
            st.error(f"SPAM (spam probability: {probability:.1%})")
        else:
            st.success(f"HAM, not spam (spam probability: {probability:.1%})")
        st.progress(min(max(probability, 0.0), 1.0))