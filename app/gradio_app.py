import gradio as gr

from src.predict import predict

MAX_CHARS = 1000

EXAMPLES = [
    ["WINNER!! You have won a $1000 prize. Call now to claim."],
    ["Free entry in a weekly competition, text WIN to 80086"],
    ["Ok, see you at 6 at the cafe."],
    ["Can you pick up some milk on your way home?"],
]


def classify(message: str) -> tuple[str, float]:
    message = (message or "").strip()
    if not message:
        return "Please enter a message.", 0.0

    result = predict([message[:MAX_CHARS]])[0]
    verdict = "SPAM" if result["label"] == "spam" else "HAM (not spam)"
    return verdict, result["spam_probability"]


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="SMS message", placeholder="Type a message..."),
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Spam probability"),
    ],
    examples=EXAMPLES,
    title="SMS Spam Detector",
    description=(
        "BiLSTM classifier trained on the UCI SMS Spam Collection (English). "
        "Short or non-English messages may be unreliable."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()