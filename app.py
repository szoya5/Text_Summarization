import streamlit as st
import transformers
from transformers import pipeline

# Set up model paths (you can later replace these with fine-tuned model folders)
model_map = {
    "BART": "zoya23/bartmodel-summarization",
    "T5": "t5-small",
    "PEGASUS": "google/pegasus-cnn_dailymail"
}

# App Title
st.markdown("<h1 style='text-align: center;'>Text Summarization App</h1>", unsafe_allow_html=True)

# UI: Mode and Length controls
mode = st.radio("Modes", ["Paragraph", "Bullet Points", "Custom"], horizontal=True)
length_slider = st.slider("Summary Length", 1, 2, 1, label_visibility="collapsed")
length_label = "Short" if length_slider == 1 else "Long"
st.markdown(f"Summary Length: **{length_label}**")

# Model selection
model_choice = st.selectbox("Choose Summarization Model", ["BART", "T5", "PEGASUS"])

# 2-column layout
col1, col2 = st.columns(2)

# Left Column: Input
with col1:
    st.markdown("### Enter your text:")
    user_input = st.text_area("", height=300, placeholder="Paste your job description or content here...")

    # Word count
    word_count = len(user_input.split())
    st.markdown(f"**{word_count} words**")

    # Summarize Button
    if st.button("Summarize", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter text to summarize.")
        else:
            # Load model
            summarizer = pipeline("summarization", model=model_map[model_choice])

            # Set length dynamically
            max_len = 150 if length_label == "Short" else 300
            min_len = 40

            # Generate summary
            summary = summarizer(user_input, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            st.session_state["summary"] = summary

# Right Column: Output
with col2:
    st.markdown("### Summary")
    if "summary" in st.session_state:
        st.success(st.session_state["summary"])
        summary_words = len(st.session_state["summary"].split())
        st.markdown(f"📝 1 sentence • {summary_words} words")
        st.button("Paraphrase Summary")
        st.download_button("📥 Download Summary", st.session_state["summary"], file_name="summary.txt")
    else:
        st.info("Your summary will appear here.")
