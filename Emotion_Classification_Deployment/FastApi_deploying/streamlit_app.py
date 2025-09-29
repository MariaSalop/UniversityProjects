import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
import mimetypes
import io

# Configure logging to both console and file
log_file_path = "frontend-logging.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.info("🔧 Streamlit app started.")

API_BASE = "http://backend:8000"

st.set_page_config(
    page_title="Content Intelligence | Emotion AI", layout="centered"
)

st.title("🧠 Content Intelligence Emotion Recognition Platform")
logger.info("Page title and header set.")

st.markdown(
    '[🔗 Powered by Content Intelligence]'
    '(https://www.contentintelligence.nl/)',
    unsafe_allow_html=True,
)

st.markdown(
    """
Welcome to the Content Intelligence emotion AI platform — 
enabling data-driven insights into emotions in text to power smarter content decisions.
Our app works for the English language only.
"""
)
logger.info("Introductory markdown rendered.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict Emotion",
    "📁 Upload CSV",
    "📺 YouTube Transcript",
    "🎤 Upload Audio",
])
logger.info("Tabs created.")

# TAB 1: Predict Emotion
with tab1:
    st.subheader("🎯 Predict Emotion from Text")
    st.markdown(
        "If the sentence is longer than 7 words or if it has conflicting "
        "words the model may have trouble identifying the correct emotion."
    )
    input_text = st.text_area("Enter a sentence to analyze emotion:", height=150)
    logger.info("Text input rendered.")

    if st.button("Predict"):
        logger.info("Predict button clicked.")
        if input_text.strip():
            logger.info(f"User input received: {input_text}")
            try:
                response = requests.post(
                    f"{API_BASE}/predict", json={"text": input_text}
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Prediction result received: {result}")

                st.success("✅ Emotion Prediction Result")
                st.markdown(f"**Text:** {result.get('text', 'N/A')}")
                st.markdown(
                    f"**Top Emotion:** "
                    f"`{result.get('predicted_label', 'N/A').capitalize()}`"
                )

                emotions = result.get("emotions", [])
                if emotions:
                    logger.info("Rendering emotion score visualizations.")
                    emotions_df = pd.DataFrame(emotions)

                    st.subheader("📊 Emotion Scores - Bar Chart")
                    st.plotly_chart(
                        px.bar(
                            emotions_df,
                            x="label",
                            y="score",
                            color="label",
                            labels={
                                "label": "Emotion",
                                "score": "Confidence Score",
                            },
                            title="Emotion Confidence Scores",
                        ),
                        use_container_width=True,
                    )

                    st.subheader("🥧 Emotion Distribution - Pie Chart")
                    st.plotly_chart(
                        px.pie(
                            emotions_df,
                            names="label",
                            values="score",
                            title="Emotion Score Distribution",
                        ),
                        use_container_width=True,
                    )

                    st.subheader("📈 Emotion Score Distribution (Horizontal)")
                    fig = go.Figure(
                        go.Bar(
                            x=emotions_df["score"],
                            y=emotions_df["label"],
                            orientation="h",
                            marker=dict(
                                color=emotions_df["score"],
                                colorscale="Viridis",
                            ),
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Score",
                        yaxis_title="Emotion",
                        title="Emotion Score Distribution (Horizontal)",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    logger.warning("No emotion data received from API.")
                    st.warning("⚠️ No emotion data received.")
            except requests.RequestException as e:
                logger.error(f"API request failed: {e}")
                st.error(f"❌ API request failed: {e}")
            except Exception as e:
                logger.exception("Unexpected error during emotion prediction.")
                st.error(f"❌ Unexpected error: {e}")
        else:
            logger.warning("Predict clicked but no input provided.")
            st.warning("⚠️ Please enter some text before predicting.")

# TAB 2: Upload CSV
with tab2:
    st.subheader("📁 Upload CSV for Bulk Emotion Prediction")
    st.markdown(
        "Upload a `.csv` file with a column of sentences. The `.csv` "
        "file is expected to have 1 column called `text`."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    logger.info("CSV uploader rendered.")

    # Local session state to track last predicted CSV
    if "last_pred_csv" not in st.session_state:
        st.session_state["last_pred_csv"] = None
        st.session_state["csv_feedback_submitted"] = False

    if uploaded_file and st.button("Upload and Predict"):
        logger.info(f"CSV file uploaded: {uploaded_file.name}")
        try:
            files = {
                "file": (uploaded_file.name, uploaded_file, "text/csv"),
            }
            response = requests.post(f"{API_BASE}/upload-csv", files=files)
            response.raise_for_status()
            st.success("✅ File uploaded and processed successfully.")
            logger.info("CSV uploaded and processed.")

            # Download the result CSV from backend
            csv_response = requests.get(f"{API_BASE}/download-csv")
            csv_response.raise_for_status()
            # Save to bytes for download button and feedback
            st.session_state["last_pred_csv"] = csv_response.content
            st.session_state["csv_feedback_submitted"] = False

            df = pd.read_csv(io.BytesIO(st.session_state["last_pred_csv"]))
            st.subheader("📊 Prediction Results Preview")
            st.dataframe(df)

            counts = df["predicted_label"].value_counts().reset_index()
            counts.columns = ["Emotion", "Count"]
            st.subheader("📊 Distribution of Predicted Emotions")
            st.plotly_chart(
                px.bar(
                    counts,
                    x="Emotion",
                    y="Count",
                    title="Predicted Emotion Distribution",
                    color="Emotion",
                ),
                use_container_width=True,
            )

            st.download_button(
                label="⬇️ Download Mapped CSV",
                data=st.session_state["last_pred_csv"],
                file_name="csv_predictions.csv",
                mime="text/csv",
            )
            logger.info("CSV results displayed and download button created.")
        except requests.RequestException as e:
            logger.error(f"CSV API request failed: {e}")
            st.error(f"❌ Failed to process CSV file: {e}")
        except Exception as e:
            logger.exception("Error processing CSV file.")
            st.error(f"❌ Error: {e}")

    # Show feedback buttons **only if there is a last_pred_csv**
    if st.session_state["last_pred_csv"] and not st.session_state["csv_feedback_submitted"]:
        st.markdown("### Were these predictions correct overall?")
        col1, col2 = st.columns(2)

        def send_feedback(feedback_type):
            try:
                feedback_files = {
                    "csv_file": ("csv_predictions.csv", st.session_state["last_pred_csv"], "text/csv"),
                }
                feedback_data = {
                    "feedback": feedback_type,
                }
                resp = requests.post(
                    f"{API_BASE}/feedback-csv",
                    data=feedback_data,
                    files=feedback_files,
                )
                resp.raise_for_status()
                st.session_state["csv_feedback_submitted"] = True
                st.success(f"Feedback submitted as '{feedback_type}'. Thank you!")
                logger.info(f"CSV feedback '{feedback_type}' submitted.")
            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")
                logger.error(f"CSV feedback failed: {e}")

        if col1.button("👍 Good"):
            send_feedback("good")
        if col2.button("👎 Bad"):
            send_feedback("bad")

    elif st.session_state.get("csv_feedback_submitted", False):
        st.info("Feedback already submitted for this prediction batch.")
    elif not uploaded_file:
        st.info("ℹ️ No file selected yet.")
        logger.info("CSV upload button shown but no file selected.")

# TAB 3: YouTube Transcript
with tab3:
    st.subheader("📺 Get Transcript from YouTube URL")
    st.markdown(
        "Only the English language is supported for transcription and "
        "emotion mapping."
    )
    yt_url = st.text_input("Enter YouTube Video URL:")
    logger.info("YouTube URL input rendered.")

    if yt_url.strip() and st.button("Generate Transcript", key="yt_transcript_button"):
        logger.info(f"YouTube URL submitted: {yt_url}")
        try:
            response = requests.post(
                f"{API_BASE}/youtube-transcript", params={"url": yt_url}
            )
            response.raise_for_status()
            st.success("✅ Transcript CSV generated.")
            logger.info("YouTube transcript generated.")

            transcript_path = os.path.join("api", "youtube_transcript.csv")
            if os.path.exists(transcript_path):
                df_transcript = pd.read_csv(transcript_path)
                st.subheader("📄 Transcript Preview")
                st.dataframe(df_transcript)

                st.download_button(
                    label="⬇️ Download Transcript CSV",
                    data=df_transcript.to_csv(index=False).encode("utf-8"),
                    file_name="youtube_transcript.csv",
                    mime="text/csv",
                )
                logger.info("YouTube transcript displayed and download provided.")
            else:
                logger.error("Transcript file not found on server.")
                st.error("❌ Transcript file not found on server.")
        except requests.RequestException as e:
            logger.error(f"YouTube transcript API error: {e}")
            st.error("❌ Failed to generate transcript. Check the URL.")
        except Exception as e:
            logger.exception("Error processing YouTube transcript.")
            st.error(f"❌ Unexpected error: {e}")
    elif not yt_url.strip():
        st.info("ℹ️ Please enter a valid YouTube URL.")
        logger.info("Transcript button shown but URL is empty.")

# TAB 4: Upload Audio Transcript
with tab4:
    st.subheader("📤 Get Transcript from Uploaded Audio")
    st.markdown(
        "Only English audio is supported for transcription and emotion mapping. "
        "Accepted formats: MP3, WAV, M4A."
    )
    audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav", "m4a"])
    logger.info("Audio file input rendered.")

    if audio_file and st.button("Generate Transcript", key="audio_transcript_button"):
        logger.info(f"Audio file submitted: {audio_file.name}")
        try:
            # Guess MIME type
            mime_type, _ = mimetypes.guess_type(audio_file.name)
            files = {
                "file": (audio_file.name, audio_file.read(), mime_type or "application/octet-stream"),
            }

            response = requests.post(f"{API_BASE}/upload-audio-transcript", files=files)
            response.raise_for_status()
            st.success("✅ Transcript CSV generated.")
            logger.info("Audio transcript generated.")

            transcript_path = os.path.join("api", "uploaded_audio_transcript.csv")
            if os.path.exists(transcript_path):
                df_transcript = pd.read_csv(transcript_path)
                st.subheader("📄 Transcript Preview")
                st.dataframe(df_transcript)

                st.download_button(
                    label="⬇️ Download Transcript CSV",
                    data=df_transcript.to_csv(index=False).encode("utf-8"),
                    file_name="audio_transcript.csv",
                    mime="text/csv",
                )
                logger.info("Audio transcript displayed and download provided.")
            else:
                logger.error("Transcript file not found on server.")
                st.error("❌ Transcript file not found on server.")
        except requests.RequestException as e:
            logger.error(f"Audio transcript API error: {e}")
            st.error("❌ Failed to generate transcript from audio.")
        except Exception as e:
            logger.exception("Error processing audio transcript.")
            st.error(f"❌ Unexpected error: {e}")
    elif not audio_file:
        st.info("ℹ️ Please upload an audio file.")
        logger.info("Transcript button shown but no audio file provided.")
