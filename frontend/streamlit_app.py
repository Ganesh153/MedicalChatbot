import streamlit as st
from components.upload_pdf import render_upload_pdf
from components.chat_ui import render_chat
from components.history_download import render_history_download

st.set_page_config(page_title="AI Medical Assistant", layout="wide")
st.title("🩺 Medical Assistant Chatbot")

render_upload_pdf()
render_chat()
render_history_download()