import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
# =======================
# Configure Gemini API Key
# =======================

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# =======================
# Load Vectorizer, Chunks, and Vectors
# =======================
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

with open("tfidf_vectors.pkl", "rb") as f:
    tfidf_vectors = pickle.load(f)

# =======================
# Gemini Prompt Function
# =======================
def get_gemini_response(user_input, retrieved_chunk):
    guide = """
    You are a helpful Q&A chatbot. Answer the user's question based on the given context.
    If the question is unrelated to the context, say "Sorry, I don't have information about that."
    Be concise, friendly, and informative.
    """

    prompt = f"""
    Guide: {guide}
    Context: {retrieved_chunk}
    User: {user_input}
    Bot:"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini API: {e}"

# =======================
# Streamlit App UI
# =======================
st.set_page_config(page_title="SmartChat", page_icon="🤖")
# Use columns to center the image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("image.png", use_column_width=True)
st.title("🤖 CSN SmartCity Chatbot")
st.markdown("Ask questions based on the knowledge base.")

# Maintain chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message on right
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Vectorize query and compute similarity
    query_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(query_vector, tfidf_vectors)
    best_match_idx = np.argmax(similarities)
    retrieved_chunk = text_chunks[best_match_idx]

    # Get Gemini response
    response = get_gemini_response(user_input, retrieved_chunk)

    # Show bot response on left
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
