import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Load environment
load_dotenv()

# Initialize models
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Helper: Extract video ID
def extract_youtube_video_id(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "v" in query_params:
        return query_params["v"][0]
    path_segments = parsed_url.path.strip("/").split("/")
    if parsed_url.netloc == "youtu.be":
        return path_segments[0]
    if len(path_segments) >= 2 and path_segments[0] in ["live", "shorts", "embed"]:
        return path_segments[1]
    return None

# Helper: Fetch transcript, translate & clean it using Groq
def fetch_transcript_cleaned(video_id, llm, chunk_size=1000):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        raw_text = " ".join([chunk["text"] for chunk in transcript])
        lang = "en"
    except NoTranscriptFound:
        try:
            available = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in available:
                try:
                    transcript = t.fetch()
                    lang = t.language_code
                    break
                except:
                    continue
            else:
                return None
            raw_text = " ".join([chunk["text"] for chunk in transcript])
        except:
            return None
    except (TranscriptsDisabled, Exception):
        return None

    # Translate to English if needed
    if lang != "en":
        chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        translated = []
        for chunk in chunks:
            prompt = f"Translate the following transcript to English:\n\n{chunk}"
            translated.append(str(llm.invoke(prompt)))
        raw_text = " ".join(translated)

    # Final grammar and clarity correction
    final_chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    cleaned = []
    for chunk in final_chunks:
        prompt = f"Please improve the clarity, fix grammar, and make this text more readable:\n\n{chunk}"
        cleaned.append(str(llm.invoke(prompt)))

    return " ".join(cleaned)

# UI setup
st.set_page_config("YouTube ChatBot", layout="wide")

# Layout: Title + Video side-by-side
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🎥 YouTube ChatBot")
    yt_link = st.text_input("🔗 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

with col2:
    if yt_link:
        video_id = extract_youtube_video_id(yt_link)
        if video_id:
            st.markdown(
                f"""
                <iframe width="80%" height="200"
                        src="https://www.youtube.com/embed/{video_id}"
                        frameborder="0" allowfullscreen>
                </iframe>
                """,
                unsafe_allow_html=True
            )

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None

# Process video
if yt_link:
    new_video_id = extract_youtube_video_id(yt_link)
    if new_video_id and new_video_id != st.session_state.video_id:
        st.session_state.video_id = new_video_id
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        with st.spinner("🔄 Fetching and processing transcript..."):
            transcript = fetch_transcript_cleaned(new_video_id, llm)
            if transcript:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                docs = splitter.create_documents([transcript])
                st.session_state.vectorstore = FAISS.from_documents(docs, embedding_model)
                st.success("✅ Transcript processed successfully!")
            else:
                st.error("❌ Failed to fetch transcript.")

# Show chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# User chat input
if yt_link and st.session_state.vectorstore:
    user_input = st.chat_input("💬 Ask a question about the video")
    if user_input:
        st.chat_message("user").markdown(user_input)
        with st.spinner("🤖 Thinking..."):
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

            prompt = PromptTemplate.from_template("""
            You are a helpful assistant.
            Use ONLY the provided context to answer the question.
            If not found, say "I am unable to understand the query."

            Context:
            {context}

            Chat History:
            {chat_history}

            Question: {question}
            """)

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt}
            )

            response = chain.run({"question": user_input})

        st.chat_message("bot").markdown(response)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))
