
import streamlit as st
import os
import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Video AI Assistant")
st.write(
    "Summarize YouTube videos, chat with their content, "
    "and get answers with video timestamps."
)


# --------------------------------------------------
# 2. LOAD GEMINI AND EMBEDDING MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2
    )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model, embedding


# --------------------------------------------------
# 3. EXTRACT YOUTUBE VIDEO ID
# --------------------------------------------------

def extract_video_id(video_input):

    video_input = video_input.strip()

    # If user enters only video ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_input):
        return video_input

    patterns = [
        r"(?:youtube\.com/watch\?.*?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:

        match = re.search(pattern, video_input)

        if match:
            return match.group(1)

    return None


# --------------------------------------------------
# 4. FETCH TRANSCRIPT WITH TIMESTAMPS
# --------------------------------------------------

def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en"]
    )

    return list(transcript)


# --------------------------------------------------
# 5. CONVERT SECONDS INTO HH:MM:SS
# --------------------------------------------------

def format_timestamp(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# --------------------------------------------------
# 6. SPLIT TRANSCRIPT WHILE PRESERVING TIMESTAMPS
# --------------------------------------------------

def split_transcript(transcript, chunk_size=500):

    documents = []

    current_text = []
    current_length = 0

    chunk_start = None
    chunk_end = None

    for item in transcript:

        text = item.text.strip()

        if not text:
            continue

        start = float(item.start)
        end = start + float(item.duration)

        if chunk_start is None:
            chunk_start = start

        # Keep track of the ending timestamp
        chunk_end = end

        current_text.append(text)
        current_length += len(text)

        if current_length >= chunk_size:

            page_content = " ".join(current_text)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "start": chunk_start,
                        "end": chunk_end
                    }
                )
            )

            # Reset for the next chunk
            current_text = []
            current_length = 0
            chunk_start = None
            chunk_end = None

    # Save remaining transcript text
    if current_text:

        documents.append(
            Document(
                page_content=" ".join(current_text),
                metadata={
                    "start": chunk_start,
                    "end": chunk_end
                }
            )
        )

    return documents


# --------------------------------------------------
# 7. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vectorstore(chunks, embedding):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vectorstore


# --------------------------------------------------
# 8. GENERATE INITIAL SUMMARY
# --------------------------------------------------

def generate_summary(chunks, model):

    map_prompt = PromptTemplate(
        template="""
You are an expert video summarizer.

Summarize the important information from this section
of the YouTube video.

Include important facts, explanations, examples,
and useful insights.

Transcript:
{text}

Summary:
""",
        input_variables=["text"]
    )

    reduce_prompt = PromptTemplate(
        template="""
You are an expert YouTube video summarizer.

Combine the following section summaries.

Return ONLY the 5 most important points.

Rules:
1. Generate up to 5 important points.
2. Do not repeat information.
3. Use simple and understandable English.
4. Do not invent information.
5. Focus on the most valuable information.

Section summaries:
{summaries}

Final summary:
""",
        input_variables=["summaries"]
    )

    parser = StrOutputParser()

    map_chain = map_prompt | model | parser
    reduce_chain = reduce_prompt | model | parser

    summaries = []

    progress = st.progress(0)

    for index, chunk in enumerate(chunks):

        result = map_chain.invoke({
            "text": chunk.page_content
        })

        summaries.append(result)

        progress.progress(
            (index + 1) / len(chunks)
        )

    progress.empty()

    combined_summaries = "\n".join(summaries)

    final_summary = reduce_chain.invoke({
        "summaries": combined_summaries
    })

    return final_summary, combined_summaries


# --------------------------------------------------
# 9. GENERATE ADDITIONAL SUMMARY POINTS
# --------------------------------------------------

def generate_additional_points(
    combined_summaries,
    previous_points,
    model
):

    prompt = PromptTemplate(
        template="""
You are an expert YouTube video summarizer.

Generate up to 5 NEW important points from the video.

Previously generated points:
{previous_points}

Video section summaries:
{summaries}

Instructions:
1. Do not repeat previous points.
2. Find different useful information.
3. Include new facts, examples, explanations or insights.
4. Do not invent information.
5. Use simple English.
6. If no new information exists, say:
   "No more important points are available."
7. Return only numbered points.

Additional points:
""",
        input_variables=[
            "previous_points",
            "summaries"
        ]
    )

    chain = prompt | model | StrOutputParser()

    return chain.invoke({
        "previous_points": previous_points,
        "summaries": combined_summaries
    })


# --------------------------------------------------
# 10. CHAT WITH VIDEO USING FAISS
# --------------------------------------------------

def chat_with_video(question, vectorstore, model):

    # Search for the 4 most relevant transcript chunks
    relevant_chunks = vectorstore.similarity_search(
        question,
        k=4
    )

    context_parts = []

    timestamp_links = []

    for chunk in relevant_chunks:

        start = float(chunk.metadata["start"])
        end = float(chunk.metadata["end"])

        start_time = format_timestamp(start)
        end_time = format_timestamp(end)

        context_parts.append(
            f"""
Timestamp: {start_time} to {end_time}

Transcript:
{chunk.page_content}
"""
        )

        # Create clickable YouTube timestamp link
        timestamp_links.append({
            "start": start,
            "end": end,
            "start_time": start_time,
            "end_time": end_time
        })

    context = "\n".join(context_parts)

    prompt = PromptTemplate(
        template="""
You are a helpful YouTube video assistant.

Answer the user's question using ONLY the transcript
context provided below.

Important rules:
1. Give a clear and understandable answer.
2. Do not use outside knowledge.
3. If the answer is not available in the transcript,
   clearly say that the video does not provide enough
   information to answer.
4. Mention the relevant timestamp in your answer.
5. Do not invent timestamps.

Transcript context:
{context}

User question:
{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    chain = prompt | model | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, timestamp_links


# --------------------------------------------------
# 11. SESSION STATE
# --------------------------------------------------

default_values = {
    "summary_generated": False,
    "summary": "",
    "combined_summaries": "",
    "previous_points": "",
    "additional_points": [],
    "transcript_text": "",
    "video_id": None,
    "vectorstore": None,
    "chunks": [],
    "chat_history": [],
    "finished_summary": False
}

for key, value in default_values.items():

    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 12. SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "Video Summary",
        "Chat with Video"
    ]
)


# --------------------------------------------------
# 13. VIDEO INPUT
# --------------------------------------------------

video_input = st.text_input(
    "Enter YouTube video URL or Video ID",
    placeholder="https://www.youtube.com/watch?v=..."
)


# --------------------------------------------------
# 14. GENERATE SUMMARY BUTTON
# --------------------------------------------------

if st.button("Generate Summary", type="primary"):

    if not video_input.strip():

        st.warning("Please enter a YouTube URL or video ID.")

    else:

        video_id = extract_video_id(video_input)

        if not video_id:

            st.error("Invalid YouTube URL or video ID.")

        else:

            try:

                with st.spinner("Loading Gemini and embedding models..."):

                    model, embedding = load_models()

                with st.spinner("Fetching YouTube transcript..."):

                    transcript = get_transcript(video_id)

                if not transcript:

                    st.error("No transcript was found.")

                else:

                    with st.spinner("Splitting transcript..."):

                        chunks = split_transcript(transcript)

                    if not chunks:

                        st.error("Transcript could not be split.")

                    else:

                        with st.spinner("Creating FAISS vector database..."):

                            vectorstore = create_vectorstore(
                                chunks,
                                embedding
                            )

                        with st.spinner("Generating video summary..."):

                            final_summary, combined_summaries = (
                                generate_summary(chunks, model)
                            )

                        # Reset old session data
                        st.session_state.summary = final_summary
                        st.session_state.combined_summaries = combined_summaries
                        st.session_state.previous_points = final_summary
                        st.session_state.additional_points = []
                        st.session_state.video_id = video_id
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chunks = chunks
                        st.session_state.chat_history = []
                        st.session_state.summary_generated = True
                        st.session_state.finished_summary = False

                        st.success("Video summary generated successfully!")

            except Exception as e:

                st.error(f"Something went wrong: {e}")


# --------------------------------------------------
# 15. VIDEO SUMMARY PAGE
# --------------------------------------------------

if page == "Video Summary":

    if st.session_state.summary_generated:

        st.subheader("🎥 Video")

        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.video_id}"
        )

        st.divider()

        st.subheader("📝 Initial Summary")

        st.markdown(st.session_state.summary)

        st.divider()

        # Display additional points
        if st.session_state.additional_points:

            st.subheader("➕ Additional Summary Points")

            for index, points in enumerate(
                st.session_state.additional_points,
                start=1
            ):

                st.markdown(f"**Additional Batch {index}**")
                st.markdown(points)

        st.divider()

        # Ask whether user wants more points
        if not st.session_state.finished_summary:

            st.write("Do you want more important points?")

            col1, col2 = st.columns(2)

            with col1:

                if st.button("Yes, show more"):

                    try:

                        model, embedding = load_models()

                        with st.spinner("Finding more important points..."):

                            new_points = generate_additional_points(
                                st.session_state.combined_summaries,
                                st.session_state.previous_points,
                                model
                            )

                        st.session_state.additional_points.append(
                            new_points
                        )

                        st.session_state.previous_points += (
                            "\n" + new_points
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(f"Error: {e}")

            with col2:

                if st.button("No, finish"):

                    st.session_state.finished_summary = True

                    st.rerun()

        else:

            st.success("You have finished exploring the summary.")

    else:

        st.info("Enter a YouTube video URL above and generate its summary.")


# --------------------------------------------------
# 16. CHAT WITH VIDEO PAGE
# --------------------------------------------------

if page == "Chat with Video":

    st.subheader("💬 Chat with Your YouTube Video")

    if st.session_state.vectorstore is None:

        st.info(
            "First open Video Summary and generate a summary "
            "for a video. Then you can chat with it."
        )

    else:

        st.write(
            "Ask questions about the video. "
            "Answers will include relevant timestamps."
        )

        # Display previous chat messages
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                # Show timestamp links for assistant answers
                if message["role"] == "assistant":

                    for link in message.get("links", []):

                        timestamp_url = (
                            f"https://www.youtube.com/watch?v="
                            f"{st.session_state.video_id}"
                            f"&t={int(link['start'])}s"
                        )

                        st.markdown(
                            f"[▶ {link['start_time']} - "
                            f"{link['end_time']}]({timestamp_url})"
                        )

        # Chat input
        question = st.chat_input(
            "Ask something about this video..."
        )

        if question:

            # Display user question
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            try:

                model, embedding = load_models()

                with st.chat_message("assistant"):

                    with st.spinner("Searching the video and generating answer..."):

                        answer, timestamp_links = chat_with_video(
                            question,
                            st.session_state.vectorstore,
                            model
                        )

                    st.markdown(answer)

                    st.markdown("**Relevant video timestamps:**")

                    for link in timestamp_links:

                        timestamp_url = (
                            f"https://www.youtube.com/watch?v="
                            f"{st.session_state.video_id}"
                            f"&t={int(link['start'])}s"
                        )

                        st.markdown(
                            f"[▶ {link['start_time']} - "
                            f"{link['end_time']}]({timestamp_url})"
                        )

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "links": timestamp_links
                })

            except Exception as e:

                st.error(f"Chat error: {e}")
