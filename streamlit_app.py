import streamlit as st
from app.inference import infer
st.title("📚 Class IX History RAG Assistant")
st.markdown("**Ask questions from *India and the Contemporary World-I* and get context-grounded answers powered by Qwen2.5.**")
with st.expander("ℹ️ About this project"):
    st.write("""
    Class IX History RAG Assistant is a Retrieval-Augmented Generation (RAG) chatbot built around the NCERT Class IX History textbook, India and the Contemporary World-I.

The system first converts the user's question into an embedding and retrieves relevant textbook passages using FAISS. A CrossEncoder reranker then ranks the retrieved passages by relevance. The selected context is passed to Qwen2.5-0.5B-Instruct, which generates the final answer.

Pipeline:
Question → Embedding → FAISS Retrieval → CrossEncoder Reranking → Qwen2.5 → Answer

The project is designed to demonstrate how retrieval, reranking, and instruction-following generation can be combined to build a lightweight, domain-specific AI assistant.

Note: The underlying language model is a compact 0.5B-parameter model, so occasional factual or contextual errors may occur. Answers should be verified against the textbook when accuracy is critical.
    """)
if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
user_input=st.chat_input("Ask a Question...")
if user_input:
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    answer=infer(user_input)
    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })
    st.rerun()
with st.sidebar:
    st.header("📖 About the Assistant")
    st.markdown("**Source**")
    st.markdown("""
    NCERT Class IX History
    *India and the Contemporary World-I*
    """)
    st.code("""
    🔎 Retrieve
Find relevant textbook passages

🎯 Rerank
Select the most relevant passages

🤖 Generate
Qwen2.5 generates the answer
    """)
    st.markdown("""
    **Built with**
    FAISS · Sentence Transformers · CrossEncoder · Qwen2.5 · Streamlit
    """)
    st.sidebar.markdown("### 📚 Example Questions")

    st.sidebar.markdown("""
    - What were the main causes of the French Revolution?
    - Who was Napoleon?
    - What were the consequences of the Russian Revolution?
    - What was India's role in the Second World War?
    - What is communism?
    """)
    if st.button("Clear Chat"):
        st.session_state.messages=[]
        st.rerun()
