from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

@st.cache_resource
def load_model():
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    embed_model=SentenceTransformer("all-MiniLM-L6-v2")

    reranker=CrossEncoder("cross-encoder/ms-marco-electra-base")
    
    model.eval()
    return model,tokenizer,embed_model,reranker

model,tokenizer,embed_model,reranker=load_model()