from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
import streamlit as st

quant_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

@st.cache_resource
def load_model():
    model=AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=quant_config
        )
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    embed_model=SentenceTransformer("all-MiniLM-L6-v2")

    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L4-v2")
    
    model.eval()
    return model,tokenizer,embed_model,reranker

model,tokenizer,embed_model,reranker=load_model()