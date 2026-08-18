from app.model import model,tokenizer,embed_model,reranker
import faiss
import pickle
import numpy as np

with open("docs.pkl","rb") as f:
    docs=pickle.load(f)
index=faiss.read_index("data/index.faiss")
def infer(user_input):
  model.eval()
  top_k=5
  query_embedding=embed_model.encode([user_input])
  query_embedding=(query_embedding).astype('float32')
  faiss.normalize_L2(query_embedding)
  distances,indices=index.search(query_embedding,top_k)
  contexts=[docs[i] for i in indices[0]]
  pairs=[[user_input,doc] for doc in contexts]
  scores=reranker.predict(pairs)
  ranked_indices=np.argsort(scores)[::-1]
  top_contexts=[contexts[i] for i in ranked_indices[:2]]
  context="\n\n".join(top_contexts)
  messages = [
    {
        "role": "system",
        "content":"""
You are a Class IX History assistant.

Answer the question using ONLY information explicitly stated in the supplied context.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not infer facts that are not explicitly supported by the context.
3. If the context does not contain enough information to answer the question,
   say that the information is not available in the provided material.
4. Do not confuse events that happened before the French Revolution
   with events that happened during or after the Revolution.
5. Answer directly and concisely.
6. If the question asks for reasons, give only reasons that are supported
   by the supplied context.
7. Follow the user's requested answer format exactly.
8. If the user asks for one word, give only one word.
9. If the user asks for numbered points, use numbered points.
10. Do not add explanations, facts, or examples that are not supported
    by the supplied context.
"""

     },
    {
        "role":"user",
        "content":f"""
        context:
        {context}

        Question:
        {user_input}
        """
    }
]
  inputs = tokenizer.apply_chat_template(
  messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
  )
  outputs=model.generate(
      **inputs,
      max_new_tokens=200
  )
  length=inputs["input_ids"].shape[-1]
  response=tokenizer.decode(outputs[0][length:],skip_special_tokens=True)
  return response.strip()