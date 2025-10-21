from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables (make sure your .env file has HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # ✅ Chat-compatible model
)

userinput = st.text_input("enter the prompt")
if st.button("generate response"):
    model = ChatHuggingFace(llm=llm)
    result = model.invoke(userinput)
    st.write(result.content)

