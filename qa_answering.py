import streamlit as st
import pdfplumber
from transformers import pipeline

st.title("PDF Question-Answering System")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Read the PDF file
    with pdfplumber.open(uploaded_file) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

    # NLP Pipeline for Question Answering
    qa_pipeline = pipeline("question-answering")

    # User Input
    user_question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if user_question:
            answer = qa_pipeline(question=user_question, context=pdf_text)
            st.write(f"Answer: {answer['answer']}")
        else:
            st.warning("Please enter a question.")

