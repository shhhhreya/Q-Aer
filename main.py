import streamlit as st
from helper import get_qa_chain, create_vector_db
from langchain_openai import ChatOpenAI

# Page configuration
st.set_page_config(page_title="Codebasics Q&A", page_icon="🌱")

# Title
st.title("Codebasics Q&A 🌱")

uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

# Create Knowledgebase Button
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner('Creating Vector Database...'):
        create_vector_db()
    st.success('Knowledgebase Created Successfully!')

# Question Input
question = st.text_input("Ask a question about Codebasics courses:")

# Response Generation
if question:
    try:
        # Show loading spinner
        with st.spinner('Generating response...'):
            # Get QA Chain
            qa_chain = get_qa_chain()
            
            # Invoke the chain
            response = qa_chain.invoke({"input": question})
            
            # Display Answer
            st.header("Answer")
            st.write(response["answer"])
            
            # Optional: Display Source Documents
            with st.expander("Source Documents"):
                for doc in response['context']:
                    st.write(doc)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info(
    "This is a Q&A chatbot for Codebasics courses. "
    "Ask questions and get instant answers from our knowledge base."
)
