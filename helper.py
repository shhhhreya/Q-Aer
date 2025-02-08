# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini")
# instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vectordb_file_path = "faiss_index"

# def create_vector_db():
#     loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
#     data = loader.load()

#     vectordb = FAISS.from_documents(documents=data,
#                                     embedding=instructor_embeddings)

#     vectordb.save_local(vectordb_file_path)
#     print("____VB CREATED_____")

# #----------------------------------------------------smn wrong w this shit--------------------------------------------------
# def get_qa_chain():
#     # Load the vector database from the local folder
#     vectordb = FAISS.load_local(
#         vectordb_file_path, 
#         instructor_embeddings, 
#         allow_dangerous_deserialization=True
#     )

#     # Create a retriever for querying the vector database
#     retriever = vectordb.as_retriever(
#         search_type="similarity_score_threshold",
#         #pt score threshold!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         search_kwargs={
#             "score_threshold":0.5,
#             "k": 5  # Number of documents to retrieve
#         }
#     )

#     # Create a more comprehensive prompt template
#     prompt = ChatPromptTemplate.from_template("""
#     You are an expert assistant for Codebasics courses.

#     CONTEXT INSTRUCTIONS:
#     - Carefully analyze the provided context
#     - Extract the most relevant and precise information
#     - If no exact match is found, provide the closest relevant information

#     CONTEXT:
#     {context}

#     QUESTION: {input}

#     RESPONSE GUIDELINES:
#     - Be concise and informative
#     - Use information directly from the context
#     - If absolutely no relevant information is found, explain why
#     """)

#     # Create the document chain
#     document_chain = create_stuff_documents_chain(
#         llm,  # Language model 
#         prompt  # Prompt template
#     )

#     # Create the retrieval chain
#     retrieval_chain = create_retrieval_chain(
#         retriever,  # Document retriever
#         document_chain  # Document processing chain
#     )

#     return retrieval_chain
# #--------------------------------------------------------------------------------------------------------------------------------------------

# def main():
#     # Create vector database
#     create_vector_db()
#     get_qa_chain()
#     # Get QA chain
#     qa_chain = get_qa_chain()

#     # Example queries
#     queries = [
#         "Do you have javascript course?",
#         "What courses do you offer?",
#         "Tell me about your training programs"
#     ]

#     for query in queries:
#         print(f"\n{'='*50}")
#         print(f"Query: {query}")
#         res = qa_chain.invoke({"input": query})
#         print(res)


# if __name__ == "__main__":
#     main()
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    vectordb.save_local(vectordb_file_path)


#----------------------------------------------------done!-------------------------------------------------
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Use Maximum Marginal Relevance
        search_kwargs={
            "k": 5,  # Increase number of documents
            "fetch_k": 10,  # Fetch more documents before filtering
            "lambda_mult": 0.5  # Balance between diversity and relevance
        }
    )

    # Create a more flexible prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for Codebasics courses.

    CONTEXT INSTRUCTIONS:
    - Carefully analyze the provided context
    - Extract the most relevant and precise information
    - If no exact match is found, provide the closest relevant information

    CONTEXT:
    {context}

    QUESTION: {input}

    RESPONSE GUIDELINES:
    - Be concise and informative
    - Use information directly from the context
    - If absolutely no relevant information is found, explain why
    """)

    # Create the document chain
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever, 
        document_chain
    )

    # Uncomment to debug retrieval
    # debug_retrieval("Do you provide any job assistance")

    return retrieval_chain

# Optional: Add a main debugging function
def main():
    # Create vector database
    create_vector_db()

    # Get QA chain
    qa_chain = get_qa_chain()

    # Test queries
    queries = [
        "Do you provide any job assistance?",
        "What courses do you offer?",
        "Tell me about your training programs"
    ]

    # Run queries
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        res = qa_chain.invoke({"input": query})
        print(res)

        
if __name__ == "__main__":
    main()
