# Q-AER

## Overview
This project is a Q&A chatbot built with Streamlit and LangChain, designed to answer questions about a pdf file provided using a knowledge base stored in a FAISS vector database.

## Getting Started
Follow these steps to set up the project on your local machine.

### Clone the Repository
```sh
git clone https://github.com/shhhhreya/Q-Aer
cd Q-Aer
```

### Create and Activate a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

#### On macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Running the Project
```sh
python main.py 
```


## Key Features
✅ **Upload pdf files** to create a knowledge base.  
✅ **Generate vector embeddings** using Hugging Face models.  
✅ **Retrieve relevant information** using a similarity-based search.  
✅ **Answer user queries** with a structured response.  
✅ **Interactive UI** powered by Streamlit.  

## Tech Stack
- **Python**  
- **Streamlit** (for the UI)  
- **LangChain** (for retrieval and response generation)  
- **FAISS** (for efficient vector search)  
- **OpenAI API** (for generating responses)  


