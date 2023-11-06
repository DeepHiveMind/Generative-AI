from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



def load_and_split_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    pdf_data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_data = text_splitter.split_documents(pdf_data)
    return split_data




def create_and_persist_vectDB(splitData, embeddings, collection_name, local_directory):
    vectDB = Chroma.from_documents(
        splitData, embeddings, collection_name=collection_name, persist_directory=local_directory
    )
    vectDB.persist()
    return vectDB



def create_chat_qa(cohere_api_key, vectDB):
    # Create a ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create a ConversationalRetrievalChain
    chatQA = ConversationalRetrievalChain.from_llm(
        Cohere(cohere_api_key=cohere_api_key), vectDB.as_retriever(), memory=memory
    )
    return chatQA
