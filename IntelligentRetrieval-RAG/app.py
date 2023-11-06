import app as st
import os
import time
from config import api_key,collection_name,local_directory
from utils import  load_and_split_pdf
from utils import create_and_persist_vectDB, create_chat_qa
from langchain.embeddings import CohereEmbeddings

st.header("Automated and Intelligent Knowledge Retrieval System")

# Initialize the necessary components
cohere_api_key = api_key['api_key']                           #Fetching the API key
collection_name = collection_name['collection_name']
local_directory = local_directory['local_directory']
embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=cohere_api_key)



options = st.selectbox("Select the type of data source",
                        options=['PDF','Data source directory'])
z=os.getcwd()

#ask a query based on options of data sources
if options == 'PDF':
    st.write("Choose .pdf from your local machine")
    pdf2 = st.file_uploader("Choose pdf file:", type="pdf")
    if pdf2 is not None:
        x=pdf2.name
        a = os.path.join(z, x)
        st.write("PDF is uploading.....")
        time.sleep(5)
        st.write("PDF file uploaded successfully!")
        splitData = load_and_split_pdf(a)

        # Create and persist the Chroma vector store
        vectDB=create_and_persist_vectDB(splitData, embeddings, collection_name, local_directory)

        # # Initialize the chat QA system
        chatQA = create_chat_qa(cohere_api_key,vectDB)
        chat_history = []
        qry=""
        st.write("Ask your questions below:")
        qry = st.text_input("Question:")
        
        if st.button("Ask"):
            if qry != 'exit':
                response = chatQA({"question":qry, "chat_history": chat_history})
                st.write("Answer:", response["answer"])

elif options == 'Data source directory':
    # Folder Selection
    folder_path = st.file_uploader("Select multiple PDF:",accept_multiple_files=True)
    if folder_path:
            if folder_path is not None:
                    # a=folder_path.name
                    st.write("Files uploaded successfully!")
                    vectDB=Chroma(persist_directory="\Large_aiml_vect_embedding_cohere",collection_name="Large_aiml_vect_embedding_cohere",embedding_function=embeddings)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    chatQA = ConversationalRetrievalChain.from_llm(
                        Cohere(cohere_api_key=cohere_api_key), vectDB.as_retriever(), memory=memory
                    )
                    chat_history = []
                    qry=""
                    st.write("Ask your questions below:")
                    qry = st.text_input("Question:")
                    
                    if st.button("Ask"):
                        if qry != 'exit':
                            response = chatQA({"question":qry, "chat_history": chat_history})
                            st.write("Answer:", response["answer"])
            else:
                 st.error("Invalid folder path. Please enter a valid path.")

