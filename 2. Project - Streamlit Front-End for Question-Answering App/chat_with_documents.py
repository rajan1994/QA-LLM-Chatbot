"""
This code snippet is a Streamlit application that allows users to upload a file, split it into chunks, create embeddings using OpenAI, and ask questions about the content of the file to get answers. 

The main functions in this code snippet are:
- `load_document`: Loads a document from a file.
- `chunk_data`: Splits the document into chunks.
- `create_embeddings`: Creates embeddings using OpenAI and saves them in a Chroma vector store.
- `ask_and_get_answer`: Asks a question and gets an answer using the vector store.
- `calculate_embedding_cost`: Calculates the embedding cost using tiktoken.
- `clear_history`: Clears the chat history from the Streamlit session state.

The code also includes a Streamlit interface with widgets for file uploading, chunk size selection, question input, and displaying the answer and chat history.

To use the code, the user needs to provide an OpenAI API key, upload a file, select the chunk size and number of answers to return, and ask a question. The code will then process the file, create embeddings, and display the answer and chat history.

Note: This code snippet assumes the presence of certain dependencies and environment variables, such as the OpenAI API key and the necessary packages for document loading, text splitting, and embedding creation.
"""

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_document(file):
    """
    Load a document from a file.
    Supported file formats are:
    - PDF
    - DOCX
    - TXT
    :param file: path to file
    :return: LangChain Document or None if file format is not supported.
    """
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    Split data into chunks.
    :param data: LangChain Document
    :param chunk_size: size of each chunk
    :param chunk_overlap: overlap between chunks
    :return: list of chunks. Each chunk is a LangChain Document.
    :raise ValueError: if chunk_size is smaller than chunk_overlap.
    :raise ValueError: if chunk_overlap is larger than chunk_size.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    """
    Create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store.
    :param chunks: list of LangChain Document
    :return: vector store
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    """
    Ask a question and get an answer using a vector store.
    :param vector_store: vector store
    :param q: question
    :param k: number of answers to return
    :return: answer or None if no answer is found.
    """
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


def calculate_embedding_cost(texts):
    """
    Calculate embedding cost using tiktoken.
    :param texts: list of LangChain Document
    :return: total tokens, total cost in USD
    """
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    """
    Clear the chat history from streamlit session state.
    :return: None.
    """
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # setting the title and subtitle of the app
    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')

    # Create a sidebar with a text input widget and a button widget
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget (slider)
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget (slider)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget 
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

