import tempfile
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
import chainlit as cl
from chainlit.types import AskFileResponse

from .config import chat_settings

load_dotenv()

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=chat_settings.CHUNK_SIZE,
    chunk_overlap=chat_settings.CHUNK_OVERLAP,
    )
EMBEDDINGS = OpenAIEmbeddings()
FILE_LOADERS = {
    "text/plain": TextLoader,
    "application/pdf": PyPDFLoader
}

def create_docs(file: AskFileResponse) -> list[Document]:
    """
    Create documents from given file.

    Args
    ----
        file: AskFileResponse
            File from which to create document corpus.
    Returns
    -------
        list[Document]
            File split into a list of documents
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(file.content)
        loader = FILE_LOADERS[file.type](tmp.name)

        docs = loader.load_and_split(
            text_splitter=TEXT_SPLITTER
        )
        for i,doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
    return docs

def create_vector_db(file: AskFileResponse) -> Chroma:
    """
    Create vector db from a given text file.

    Args
    ----
        file: AskFileResponse
            File from which to create document corpus.
    Returns
    -------
        Chroma
            Chroma vectorstore containing embedded documents.
    """
    docs = create_docs(file)

    # Add documents to chainlit session data
    cl.user_session.set("docs", docs)

    doc_search = Chroma.from_documents(
        docs,
        embedding=EMBEDDINGS
    )

    return doc_search
