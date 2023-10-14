import chainlit as cl
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

import config
from db import create_vector_db

@cl.on_chat_start
def start():
    await cl.Message(content="Welcome to Chat-GPT Q&A!").send()
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Upload text/PDF file to chat with it",
            accept=["text/plain", "application/pdf"],
            max_size_mb=config.MAX_SIZE_MB,
            timeout=config.TIMEOUT
        ).send()
    file = files[0]

    msg = cl.Message(
        content=f"Reading in {file.name}"
    )
    await msg.send()

    doc_search = await cl.make_async(create_vector_db)(file)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            temperature=config.TEMPERATURE,
            streaming=True
        ),
        chain_type="stuff",
        retriever=doc_search.as_retriever(max_tokens_limit=config.MAX_TOKENS_LIMIT)
    )

    msg.content = f"Finished processing {file.name}. You may ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)