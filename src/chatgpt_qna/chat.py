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

@cl.on_message
async def main(msg):
    chain = cl.user_session.get("chain")
    callback = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback.answer_reached=True

    response = await chain.acall(msg, callbacks=[callback])
    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    docs = cl.user_session.get("docs")
    metadata = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadata]

    if sources:
        found_sources = []
        for s in sources.split(","):
            s_name = s.strip().replace(".", "")
            if s_name in all_sources:
                s_idx = all_sources.index(s_name)
                text = docs[s_idx].page_content
                found_sources.append(s_name)
                source_elements.append(cl.Text(
                    content=text,
                    name=s_name
                ))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += f"\nNo sources found."

    if callback.has_streamed_final_answer:
        callback.final_stream.elements = source_elements
        await callback.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()