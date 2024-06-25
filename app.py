import os
import pathlib
import requests
import pymupdf4llm
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

 

load_dotenv()
qdrant_api_key = os.environ["QDRANT_API_KEY"]



embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_chat_model = ChatOpenAI(model="gpt-4o", streaming=True)


### I tried to get pdf file with PyPDFLoader, but for some reason it didn't work
### Also, converting pdf to md file seems to be the best option. I tried with loading
### pdf from dirct /data but i couldn't get right answer for Q2



# Loading pdf file
md_text = pymupdf4llm.to_markdown('./data/airbnb_file.pdf')

# save file as .md
pathlib.Path('./data/airbnb_file.md').write_bytes(md_text.encode())

# Loading .md file
text_loader = TextLoader('./data/airbnb_file.md')
docs = text_loader.load()


# Specs for chunk sizes
text_splitter = MarkdownTextSplitter(
    chunk_size = 700,
    chunk_overlap = 30
)

# Splitting docs in right chunk size
split_documents = text_splitter.split_documents(docs)


# Checking if collecion exists
qdrant_collection_check = requests.get(
                'https://7449e273-c21b-4960-aaec-ff1519d424c6.us-east4-0.gcp.cloud.qdrant.io:6333/collections/airbnb_pdf_v2/exists', 
                 headers={'api-key': qdrant_api_key}).json()['result']['exists']




if qdrant_collection_check:

    vectorstore = Qdrant.from_existing_collection(
            embedding_model,
            collection_name='airbnb_pdf_v2',
            url='https://7449e273-c21b-4960-aaec-ff1519d424c6.us-east4-0.gcp.cloud.qdrant.io:6333',
            api_key=qdrant_api_key,
            prefer_grpc=True,
            path=None,
    )
else:

    vectorstore = Qdrant.from_documents(
        split_documents,
        embedding_model,
        collection_name='airbnb_pdf_v2',
        location='https://7449e273-c21b-4960-aaec-ff1519d424c6.us-east4-0.gcp.cloud.qdrant.io:6333',
        api_key=qdrant_api_key,
        prefer_grpc=True,
    )

qdrant_retriever = vectorstore.as_retriever()


### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
CONTEXT:
{context}

QUERY:
{query}

You are a helpful assistant. You answer user questions based on provided context. 
If you can't answer the question with the provided context, say "I am only answering question about Airbnb document".
"""

### 2. CREATE PROMPT TEMPLATE

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "Airbnb Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = {"context": itemgetter("query") | qdrant_retriever, "query": itemgetter("query")}| rag_prompt | openai_chat_model | StrOutputParser()

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RlAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()