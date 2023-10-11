# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Using LLaMA 2.0, FAISS and LangChain for Question-Answering on Your Own Data
# MAGIC
# MAGIC <img src ='https://miro.medium.com/v2/resize:fit:720/format:webp/1*jH2Hmz8g0pR-wt7zc0knMw.png' style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC ## LangChain
# MAGIC A powerful, open-source framework designed to help you develop applications powered by a language model, particularly a large language model (LLM). The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. LangChain consists of multiple components from several modules.
# MAGIC
# MAGIC <img src ='https://miro.medium.com/v2/resize:fit:720/format:webp/1*DuixZuwaK6SPsvT_nfyBfQ.png' style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC ## Modules
# MAGIC - Prompts: This module allows you to build dynamic prompts using templates. It can adapt to different LLM types depending on the context window size and input variables used as context, such as conversation history, search results, previous answers, and more.
# MAGIC - Models: This module provides an abstraction layer to connect to most available third- party LLM APIs. It has API connections to ~40 public LLMs, chat and embedding models.
# MAGIC - Memory: This gives the LLMs access to the conversation history.
# MAGIC - Indexes: Indexes refer to ways to structure documents so that LLMs can best interact with them. This module contains utility functions for working with documents and integration to different vector databases.
# MAGIC - Agents: Some applications require not just a predetermined chain of calls to LLMs or other tools, but potentially to an unknown chain that depends on the user’s input. In these types of chains, there is an agent with access to a suite of tools. Depending on the user’s input, the agent can decide which — if any — tool to call.
# MAGIC - Chains: Using an LLM in isolation is fine for some simple applications, but many more complex ones require the chaining of LLMs, either with each other, or other experts. LangChain provides a standard interface for Chains, as well as some common implementations of chains for ease of use.
# MAGIC
# MAGIC ## FAISS (Facebook AI Similarity Search)
# MAGIC library for efficient similarity search and clustering of dense vectors. It can search multimedia documents (e.g. images) in ways that are inefficient or impossible with standard database engines (SQL). It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.
# MAGIC
# MAGIC <img src ='https://engineering.fb.com/wp-content/uploads/2017/03/GOcmDQEFmV52jukHAAAAAAAqO6pvbj0JAAAB.jpg' style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ##Process Flow
# MAGIC
# MAGIC <img src ='https://miro.medium.com/v2/resize:fit:720/format:webp/1*mdzyMcDhL0YLtgtfw7eaaA.png' style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC 1. **Initialize model pipeline**: initializing text-generation pipeline with Hugging Face transformers for the pretrained Llama-2-7b-chat-hf model.
# MAGIC 1. **Ingest data**: loading the data from arbitrary sources in the form of text into the document loader.
# MAGIC 1. **Split into chunks**: splitting the loaded text into smaller chunks. It is necessary to create small chunks of text because language models can handle limited amount of text.
# MAGIC 1. **Create embeddings**: converting the chunks of text into numerical values, also known as embeddings. These embeddings are used to search and retrieve similar or relevant documents quickly in large databases, as they represent the semantic meaning of the text.
# MAGIC 1. **Load embeddings into vector store**: loading the embeddings into a vector store i.e. “FAISS” in this case. Vector stores perform extremely well in similarity search using text embeddings compared to the traditional databases.
# MAGIC 1. **Enable memory**: combing chat history with a new question and turn them into a single standalone question is quite important to enable the ability to ask follow up questions.
# MAGIC 1. **Query data**: searching for the relevant information stored in vector store using the embeddings.
# MAGIC 1. **Generate answer**: passing the standalone question and the relevant information to the question-answering chain where the language model is used to generate an answer.

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %pip install langchain
# MAGIC %pip install pypdf
# MAGIC %pip install faiss-cpu
# MAGIC %pip install ctransformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" ## https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


# COMMAND ----------

# DBTITLE 1,PDF to Vector
# Load PDF file from data path
loader = DirectoryLoader('/Volumes/chlor_catalog/chlor_pdf_demo/pdfs',
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
documents = loader.load()

# Split text from PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                   model_kwargs={'device': 'cpu'})

# Build and persist FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('Volumes/chlor_catalog/chlor_pdf_demo/vector')

def load_retriever(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                   model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore.as_retriever()

# COMMAND ----------

from langchain.document_loaders import WebBaseLoader

web_links = ["https://www.databricks.com/","https://help.databricks.com","https://databricks.com/try-databricks","https://help.databricks.com/s/","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/release-notes/index.html","http://docs.databricks.com/ingestion/index.html","http://docs.databricks.com/exploratory-data-analysis/index.html","http://docs.databricks.com/data-preparation/index.html","http://docs.databricks.com/data-sharing/index.html","http://docs.databricks.com/marketplace/index.html"] 

loader = WebBaseLoader(web_links)
documents = loader.load()

# Split text from PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                   model_kwargs={'device': 'cpu'})

# Build and persist FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('Volumes/chlor_catalog/chlor_pdf_demo/vector_complete')

def load_retriever(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                   model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore.as_retriever()

# COMMAND ----------

# DBTITLE 1,Prompts
qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# COMMAND ----------

# DBTITLE 1,LLM
# File: llm.py
from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', # Location of downloaded GGML model
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 256, # max lenght to generate
                            'temperature': 0.01}) # The temperature to use for sampling. A higher temperature value typically makes the 
                                                  # output more diverse and creative but might also increase its likelihood of straying # from the context.



# COMMAND ----------

# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa

# Instantiate QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('Volumes/chlor_catalog/chlor_pdf_demo/vector_complete', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa

# COMMAND ----------

import argparse
import timeit

def main(input_str):
    start = timeit.default_timer() # Start timer

    # Setup QA object
    dbqa = setup_dbqa()
    
    # Parse input from argparse into QA object
    response = dbqa({'query': input_str})
    end = timeit.default_timer() # End timer

    # Print document QA response
    print(f'\nAnswer: {response["result"]}')
    print('='*50) # Formatting separator

    # Process source documents for better display
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        
    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")

# COMMAND ----------

main("What model can I use in Mlflow")

# COMMAND ----------

main("On which cloud does Databricks run?")

# COMMAND ----------

main("what is Databricks good at")

# COMMAND ----------

main("what are typical data preparation tasks")

# COMMAND ----------

main("what do you don't know")

# COMMAND ----------

main("what is the relationship between apache and databricks")

# COMMAND ----------

 import mlflow

dbqa = setup_dbqa()

with mlflow.start_run() as run:
    logged_model = mlflow.langchain.log_model(
        artifact_path="retrieval_qa",
        lc_model=dbqa,
        registered_model_name="chad_test_llm",
        loader_fn=load_retriever,
    )

# COMMAND ----------

import mlflow
logged_model = 'runs:/31f6c0052b774201863f87d78a19b3ec/retrieval_qa'

# Load model as a PyFuncModel.
loaded_model = mlflow.langchain.load_model(logged_model)
# Predict on a Pandas DataFrame.
loaded_model({'query': "what is Databricks good at"})['result']

# COMMAND ----------

import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config_file = read_yaml("config.yaml")

# COMMAND ----------



# COMMAND ----------

from mcli.sdk import predict

model_requests = {
    "inputs": [
        [
            "Represent the Science title:",
            "3D ActionSLAM: wearable person tracking in multi-floor environments"
        ]
    ]
}
predict('https://models.hosted-on.mosaicml.hosting/instructor-large/v1', model_requests)

# COMMAND ----------

# MAGIC %pip install --upgrade mosaicml-cli

# COMMAND ----------


from mcli.sdk import predict

prompt = """Below is an instruction that describes a photo. Write an adventure story linked to the caption.
### Instruction: Write a story for an image with the caption : there is a man riding a bike down a winding road.
### Response: """

# COMMAND ----------

# MAGIC %sh
# MAGIC # mcli init
# MAGIC mcli set api-key C7VuAm.Kq5k7cSEtJjTvVhDf.yi/YuFu9GxhDIdOURsHCgQ

# COMMAND ----------



# COMMAND ----------

predict('https://models.hosted-on.mosaicml.hosting/mpt-30b-instruct/v1', {"inputs": [prompt], "parameters": {"temperature": .2}})

# COMMAND ----------


