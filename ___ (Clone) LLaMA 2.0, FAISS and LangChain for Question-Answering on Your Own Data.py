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
# MAGIC %pip install einop
# MAGIC %pip install xformers
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


# COMMAND ----------

from PIL import Image


# load image from the IAM dataset
image = Image.open("/Volumes/chlor_catalog/flickr_llm/photos/2023_Teracea_236").convert("RGB")

image

# COMMAND ----------

from transformers import pipeline

captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-large")
title = captioner(image)
title[0]['generated_text']


# COMMAND ----------

title = captioner(image)
title[0]['generated_text']

# COMMAND ----------

from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
model_name = "aspis/gpt2-genre-story-generation"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# Input should be of format "<BOS> <Genre token> Optional starter text"
input_prompt = "<BOS> <adventure> there is a field with a few trees and a few clouds"

# COMMAND ----------

context = "<BOS> <adventure> " + title[0]['generated_text']
story = generator(context, max_length=200, do_sample=True,
               repetition_penalty=1.5, temperature=1.2, 
               top_p=0.95, top_k=50)
print(story)

# COMMAND ----------

context

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


