import streamlit as st
import langchain_helper as lch

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pages.submodules.create_index import *

import pinecone
from sentence_transformers import SentenceTransformer

## Handling some error due to conda-macos conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
##

env_vars = dotenv_values(".env")

# Access specific variables
pinecone_api = env_vars.get("PINECONE_API_KEY")
pinecone_env = env_vars.get("PINECONE_ENV")

## 
os.write(1,b'Something was executed.\n')
model = SentenceTransformer('pinecone/mpnet-retriever-squad2')
os.write(1,b'Something was finished.\n')

# Initialize connection to pinecone
# pinecone.init(
#     api_key=pinecone_api,
#     environment=pinecone_env
# )

# # check if index already exists and if not we create it
# if 'qa-index' not in pinecone.list_indexes():
#     pinecone.create_index(
#         name='qa-index',
#         dimension=model.get_sentence_embedding_dimension()
#     )

# # connect to the index
# index = pinecone.Index('qa-index')

st.write("""
# AI Q&A
Ask me a question!
""")

query = st.text_input('Search!', '')

