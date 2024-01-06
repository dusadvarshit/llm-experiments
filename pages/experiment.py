import streamlit as st

## Handling some error due to conda-macos conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pages.submodules.create_index import *

import pinecone
from sentence_transformers import SentenceTransformer


##

env_vars = dotenv_values(".env")

# Access specific variables
pinecone_api = env_vars.get("PINECONE_API_KEY")
pinecone_env = env_vars.get("PINECONE_ENV")

## 
@st.cache_resource
def init_retriever():
    return SentenceTransformer('pinecone/mpnet-retriever-squad2')

# Initialize connection to pinecone
def init_pinecone():
    pinecone.init(
        api_key=pinecone_api,
        environment=pinecone_env
    )

    # # check if index already exists and if not we create it
    if 'qa-index' not in pinecone.list_indexes():
        pinecone.create_index(
        name='qa-index',
        dimension=model.get_sentence_embedding_dimension()
        )

    # # connect to the index
    index = pinecone.Index('qa-index')
    return index


st.write("""
# AI Q&A
Ask me a question!
""")

model = init_retriever()
index = init_pinecone()

query = st.text_input('Search!', '')

if query != "":
    xq = model.encode([query]).tolist()
    # get relevant context
    xc = index.query(xq, top_k=5, include_metadata=True)

    for context in xc['matches']:
        st.write(context['metadata']['text'])