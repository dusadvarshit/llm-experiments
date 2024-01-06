import streamlit as st
import langchain_helper as lch

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
model = SentenceTransformer('pinecone/mpnet-retriever-squad2')

# Initialize connection to pinecone
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

st.write("""
# AI Q&A
Ask me a question!
""")

query = st.text_input('Search!', '')

if query != "":
    xq = model.encode([query]).tolist()
    # get relevant context
    xc = index.query(xq, top_k=5, include_metadata=True)

    st.write(xc)

    # for context in xc['results'][0]['matches']:
    #     st.write(context['metadata']['text'])