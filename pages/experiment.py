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

def card(id_val, title, context):
    return f"""
    <div class="card" style="margin:1rem;">
  <div class="card-body">
    <h5 class="card-title">{title}</h5>
    <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
    <p class="card-text">{context}</p>
  </div>
</div>
    """

st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True
)
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
        # st.write(context['metadata']['text'])
        st.markdown(card(
            context['id'],
            "Paragraph text",
            context['metadata']['text']

        ), unsafe_allow_html=True)