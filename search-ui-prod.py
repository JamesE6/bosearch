from pinecone import Pinecone
import cohere
import streamlit as st
import pandas as pd
import requests, uuid, json

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Tibetan Corpus Search")
    password = st.text_input("Enter access code:", type="password")
    if st.button("Continue"):
        if password == st.secrets["ACCESS_CODE"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect code.")
    st.stop()

# Add your key and endpoint
co = cohere.Client(st.secrets['COHERE_API_KEY'])
pinecone_API_key = st.secrets['PINECONE_API_KEY']
azure_API_key = st.secrets["AZURE_API_KEY"]
endpoint = "https://api.cognitive.microsofttranslator.com"

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "uksouth"

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'bo',
    'to': 'en'
}

headers = {
    'Ocp-Apim-Subscription-Key': azure_API_key,
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

def translate_text(input_str):
    body = [{'text': input_str}]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

pc = Pinecone(api_key=pinecone_API_key)

index_name = 'diverge-test'
index = pc.Index(index_name)


def normalize_matches(matches):
    """Turn Pinecone matches (ScoredVector or dict) into plain dict rows."""
    rows = []
    for m in matches:
        # New client: m is a ScoredVector
        if not isinstance(m, dict):
            md = getattr(m, "metadata", {}) or {}
            row = {
                **md,
                "ID": getattr(m, "id", None),
                "Score": getattr(m, "score", None),
            }
        else:
            # Older dict-style matches
            md = m.get("metadata", {}) or {}
            row = {
                **md,
                "ID": m.get("id"),
                "Score": m.get("score"),
            }
        rows.append(row)
    return rows


def render_results_table(matches, translate=False):
    data = normalize_matches(matches)
    df = pd.DataFrame(data)

    if translate:
        # Guard against missing keys just in case
        if 'text' in df.columns:
            df['Translated Text'] = df['text'].apply(lambda x: translate_text(x))
        if 'title' in df.columns:
            df['Translated Title'] = df['title'].apply(lambda x: translate_text(x))

    return df


publications = ["All", "Tibet Daily"]
selected_publication = st.selectbox("Select a publication:", publications)

# Date range sliders
selected_years = st.slider("Select a year range:", 2020, 2023, (2021, 2022))
selected_months = st.slider("Select a month range:", 1, 12, (1, 12))

# Constructing filters based on the inputs
filters = {}
if selected_publication != "All":
    filters["publication"] = {"$eq": selected_publication}

filters["year"] = {'$in': [str(i) for i in range(selected_years[0], selected_years[1] + 1)]}
filters["month"] = {'$in': [str(i) for i in range(selected_months[0], selected_months[1] + 1)]}


def index_query(input_string, top_k=25, filters=None):
    # Embed the input string
    xq = co.embed(
        texts=[input_string],
        model='embed-multilingual-v2.0',
        input_type='search_query',
        truncate='END'
    ).embeddings

    # Cohere returns list of embeddings. Take the first one.
    xq = xq[0]

    query_params = {
        "vector": xq,
        "top_k": top_k,
        "include_metadata": True,
    }

    if filters:
        query_params['filter'] = filters

    return index.query(**query_params)


input_text = st.text_input("Enter your query text: ")

if input_text:
    translate_query = st.checkbox("Toggle translated query", value=False)
    if translate_query:
        st.write(translate_text(input_text))

    num_results = st.slider("Number of results to display", min_value=1, max_value=100, value=10)
    pretty_print = st.checkbox("Toggle formatted display", value=False)

    results = index_query(input_text, top_k=num_results, filters=filters)
    matches = results.matches  

    if pretty_print:
        translator = st.checkbox("Translate Tibetan to English", value=False)
        df = render_results_table(matches, translate=translator)
        st.dataframe(df)
    else:
        
        st.json(results.to_dict() if hasattr(results, "to_dict") else {"matches": normalize_matches(matches)})
