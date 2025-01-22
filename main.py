import google.generativeai as genai
import os
import streamlit as st
from utilities.vector_db import VectorDB
from utilities.llm_prompt import get_summary, enhance_query
from utilities.config_logger import get_logger
import json
from dotenv import load_dotenv
load_dotenv(override=True)

# Logger and API key setup
logger_st = get_logger("Streamlit", "app.log")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Document Summarizer")
st.write("Upload a PDF document to extract and summarize important sections categorized by topics.")  # noqa: E501


uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if uploaded_file:
    logger_st.info("File uploaded successfully")
    with st.spinner("Extracting and processing PDF..."):
        # Initializing the VectorDB
        vectordb = VectorDB(uploaded_file=uploaded_file)

        # Get hash value of the pdf

        logger_st.info("PDF data vectorized")
        st.info("Data loaded successfully")
        query = st.text_input("Ask your Query")
        summary_btn = st.button("Give Summary", type='secondary')

        if summary_btn:
            hash_value = vectordb.get_hash(uploaded_file)
            # Check the index is already present in the PineCone
            is_index_exist = vectordb.validate_index(hash_value)
            # Enhance the User query to get higher output value.
            enhanced_query = enhance_query(query, model)

            if not is_index_exist:
                # Get vectors and store it in index
                index = vectordb.convert_vectors()
                try:
                    # Attempt to open and read the JSON file
                    with open('const.json', 'r') as file:
                        data = json.load(file)
                        vectordb.delete_index(data.get('name'))
                except FileNotFoundError:
                    logger_st.info("File Not Found")
            else:
                # Get the index object by passing the hash value
                index = vectordb.pc.Index(hash_value)
            with open("const.json", "w") as json_file:
                json.dump({"name": hash_value}, json_file, indent=4)
            logger_st.info(f'{query} - {enhanced_query}')
            # Perform the similarity search to get the answer
            response = vectordb.get_results(index, enhanced_query)
            response = '\n'.join(response)
            # Passing the results to LLM to summarize the output
            result = get_summary(model, response, enhanced_query)

            logger_st.info("Summary loaded successfully")
            st.write(result)
