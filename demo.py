import streamlit as st
import time
import google.generativeai as genai
from dotenv import load_dotenv
import os

st.set_page_config(layout="wide")  # Set the layout to wide

st.header("Interact with Google Gemini")
st.write("")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Create two columns with adjusted width ratio
col1, col2 = st.columns([1, 3])  # 1/4th for the first column, 3/4th for the second column

# Left column with the select box
with col1:
    industry = st.selectbox("Select Industry:", ["Payments", "HealthCare", "Insurance", "Banking"])

# Right column with the rest of the content
with col2:
    prompt_text = f"Create 10 glossary records for {industry} industry in table format with the headers as ( Reference ID, Name, Description, Alias Names, Business Logic, Critical Data Element, Examples, Format Type, Format Description, Lifecycle, Security Level, Classifications, Operation ) show only the table in response"
    prompt = prompt_text
    model = genai.GenerativeModel("gemini-2.0-flash")

    if st.button("SEND", use_container_width=True):
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()
        response_time = end_time - start_time

        st.write("")
        st.header("Response : ")
        st.write("")
        st.write("Response Time: ", response_time)
        st.markdown(response.text)
        with open("gemini_response.txt", "w") as file:
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response Time: {response_time}\n")
            file.write(f"Response: {response.text}\n")

        st.write("Response has been written to gemini_response.txt")