#python -m streamlit run .\GhostDataAI.py

import streamlit as st
from PIL import Image
import io
import sqlite3
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
from groq import Groq
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent import RunnableAgent
from langchain.agents.agent import RunnableMultiActionAgent
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from operator import itemgetter
import time
from youtube_transcript_api import YouTubeTranscriptApi
import warnings


safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]


st.set_page_config(layout="wide")  # Set the layout to wide

# Load environment variables from .env file
load_dotenv()

DB_Location="sqlite:///../GhostAI/db/chinook.db"
#DB_Location="sqlite:///C:/AIContent/CODE/GhostAI/db/chinook.db"

llama_local='../GhostAI/llama-2-7b-chat.ggmlv3.q8_0.bin'
logo="../GhostAI/pics/capgemini.png"
logo1="../GhostAI/pics/DataGhostAI.jpg"
logo2="../GhostAI/pics/InformaticaIcon.png"
db_image="../GhostAI/pics/sqlliteDBSample.png"

# getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript

    except Exception as e:
        raise e

# getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text,prompt):

    #model=genai.GenerativeModel("gemini-pro")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response=model.generate_content(prompt+transcript_text)
    return response.text


def get_answer(question):
   
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    db = SQLDatabase.from_uri(DB_Location)
    llm = ChatGoogleGenerativeAI(
        #model="gemini-pro",
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
        temperature=0.0
    )

    execute_query = QuerySQLDataBaseTool(db=db)

    template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the {top_k} answer, also the sql code should not have ``` in beginning or end and sql word in output.
    Use the following format:

    Question: "Question here"
    "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    Only use the following tables:

    {table_info}.

    Question: {input}'''

    prompt = PromptTemplate.from_template(template)
    write_query = create_sql_query_chain(llm, db, prompt)
    querychain = write_query
    responsequerychain = querychain.invoke({"question": question})
    st.write("Query getting executed","\n")
    st.write("",responsequerychain)
    st.write("\n")
    chain = write_query | execute_query
    response = chain.invoke({"question": question})

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    {query}
    SQL Result: {result}
    Answer: """
    )

    
    answer = answer_prompt | llm | StrOutputParser()
    chain = (RunnablePassthrough.assign(query=write_query).assign(result=itemgetter("query") | execute_query) | answer)

    return chain.invoke({"question": question})

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    # Load embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Error handling if embeddings model is not available
    if not embeddings:
        st.error("Error: Failed to generate embeddings.")
        return

    # Error handling if no text chunks are found
    if not text_chunks:
        st.error("Error: No text chunks found.")
        return

    try:
        # Create vector store using FAISS
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Function to initialize conversational chain
def get_conversational_chain():
    # Define prompt template for conversational chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize ChatGoogleGenerativeAI model with Gemini Pro
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    # Create prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load QA chain using the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate response
def user_input(user_question):
    # Load embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load vector store
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    
    # Perform similarity search based on user question
    docs = new_db.similarity_search(user_question)

    # Initialize conversational chain
    chain = get_conversational_chain()

    # Generate response using conversational chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

# Function to get response from Gemini Pro model
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([prompt[0], question])
    return response.text
    
# Function to get response from LLAMA 2 model
def getLLamaresponse(input_text):
    llm = CTransformers(model=llama_local,
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01})
    response = llm(input_text)
    return response

# Function to convert image to byte array
def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Configure Google API key
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

st.image(logo, width=100)  # Column 3, Row 1


# Create two columns with adjusted width ratio
col1, col2 = st.columns([1, 4])  # Adjust the width ratio as needed

# Left column with three options
with col1:
    # st.markdown("<h2 style='color: #4b4b4b;'>Select Role:</h2>", unsafe_allow_html=True)
    option = st.radio(
        "Select Role:",
        ("D-Stewards", "Privacy_Office","Admin")
    )

# Right column with tabs based on selected option
with col2:
    if option == "Admin":
        gemini_pro, groc_interaction, local_llama = st.tabs(["Gemini Pro Performance", "Groq Performance", "Local Llama Performance"])

        with gemini_pro:
            # Interaction with Gemini Pro
            # st.header("Gemini Peformance for given request")
            st.write("")

            prompt = st.text_input("Input for Gemini:", "Give me name of all states in USA")
            model = genai.GenerativeModel("gemini-2.0-flash")

            if st.button("SEND", use_container_width=True):
                start_time = time.time()
                response = model.generate_content(prompt)
                end_time = time.time()
                response_time = end_time - start_time

                st.write("")
                st.header("Response : ")
                st.write("")
                st.write("response Time ", response_time)

                st.markdown(response.text)


        with groc_interaction:
            # Interaction with Groc
            groq_client = Groq(api_key=os.environ.get("groq_api_key"))
            # st.header("Groq Peformance for given request")
            user_input_detail = st.text_input("Input for Groq:","Give me name of all states in USA")
            submit_button = st.button("Submit")

            # Handle submit button click
            if submit_button:
                try:
                    # Generate response and display it
                    start_time = time.time()
                    chat_completion = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": user_input_detail}],
                        model="mixtral-8x7b-32768",
                    )
                    response = chat_completion.choices[0].message.content
                    st.header("Response:")
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.write("")
                    st.write("Response Time ", response_time)
                    st.write("")
                    st.write(response)

                except Exception as e:
                    st.error(f"An error occurred in getting response from Groq")



        with local_llama:
            # Interaction with Local LLAMA
            # st.header("LLAMA Peformance for given request")
            st.write("")

            input_text = st.text_input("Input for LLAMA:","Give me name of all states in USA")
            submit = st.button("Send")

            if submit:
                start_time = time.time()
                response = getLLamaresponse(input_text)
                # st.text_area("LLAMA:", value=response, height=200)
                end_time = time.time()
                response_time = end_time - start_time
                st.header(":blue[Local LLM Response]")
                st.write("")
                st.write("Response Time ", response_time)

                st.markdown(response)



    elif option == "D-Stewards":
        glossary_creator, db_interaction, pdf_interaction = st.tabs(["GhostAI Glossary Generator","Data Steward's Query Hub", "Document Insight Portal"])
        with db_interaction:
            st.header("Interact with Database")
            user_question = st.text_input("Enter your question:","Give me list of first 2 customers")
            if st.button("Get Answer"):
                answer_text = get_answer(user_question)
                st.write("Answer:", answer_text)
            st.image(db_image)

        with pdf_interaction:
            # Interaction with PDF
            st.header("Interaction with Knowledge Base")
            # user_question = st.text_input("Ask a Question from the PDF Files")

            user_question = st.text_input("Input: ", key="PDF_INPUT")

            if user_question:
                user_input(user_question)

            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Process to store in vector Store"):
                if pdf_docs is not None:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")


        with glossary_creator:
            st.header("Create Glossaries")
            st.write("")

            # Dropdown menu for industry selection
            industry = st.selectbox("Select Industry:", ["Payments", "HealthCare", "Insurance", "Banking"])
            # Generate the prompt based on the selected industry
            prompt_text = f"Create 100 glossary records for {industry} industry in table format with the headers as ( Reference ID, Name, Description, Alias Names, Business Logic, Critical Data Element, Examples, Format Type, Format Description, Lifecycle, Security Level, Classifications, Operation ) , here the Critical Data Element	can be only TRUE or FALSE and Format type can be only Text,Number,Date,Decimal,DateTime and Lifecycle can only be Draft and operation can only be 'create' show only the table in response"

            # Use the generated prompt_text internally
            prompt = prompt_text

            model = genai.GenerativeModel("gemini-2.0-flash")

            if st.button("SEND", use_container_width=True):
                response = model.generate_content(prompt)
                st.write("")
                # st.markdown(response.text)

                response_text = response.text

                # Remove all occurrences of triple backticks
                response_text = response_text.replace("```", "")


                # Process the response to remove blank first line and strip '|' from start and end of each line
                lines = response_text.strip().split('\n')

                if lines[0].strip() == "":
                    lines.pop(0)

                # Remove the second line, no matter what its content is
                if len(lines) > 1:
                    lines.pop(1)

                    
                processed_lines = [line.strip('|').strip() for line in lines]
                processed_content = '\n'.join(processed_lines)

                
                # Write the response to a file
                with open("Synthetic_Glossaries.csv", "w") as file:

                    file.write(f"{processed_content}\n")
                
                # Display the cleaned content
                st.write(f"Here are the first version of Glossaries for {industry} : {response_text}\n")

    elif option == "Privacy_Office":
        gemini_vision, video_summary = st.tabs(["Data Privacy & Security Inspector (Images)", "Media Governance Analyzer (Video)"])
        with gemini_vision:
            # Interaction with Photos
            st.header("Privacy checks for Pictures")
            st.write("")

            default_image_path = "../GhostAI/pics/DataGhost_AI_Architecture_Diagram.png"
            image_prompt = st.text_input("Interact with the Image",  "Give me all the details in the image")
            uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

            if uploaded_file is not None:
                st.image(Image.open(uploaded_file), use_column_width=True)

                st.markdown("""
                    <style>
                            img {
                                border-radius: 10px;
                            }
                    </style>
                    """, unsafe_allow_html=True)
                
            if st.button("GET RESPONSE", use_container_width=True):
                #model = genai.GenerativeModel("gemini-pro-vision")
                model = genai.GenerativeModel("gemini-1.5-flash")

                if uploaded_file is not None:
                    if image_prompt != "":
                        image = Image.open(uploaded_file)

                        response = model.generate_content(
                            glm.Content(
                                parts = [
                                    glm.Part(text=image_prompt),
                                    glm.Part(
                                        inline_data=glm.Blob(
                                            mime_type="image/jpeg",
                                            data=image_to_byte_array(image)
                                        )
                                    )
                                ]
                            )
                        )

                        response.resolve()

                        st.write("")
                        st.write(":blue[Response]")
                        st.write("")

                        st.markdown(response.text)

                    else:
                        st.write("")
                        st.header(":red[Please Provide a prompt]")

                else:
                    st.write("")
                    st.header(":red[Please Provide an image]")



        with video_summary:
            st.header("Privacy check for Video's")
            st.write("")

            prompt="""You are Yotube video summarizer. You will be taking the transcript text
            and summarizing the entire video and providing the important summary in points
            within 250 words. Please provide the summary of the text given here:  """

            youtube_link = st.text_input("Enter YouTube Video Link:", "https://www.youtube.com/watch?v=tcui215ghhw")
            #youtube_link = st.text_input("Enter YouTube Video Link:", "https://www.youtube.com/watch?v=2IK3DFHRFfw")
            
            if youtube_link:
                video_id = youtube_link.split("=")[1]
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

            if st.button("Get Detailed Notes"):
                transcript_text=extract_transcript_details(youtube_link)

                if transcript_text:
                    summary=generate_gemini_content(transcript_text,prompt)
                    st.markdown("## Detailed Notes:")
                    st.write(summary)