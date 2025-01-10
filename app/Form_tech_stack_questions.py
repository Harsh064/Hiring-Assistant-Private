import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("Hugging Face API token is missing. Please add it to the .env file.")
    st.stop()

# Define PromptTemplate for generating questions
question_prompt = PromptTemplate(
    input_variables=["tech_stack", "previous_answer", "context"],
    template=(
        "You are a professional interview question generator. The user is proficient in "
        "{tech_stack}. Based on the user's past responses: {previous_answer} and context: {context}, "
        "generate a specific, relevant, and challenging interview question related to {tech_stack}."
    )
)

# Set up LangChain components
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.2, "max_length": 512},
    huggingfacehub_api_token=hf_token,
)
memory = ConversationBufferMemory(input_key="tech_stack", memory_key="context", return_messages=True)

# Helper functions
def create_candidate_profile(full_name, email, phone, experience, position, location, tech_stack):
    return {
        "Full Name": full_name,
        "Email Address": email,
        "Phone Number": phone,
        "Years of Experience": experience,
        "Desired Position(s)": position,
        "Current Location": location,
        "Tech Stack": tech_stack,
    }

def get_vectorstore(candidate_profile):
    profile_text = "\n".join([f"{key}: {value}" for key, value in candidate_profile.items()])
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_text(profile_text)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def save_conversation(candidate_name, candidate_profile, conversation_data):
    folder_name = "candidate_conversations"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{candidate_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    combined_data = {
        "candidate_profile": candidate_profile,
        "conversation_data": conversation_data,
    }
    with open(os.path.join(folder_name, file_name), "w") as f:
        json.dump(combined_data, f, indent=4)
    st.success(f"Conversation saved successfully as {file_name}!")

def main():
    st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon=":briefcase:")
    st.markdown(css, unsafe_allow_html=True)  # Add CSS styles

    st.title("TalentScout Hiring Assistant :briefcase:")
    st.markdown("<div style='text-align: center; margin: 10px'>Welcome to your personalized hiring assistant!</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin: 10px'>Please fill the details below to process further</div>", unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "tech_stacks" not in st.session_state:
        st.session_state.tech_stacks = []

    # Candidate information form
    with st.form("candidate_form"):
        st.subheader("Candidate Information")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        experience = st.number_input("Years of Experience", min_value=0, step=1)
        position = st.text_input("Desired Position(s)")
        location = st.text_input("Current Location")
        tech_stack = st.text_area("Tech Stack (comma-separated)")

        submitted = st.form_submit_button("Submit")

    if submitted:
        candidate_profile = create_candidate_profile(full_name, email, phone, experience, position, location, tech_stack)
        st.session_state.candidate_profile = candidate_profile
        st.session_state.vectorstore = get_vectorstore(candidate_profile)
        st.session_state.tech_stacks = [tech.strip() for tech in tech_stack.split(",") if tech.strip()]
        st.session_state.full_name = full_name
        st.success("Candidate profile created successfully!")
        st.markdown("<div style='text-align: center; margin: 5px; border:1px white'>To Submit your application, Please answer the below questions</div>", unsafe_allow_html=True)

    # Interview question generation
    if st.session_state.tech_stacks:
        tech_stacks = st.session_state.tech_stacks

        if st.session_state.current_index < len(tech_stacks):
            current_tech_stack = tech_stacks[st.session_state.current_index]
            previous_answer = ""
            if st.session_state.conversation_history:
                previous_answer = st.session_state.conversation_history[-1]["answer"]

            context = "\n".join(
                f"Q: {item['question']} A: {item['answer']}" for item in st.session_state.conversation_history
            )

            if not st.session_state.conversation_history or st.session_state.conversation_history[-1]["answer"]:
                question_chain = LLMChain(llm=llm, prompt=question_prompt, memory=memory)
                question = question_chain.run(
                    tech_stack=current_tech_stack,
                    previous_answer=previous_answer,
                    context=context,
                )
                st.session_state.conversation_history.append({"question": question, "answer": ""})

        # Display conversation history
        for i, qa in enumerate(st.session_state.conversation_history):
            st.markdown(bot_template.replace("{{MSG}}", qa["question"]), unsafe_allow_html=True)
            if qa["answer"]:
                st.markdown(user_template.replace("{{MSG}}", qa["answer"]), unsafe_allow_html=True)

        if st.session_state.conversation_history[-1]["answer"] == "":
            answer = st.text_input("Your Answer:", key=f"answer_{st.session_state.current_index}")

            if st.button("Submit Answer", key=f"submit_{st.session_state.current_index}"):
                st.session_state.conversation_history[-1]["answer"] = answer
                st.session_state.current_index += 1
                st.experimental_rerun()

        if st.session_state.current_index >= len(tech_stacks):
            st.write("Thank you for your answers. We will get back to you soon!")
            st.write("Conversation History:")
            st.json(st.session_state.conversation_history)
            save_conversation(st.session_state.full_name, st.session_state.candidate_profile, st.session_state.conversation_history)

if __name__ == "__main__":
    main()
