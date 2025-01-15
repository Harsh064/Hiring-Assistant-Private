import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import json
import os, re
from datetime import datetime
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.2, "max_length": 512},
        huggingfacehub_api_token=hf_token,
    )

profile_prompt = PromptTemplate(
    input_variables=["previous_question", "index"],
    template=(
        "You are an expert interview question generator tasked with gathering a candidate's profile. "
        "Generate one question at a time in the exact order from the following sequence of steps:\n\n"
        "Step 1. Full Name\n"
        "Step 2. Email Address\n"
        "Step 3. Phone Number\n"
        "Step 4. Years of Experience\n"
        "Step 5. Desired Position(s)\n"
        "Step 6. Current Location\n"
        "Step 7. Tech Stack (comma-separated).\n\n"
        "Strictly follow this sequence and do not deviate. Each question must ask only about the current step, "
        "using clear, concise, and professional language. Ensure that previously asked steps are not repeated.\n\n"
        "Here are examples of questions for each step:\n"
        "Step 1: 'What is your full name?'\n"
        "Step 2: 'Could you please provide your email address?'\n"
        "Step 3: 'May I have your phone number?'\n"
        "Step 4: 'How many years of experience do you have?'\n"
        "Step 5: 'What position(s) are you aiming for?'\n"
        "Step 6: 'Where are you currently located?'\n"
        "Step 7: 'What is your tech stack? Please list the technologies you are proficient in, separated by commas.'\n\n"
        "This is the previous question you generated: '{previous_question}'.\n"
        "Now generate the question for step {index}."
    ),
)

# question_prompt = PromptTemplate(
#     input_variables=["tech_stack", "previous_answer", "context"],
#     template=(
#         "You are a professional interview question generator. The user is proficient in "
#         "{tech_stack}. Based on the user's past responses: {previous_answer} and context: {context}, "
#         "generate a specific, relevant, and challenging interview question related to {tech_stack}."
#     )
# )
question_prompt = PromptTemplate(
    input_variables=["tech_stack", "previous_answer", "context"],
    template=(
        "You are a highly skilled interview question generator and conversational agent. "
        "The user is proficient in the following technical skills: {tech_stack}. "
        "Using their past responses: '{previous_answer}' and context: '{context}', "
        "generate a specific, relevant, and challenging interview question related to {tech_stack}.\n\n"
        "Guidelines:\n"
        "1. Ensure the question is highly specific, non-generic, and tailored to the mentioned tech stack.\n"
        "2. Take into account any patterns or gaps in the user's previous answers to refine the question.\n"
        "3. Avoid repeating questions or topics already addressed in the context.\n"
        "4. If a conversation-ending keyword is detected (e.g., 'stop', 'exit', 'end', 'quit', or similar), "
        "immediately respond with a fix polite message - 'Thank you for the conversation!' and terminate the interaction."
    )
)

# Conversation memory to track context
profile_memory = ConversationBufferMemory(input_key="index", memory_key="conversation_history", return_messages=True)
questions_memory = ConversationBufferMemory(input_key="tech_stack", memory_key="context", return_messages=True)

# Set up LangChain's chain for handling the prompt
profile_chain = LLMChain(llm=llm, prompt=profile_prompt, memory=profile_memory)
question_chain = LLMChain(llm=llm, prompt=question_prompt, memory=questions_memory)

st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon=":briefcase:")
st.markdown(css, unsafe_allow_html=True)

st.title("TalentScout Hiring Assistant :briefcase:")
st.markdown("<div style='text-align: center; margin: 10px'>Welcome to your personalized hiring assistant!</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin: 10px'>Please fill the details below to process further</div>", unsafe_allow_html=True)


def detect_conversation_end(response):
    
    keywords= ['end','bye','close','stop','terminate']
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b', re.IGNORECASE)

    return bool(pattern.search(response))

def save_conversation(candidate_name, candidate_profile, technical_questions_data):
    folder_name = "candidate_conversations"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{candidate_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    combined_data = {
        "candidate_profile": candidate_profile,
        "technical_questions_conersation": technical_questions_data,
    }
    with open(os.path.join(folder_name, file_name), "w") as f:
        json.dump(combined_data, f, indent=4)
    st.success(f"All details send to Evaluation!  We will get back to you soon!")


def make_candidate_profile(profile_chain):
    st.markdown("<h2 class='title'>Make Candidate Profile</h2>", unsafe_allow_html=True)

    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "question_index" not in st.session_state:
        st.session_state.question_index = 0

    if "candidate_profile_dict" not in st.session_state:
        st.session_state.candidate_profile_dict = {}

    if st.session_state.question_index < 7:  # Total of 7 questions
        previous_question = ""
        if st.session_state.conversation_history:
            previous_question = st.session_state.conversation_history[-1]["question"]

        if not st.session_state.conversation_history or st.session_state.conversation_history[-1]["answer"]:
          
            question = profile_chain.run(
                previous_question=previous_question,
                index=st.session_state.question_index + 1
            )
    
            st.session_state.conversation_history.append({"question": question, "answer": ""})
            # st.write(f"Question {st.session_state.question_index + 1}: {question}")

    for i, qa in enumerate(st.session_state.conversation_history):
            st.markdown(bot_template.replace("{{MSG}}",f"Question: {i + 1}: {qa['question']}"), unsafe_allow_html=True)
            if qa["answer"]:
                st.markdown(user_template.replace("{{MSG}}", qa["answer"]), unsafe_allow_html=True)

    if st.session_state.conversation_history:
        if st.session_state.conversation_history[-1]["answer"] == "":
            user_response = st.text_input("Your Answer:", key=f"response_{st.session_state.question_index}")
            
            if st.button("Submit Answer", key=f"submit_{st.session_state.question_index}"):
            
                st.session_state.conversation_history[-1]["answer"] = user_response.strip()

                if user_response and st.session_state.question_index==0:
                    st.session_state.name = user_response
                    
                st.session_state.question_index += 1
                st.experimental_rerun()  

    if st.session_state.question_index >= 7:
        st.success("User Profile Created!")
        
        # st.write("### Candidate Profile:")
        # for item in st.session_state.conversation_history:
        #     st.write(f"- **{item['question']}**: {item['answer']}")
        last_answer = st.session_state.conversation_history[-1]["answer"]
        tech_stack = [item.strip() for item in last_answer.split(",")]
        
        # st.write(f"Last Answer: {result}")
        # st.write(f"Type of Last Answer: {type(result)}")
        return tech_stack
                
def ask_tech_questions(tech_stack,question_chain):
    st.markdown("<h2 class='title'>Answer Some Technical Questions to Proceed</h2>", unsafe_allow_html=True)

    # User's tech stacks
    if "tech_stacks" not in st.session_state:
        st.session_state.tech_stacks = tech_stack
        #st.write(f"st.session_state.tech_stacks: {st.session_state.tech_stacks}")
    tech_stacks = st.session_state.tech_stacks
    # st.write(f"st.session_state.tech_stacks: {st.session_state.tech_stacks}")

    if "conversation_history2" not in st.session_state:
        st.session_state.conversation_history2 = []

    if len(tech_stacks) == 0:
        st.write("Please enter your tech stacks to proceed.")
        return

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if st.session_state.current_index < len(tech_stacks):
        current_tech_stack = tech_stacks[st.session_state.current_index]
        previous_answer = ""
        if st.session_state.conversation_history2:
        
            previous_answer = st.session_state.conversation_history2[-1]["answer"]

        context = "\n".join(
            f"Q: {item['question']} A: {item['answer']}" for item in st.session_state.conversation_history2
        )
        # Check if the last question was answered and generate a new question
        if not st.session_state.conversation_history2 or st.session_state.conversation_history2[-1]["answer"]:
            
            question = question_chain.run(
                tech_stack=current_tech_stack,
                previous_answer=previous_answer,
                context=context,
            )
            st.session_state.conversation_history2.append({"question": question, "answer": ""})
            
    for i, qa in enumerate(st.session_state.conversation_history2):
            st.markdown(bot_template.replace("{{MSG}}", qa["question"]), unsafe_allow_html=True)
            if qa["answer"]:
                st.markdown(user_template.replace("{{MSG}}", qa["answer"]), unsafe_allow_html=True)

    # Allow user to answer the current question
    if st.session_state.conversation_history2:
        if st.session_state.conversation_history2[-1]["answer"] == "":
            answer = st.text_input("Your Answer:")
                
            if st.button("Submit Answer"):
                st.session_state.conversation_history2[-1]["answer"] = answer
                st.session_state.current_index += 1
                if detect_conversation_end(answer):
                    st.session_state.current_index = 1000
                st.experimental_rerun()  # Automatically refresh the app state

    if st.session_state.current_index >= len(tech_stacks):
        st.success("Thank You! Conversation is ended.")
        return True
        # st.write("Conversation History:")

# Run the application
if __name__ == "__main__":
    tech_stack = make_candidate_profile(profile_chain=profile_chain)
   
    if tech_stack:
        questions = ask_tech_questions(tech_stack=tech_stack,question_chain=question_chain)
        
        if questions:
            save_conversation(st.session_state.name, st.session_state.conversation_history, st.session_state.conversation_history2)
    

