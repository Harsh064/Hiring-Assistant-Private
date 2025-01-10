from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import streamlit as st
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Streamlit app
st.title("Candidate Profile Question Generator")

# Set up LangChain components
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.2, "max_length": 512},
    huggingfacehub_api_token=hf_token,
)

# Updated question prompt without context
question_prompt = PromptTemplate(
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

# Conversation memory to track context
memory = ConversationBufferMemory(input_key="index", memory_key="conversation_history", return_messages=True)

# Initialize LangChain's chain
question_chain = LLMChain(llm=llm, prompt=question_prompt, memory=memory)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "candidate_profile_dict" not in st.session_state:
    st.session_state.candidate_profile_dict = {}

# Main application logic
def main():
    # Generate questions dynamically
    if st.session_state.question_index < 7:  # Total of 7 questions
        previous_question = ""
        if st.session_state.conversation_history:
            previous_question = st.session_state.conversation_history[-1]["question"]

        if not st.session_state.conversation_history or st.session_state.conversation_history[-1]["answer"]:
            # Generate the next question
            question = question_chain.run(
                previous_question=previous_question,
                index=st.session_state.question_index + 1
            )

            # Save the question in the session
            st.session_state.conversation_history.append({"question": question, "answer": ""})
            st.write(f"Question {st.session_state.question_index + 1}: {question}")

    # Input for user's response
    if st.session_state.conversation_history:
        if st.session_state.conversation_history[-1]["answer"] == "":
            user_response = st.text_input("Your Answer:", key=f"response_{st.session_state.question_index}")
            if st.button("Submit Answer", key=f"submit_{st.session_state.question_index}"):
                # Save the answer and increment the index
                st.session_state.conversation_history[-1]["answer"] = user_response.strip()
                st.session_state.question_index += 1
                st.experimental_rerun()  # Rerun the app to show the next question

    # Display the profile summary after completing all questions
    if st.session_state.question_index >= 7:
        st.write("### Candidate Profile:")
        for item in st.session_state.conversation_history:
            st.write(f"- **{item['question']}**: {item['answer']}")
        # last_answer = st.session_state.conversation_history[-1]["answer"]
        # result = [item.strip() for item in last_answer.split(",")]
        # st.write(f"Last Answer: {result}")
        # st.write(f"Type of Last Answer: {type(result)}")
            


# Run the application
if __name__ == "__main__":
    main()
