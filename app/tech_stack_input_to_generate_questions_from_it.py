from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import streamlit as st
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the Streamlit app
st.title("Professional Interview Question Generator")

# Set up LangChain components
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.2, "max_length": 512},
    huggingfacehub_api_token=hf_token,
)
memory = ConversationBufferMemory(input_key="tech_stack", memory_key="context", return_messages=True)

# Define PromptTemplate for generating questions
question_prompt = PromptTemplate(
    input_variables=["tech_stack", "previous_answer", "context"],
    template=(
        "You are a professional interview question generator. The user is proficient in "
        "{tech_stack}. Based on the user's past responses: {previous_answer} and context: {context}, "
        "generate a specific, relevant, and challenging interview question related to {tech_stack}."
    )
)

# Set up LangChain's chain for handling the prompt
question_chain = LLMChain(llm=llm, prompt=question_prompt, memory=memory)

# Main application logic
def main():
    # User's tech stacks
    if "tech_stacks" not in st.session_state:
        st.session_state.tech_stacks = ["python", "css", "html", "sql"]
        #st.write(f"st.session_state.tech_stacks: {st.session_state.tech_stacks}")
    tech_stacks = st.session_state.tech_stacks
    st.write(f"st.session_state.tech_stacks: {st.session_state.tech_stacks}")

    # Maintain conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if len(tech_stacks) == 0:
        st.write("Please enter your tech stacks to proceed.")
        return

    # Track the current question index
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Generate a new question automatically if there's no ongoing question
    if st.session_state.current_index < len(tech_stacks):
        current_tech_stack = tech_stacks[st.session_state.current_index]
        previous_answer = ""
        if st.session_state.conversation_history:
            # st.write(f"st.session_state.conversation_history[-1]['answer']-->{st.session_state.conversation_history[-1]['answer']}")
            previous_answer = st.session_state.conversation_history[-1]["answer"]

        context = "\n".join(
            f"Q: {item['question']} A: {item['answer']}" for item in st.session_state.conversation_history
        )
        # Check if the last question was answered and generate a new question
        if not st.session_state.conversation_history or st.session_state.conversation_history[-1]["answer"]:
            # st.write(f"context:{context}")
            
            question = question_chain.run(
                tech_stack=current_tech_stack,
                previous_answer=previous_answer,
                context=context,
            )
            st.session_state.conversation_history.append({"question": question, "answer": ""})
            st.write(f"Q: {question}")

    # Allow user to answer the current question
    if st.session_state.conversation_history:
        if st.session_state.conversation_history[-1]["answer"] == "":
            answer = st.text_input("Your Answer:")
            if st.button("Submit Answer"):
                st.session_state.conversation_history[-1]["answer"] = answer
                st.session_state.current_index += 1
                st.experimental_rerun()  # Automatically refresh the app state

    # Check if there are more tech stacks
    if st.session_state.current_index >= len(tech_stacks):
        st.write("Interview questions for all tech stacks are completed.")
        st.write("Conversation History:")
        st.json(st.session_state.conversation_history)

if __name__ == "__main__":
    main()
