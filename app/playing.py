import streamlit as st

# p=0
print("First value of p: ")
with st.form("candidate_form"):
        st.subheader("Candidate Information")
        full_name = st.text_input("Full Name")
        
        experience = st.number_input("Years of Experience", min_value=0, step=1)
        

        submitted = st.form_submit_button("Submit")
        
if submitted:
        p=25
        print("Pacchhis p: ",p)

st.write("**Writing Text**")
user_answer = st.text_input("Your Answer:")
        
        
if st.button("Submit Answer"):
    # p=10
    print("In button P: ")
    
# p=20
print("Last P: ")