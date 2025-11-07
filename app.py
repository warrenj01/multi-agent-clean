import streamlit as st
from multi_agents import initialize_workflow, execute_workflow

st.title('Multi-Agent Exercise - With Streamlit')

if 'agent_workflow' not in st.session_state:
    try:
        with st.spinner("Initializing agents..."):
            st.session_state.agent_workflow = initialize_workflow()
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        st.stop()

user_query = st.text_area("Enter a topic for the blog post:", "")

if st.button("Generate Blog Post"):
    if user_query.strip():
        with st.spinner("Agents are at work... Please wait."):
            try:
                response = execute_workflow(st.session_state.agent_workflow, user_query)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic for the blog post.")
