import streamlit as st
from langchain import OpenAI, LLMChain
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import json

def load_json_to_dataframe(json_file):
    data = json.load(json_file)
    df = pd.json_normalize(data)
    return df

def create_agent(df):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)
    return agent

def main():
    st.title("AI Chatbot for Custom JSON Files")
    
    uploaded_file = st.file_uploader("Upload your JSON file", type="json")
    
    if uploaded_file is not None:
        df = load_json_to_dataframe(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        agent = create_agent(df)
        
        question = st.text_input("Ask a question about your data:")
        
        if question:
            with st.spinner("Thinking..."):
                try:
                    answer = agent.run(question)
                    st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
