import os
import streamlit as st
from rag_engine import RAGAssistant


st.set_page_config(
    page_title="Document Q&A Assistant", page_icon="🤖", layout="centered"
)

st.title("📄 Document Q&A Assistant with RAG and Vector Embeddings")
st.write("Upload your PDF document, configure your OpenAI settings, and start asking questions!")


with st.sidebar:
    st.header("⚙️ OpenAI Settings")
    
   
    user_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    
    selected_model = st.selectbox(
        "Select OpenAI Model:",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    )
    
    st.markdown("---")
    st.header("📂 File Status")
    pdf_file = os.path.join("data", "sample.pdf")
    if os.path.exists(pdf_file):
        st.success("✅ `sample.pdf` file found!")
    else:
        st.error("❌ `sample.pdf` not found in `data/` folder.")
    
    st.markdown("---")
    st.info("Technology Stack:\n- LangChain\n- ChromaDB\n- OpenAI API\n- Streamlit")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.cache_resource
def load_rag_engine(api_key: str, model: str):
    pdf_path = os.path.join("data", "sample.pdf")
    if not os.path.exists(pdf_path):
        return None
    assistant = RAGAssistant(pdf_path=pdf_path)
    assistant.build_pipeline(api_key=api_key, model_name=model)
    return assistant


if query := st.chat_input("Type your question here..."):
    if not user_api_key:
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar before asking questions.")
    else:
        pdf_path = os.path.join("data", "sample.pdf")
        if not os.path.exists(pdf_path):
            st.error("⚠️ Please place a PDF file named `sample.pdf` in the `data` folder.")
        else:
            
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

           
            with st.chat_message("assistant"):
                with st.spinner("Searching for answers..."):
                    try:
                        assistant = load_rag_engine(user_api_key, selected_model)
                        if assistant is None:
                            st.error("Failed to load RAG engine. Check your PDF file.")
                        else:
                            result = assistant.ask(query)
                            answer = result["answer"]

                            # Format page citations
                            citations = result["citations"]
                            citation_text = "\n\n**📌 Source Page Citations:**\n"
                            if citations:
                                for c in citations:
                                    citation_text += f"- File: `{c['file']}`, Page: **{c['page']}**\n"
                            else:
                                citation_text += "- No specific page found."

                            full_response = answer + citation_text
                            st.markdown(full_response)

                            # Save assistant response to history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_response}
                            )
                    except Exception as e:
                        st.error(f"An error occurred: {e}")