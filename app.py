import streamlit as st

from backend.rag_pipeline import ask_question_across_documents, list_indexed_documents

st.set_page_config(page_title="Local PDF RAG Chatbot", page_icon=":books:", layout="wide")
st.title("Local PDF RAG Chatbot")
st.caption("Use the Admin page to upload PDFs and build local FAISS indexes.")

indexed_docs = list_indexed_documents()
if not indexed_docs:
    st.warning("No indexed PDFs found. Open the Admin page and upload files first.")
    st.stop()

doc_names = sorted({item.get("pdf_name", "") for item in indexed_docs if item.get("pdf_name", "")})
st.caption(f"Indexed documents available: {', '.join(doc_names)}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

user_question = st.chat_input("Ask a question from any indexed PDF")
if user_question:
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question_across_documents(user_question, k=4)
                answer = result["answer"]
                sources = result.get("source_documents", [])
                if sources:
                    answer = f"{answer}\n\n**Source document(s):** {', '.join(sources)}"
            except Exception as exc:
                answer = f"Error: {exc}"
            st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))
