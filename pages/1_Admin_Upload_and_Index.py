from pathlib import Path

import streamlit as st

from backend.rag_pipeline import build_or_load_index, list_indexed_documents, save_uploaded_file

st.set_page_config(page_title="Admin - Upload and Index", page_icon=":gear:")
st.title("Admin - Upload and Index")
st.caption("Upload PDF files, then build and store local FAISS indexes.")

chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=25)
force_rebuild = st.checkbox("Force rebuild index if already present", value=False)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process and index uploaded files", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        for file in uploaded_files:
            try:
                saved_path = save_uploaded_file(file.name, file.getvalue())
                index_dir = build_or_load_index(
                    pdf_path=Path(saved_path),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    force_rebuild=force_rebuild,
                )
                st.success(f"Indexed: {saved_path.name} -> {index_dir}")
            except Exception as exc:
                st.error(f"Failed to index {file.name}: {exc}")

st.divider()
st.subheader("Indexed documents")
rows = list_indexed_documents()
if not rows:
    st.info("No indexes created yet.")
else:
    st.dataframe(
        [{"document_name": row.get("pdf_name", "")} for row in rows],
        use_container_width=True,
        hide_index=True,
    )
