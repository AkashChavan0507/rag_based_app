# Streamlit RAG App
- Chat page for asking questions from indexed PDFs.
- Admin page for uploading PDFs and building local FAISS indexes.

## Folder structure

- `app.py` - Main chatbot page.
- `pages/1_Admin_Upload_and_Index.py` - Admin upload/index builder page.
- `backend/rag_pipeline.py` - Reusable RAG/indexing logic.
- `data/docs/` - Uploaded PDFs (created automatically).
- `.indices/` - FAISS index folders (created automatically).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add `.env` file in this folder:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run

```bash
streamlit run app.py
```

## How to use

1. Open **Admin - Upload and Index** page in Streamlit sidebar.
2. Upload one or more PDF files.
3. Click **Process and index uploaded files**.
4. Return to **Local PDF RAG Chatbot** page.
5. Ask questions; the app searches across all indexed PDFs and shows source document names.
