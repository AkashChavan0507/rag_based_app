import hashlib
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_ROOT = Path(".indices")
DOC_ROOT = Path("data/docs")
INDEX_ROOT.mkdir(parents=True, exist_ok=True)
DOC_ROOT.mkdir(parents=True, exist_ok=True)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _file_fingerprint(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return {
        "sha256": sha.hexdigest(),
        "size": path.stat().st_size,
        "mtime": int(path.stat().st_mtime),
    }


def _index_key(pdf_path: Path, chunk_size: int, chunk_overlap: int, model_name: str) -> str:
    payload = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model_name": model_name,
        "format": "v1",
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_uploaded_file(file_name: str, file_bytes: bytes) -> Path:
    out_path = DOC_ROOT / file_name
    out_path.write_bytes(file_bytes)
    return out_path


def build_or_load_index(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    force_rebuild: bool = False,
) -> Path:
    embeddings = get_embeddings()
    key = _index_key(pdf_path, chunk_size, chunk_overlap, EMBEDDING_MODEL_NAME)
    index_dir = INDEX_ROOT / key

    if index_dir.exists() and not force_rebuild:
        return index_dir

    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "pdf_path": str(pdf_path.resolve()),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_dir


def list_indexed_documents() -> List[dict]:
    items: List[dict] = []
    for meta_path in INDEX_ROOT.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pdf_path = meta.get("pdf_path", "")
            items.append(
                {
                    "index_dir": str(meta_path.parent),
                    "pdf_path": pdf_path,
                    "pdf_name": Path(pdf_path).name if pdf_path else "",
                    "chunk_size": meta.get("chunk_size"),
                    "chunk_overlap": meta.get("chunk_overlap"),
                    "embedding_model": meta.get("embedding_model"),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(items, key=lambda x: x["pdf_path"])


def ask_question(index_dir: str, question: str, k: int = 4) -> str:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


def ask_question_across_documents(question: str, k: int = 4) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    indexed_docs = list_indexed_documents()
    if not indexed_docs:
        raise ValueError("No indexed PDFs found. Please index documents from the Admin page.")

    embeddings = get_embeddings()
    all_hits = []

    for item in indexed_docs:
        index_dir = item["index_dir"]
        pdf_name = item.get("pdf_name") or Path(item.get("pdf_path", "")).name
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        hits = vectorstore.similarity_search_with_score(question, k=k)
        for doc, score in hits:
            all_hits.append((doc, float(score), pdf_name))

    if not all_hits:
        return {
            "answer": "I don't know based on the indexed documents.",
            "source_documents": [],
        }

    all_hits.sort(key=lambda x: x[1])
    top_hits = all_hits[:k]
    context_docs = [hit[0] for hit in top_hits]
    source_documents = sorted({hit[2] for hit in top_hits if hit[2]})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    context = "\n\n".join(doc.page_content for doc in context_docs)
    answer = chain.invoke({"question": question, "context": context})

    return {
        "answer": answer,
        "source_documents": source_documents,
    }
