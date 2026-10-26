#step 1
import streamlit as st
st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("📄 RAG Demo")
st.write("Step 1 OK ✅ Streamlit is running")


#Step 2
import os
from dotenv import load_dotenv
load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

st.write("Step 2 OK ✅ Environment loaded")

#Step 3
uploaded = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key="pdf_uploader_main"
)


st.write("Uploaded file:", uploaded)


#Step 4
from pypdf import PdfReader


def extract_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


pdf_text = ""


if uploaded:
    pdf_text = extract_pdf_text(uploaded)
    st.subheader("Extracted Text Preview")
    st.write(pdf_text[:700] if pdf_text else "⚠️ No text extracted (scanned PDF needs OCR)")


    #Step 5
from langchain_text_splitters import RecursiveCharacterTextSplitter


chunks = []  # IMPORTANT: initialize so Step 6 can safely use it


if pdf_text:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_text(pdf_text)
    st.write("Step 5 OK ✅ Chunks created:", len(chunks))


#step 6
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# session_state keeps index alive across Streamlit reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# index_ready makes demo flow clear (no answering before indexing)
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


if st.button("✅ Build Vector Index", key="build_index_btn"):
    if not chunks:
        st.warning("Upload PDF first (no chunks available).")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.index_ready = True
        st.success("Step 6 OK ✅ FAISS index built. Now you can ask questions!")


