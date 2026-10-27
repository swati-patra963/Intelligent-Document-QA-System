```python
import streamlit as st
import PyPDF2
from io import BytesIO
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Intelligent Document QA System",
    page_icon="📄",
    layout="wide"
)

# Load Question Answering Model
@st.cache_resource
def load_model():
    return pipeline(
        "question-answering",
        model="distilbert/distilbert-base-cased-distilled-squad"
    )

# Extract text from uploaded documents
def extract_text(uploaded_file):
    text = ""

    if uploaded_file.name.lower().endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    elif uploaded_file.name.lower().endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    return text.strip()


# Split text into manageable chunks
def split_text(text, chunk_size=1800, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


# Main application
def main():
    st.title("📄 Intelligent Document QA System")

    st.write(
        "Upload a document and ask questions to get "
        "answers based on its content."
    )

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"]
    )

    if uploaded_file is not None:

        if (
            st.session_state.get("uploaded_name") != uploaded_file.name
            or st.session_state.get("uploaded_bytes")
            != uploaded_file.getvalue()
        ):
            with st.spinner("Extracting document text..."):
                text = extract_text(uploaded_file)

            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.uploaded_bytes = uploaded_file.getvalue()
            st.session_state.document_text = text
            st.session_state.qa_chunks = split_text(text)
            st.session_state.chat_history = []

        text = st.session_state.document_text

        if not text:
            st.error(
                "No readable text found. If your PDF is scanned, "
                "you may need OCR to extract its text."
            )
            return

        st.success("Document uploaded successfully!")

        with st.expander("Preview Document"):
            st.text(text[:5000])
            if len(text) > 5000:
                st.caption("Preview limited to the first 5,000 characters.")

        question = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., What is the main objective of this document?"
        )

        if st.button("🔍 Get Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    qa_model = load_model()
                    best_answer = None

                    with st.spinner("Finding the answer..."):
                        for chunk in st.session_state.qa_chunks:
                            result = qa_model(
                                question=question,
                                context=chunk
                            )

                            if (
                                best_answer is None
                                or result["score"] > best_answer["score"]
                            ):
                                best_answer = result

                    if best_answer and best_answer["score"] > 0.10:
                        st.subheader("Answer")
                        st.write(best_answer["answer"])
                        st.caption(
                            f"Model confidence score: "
                            f"{best_answer['score']:.2%}"
                        )
                    else:
                        st.warning(
                            "No reliable answer was found in the document."
                        )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": (
                            best_answer["answer"]
                            if best_answer and best_answer["score"] > 0.10
                            else "No reliable answer found."
                        )
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        if st.session_state.get("chat_history"):
            st.subheader("Question History")

            for item in reversed(st.session_state.chat_history):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.divider()

    else:
        st.info("Please upload a PDF or TXT document to get started.")


if __name__ == "__main__":
    main()
```
