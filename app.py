
import os
import streamlit as st

from rag_utility import (
    process_document_to_chroma_db,
    answer_question,
    answer_image_question
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PDF and Image QA Bot",
    page_icon="📄",
    layout="wide"
)


# ==================================================
# WORKING DIRECTORY
# ==================================================

working_dir = os.path.dirname(os.path.abspath(__file__))


# ==================================================
# APP TITLE
# ==================================================

st.title("📄 PDF and Image QA Bot")

st.markdown(
    "Upload a PDF or image and ask questions using AI."
)


# ==================================================
# PDF QUESTION ANSWERING
# ==================================================

st.header("📚 Ask Questions About a PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    key="pdf_uploader"
)


if uploaded_file is not None:

    # Save uploaded PDF
    filename = os.path.basename(uploaded_file.name)

    save_path = os.path.join(
        working_dir,
        filename
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {filename}")

    # Process PDF
    if st.button(
        "Process PDF",
        key="process_pdf_button"
    ):

        with st.spinner("Processing PDF..."):

            try:

                process_document_to_chroma_db(filename)

                st.session_state["pdf_processed"] = True
                st.session_state["processed_pdf_name"] = filename

                st.success(
                    "PDF processed successfully! "
                    "You can now ask questions."
                )

            except Exception as e:

                st.session_state["pdf_processed"] = False

                st.error(
                    f"Error processing PDF: {str(e)}"
                )


# ==================================================
# PDF QUESTION AND ANSWER
# ==================================================

st.subheader("💬 Ask a Question About Your PDF")

user_question = st.text_area(
    "Enter your question",
    placeholder="Example: Explain the project in detail.",
    key="pdf_question"
)


if st.button(
    "Generate PDF Answer",
    key="answer_pdf_button"
):

    if not user_question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Reading PDF content and generating your answer..."
        ):

            try:

                answer = answer_question(user_question)

                st.markdown("### 📄 AI Generated Answer")

                st.markdown(answer)

            except Exception as e:

                st.error(
                    f"Error answering your question: {str(e)}"
                )


# ==================================================
# DIVIDER
# ==================================================

st.markdown("---")


# ==================================================
# IMAGE QUESTION ANSWERING
# ==================================================

st.header("🖼️ Ask Questions About an Image")

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    key="image_uploader"
)


if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Image",
        use_container_width=True
    )


# Image question input
image_question = st.text_input(
    "What would you like to know about this image?",
    placeholder="Example: Describe everything in this image.",
    key="image_question"
)


# Generate image answer
if st.button(
    "Ask About Image",
    key="ask_image_button"
):

    if uploaded_image is None:

        st.warning("Please upload an image.")

    elif not image_question.strip():

        st.warning("Please enter a question about the image.")

    else:

        with st.spinner("Analyzing image using AI..."):

            try:

                image_answer = answer_image_question(
                    uploaded_image,
                    image_question
                )

                st.markdown("### 🖼️ Image Answer")

                st.write(image_answer)

            except Exception as e:

                st.error(
                    f"Error analyzing image: {str(e)}"
                )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "PDF and Image QA Bot | Powered by Streamlit, "
    "LangChain, ChromaDB and Groq"
)
