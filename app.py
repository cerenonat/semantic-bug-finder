import streamlit as st

from src.retrieval import BugReportRetriever
from src.generation import generate_bug_summary


# ============================================================
# Semantic Bug Finder - Basic Streamlit Interface
#
# Demo flow:
# User bug title/description -> FAISS retrieval -> retrieved bug reports
# -> Ollama/Mistral generated debugging report
# ============================================================


st.set_page_config(
    page_title="Semantic Bug Finder",
    page_icon="🐞",
    layout="wide"
)


@st.cache_resource
def load_retriever():
    retriever = BugReportRetriever(
        dataset_path="data/bug_reports.csv",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    retriever.build_index()
    return retriever


example_queries = {
    "Opening invalid scene file crash": {
        "title": "Opening invalid scene file crashes the frontend",
        "description": "The application crashes when I try to open an invalid scene file. Errors during scene file loading do not seem to be handled properly."
    },
    "Window watcher callback error": {
        "title": "Window watcher callback error",
        "description": "The app reports a window watcher callback error and attempts to index a nil window value."
    },
    "Location picker crash": {
        "title": "Crash when opening location picker",
        "description": "The application crashes when selecting the all locations item in the location picker."
    },
    "Broken filter": {
        "title": "Filter is broken and does not return links",
        "description": "The filter does not return the expected links and seems to be broken after recent changes."
    },
    "Button not responding": {
        "title": "Buttons do not work in listing view",
        "description": "Clickable buttons in the listing view do not respond. Clicking them does nothing and no console error is shown."
    }
}


if "bug_title" not in st.session_state:
    st.session_state.bug_title = ""

if "bug_description" not in st.session_state:
    st.session_state.bug_description = ""


st.title("🐞 Semantic Bug Finder")
st.write(
    "Write your own software bug title and description. The system retrieves similar GitHub bug reports "
    "using FAISS and can generate a debugging report with Ollama/Mistral."
)

with st.sidebar:
    st.header("Project Info")
    st.write("**Dataset:** GitHub Bugs Prediction")
    st.write("**Document representation:** title + body")
    st.write("**Retriever:** HuggingFace Embeddings + FAISS")
    st.write("**Generator:** Ollama / Mistral")
    st.write("**Default top-k:** 3")

    st.divider()
    st.subheader("Evaluation Summary")
    st.write("Mean Precision@3: **0.8333**")
    st.write("Mean Precision@5: **0.74**")
    st.write("MRR: **1.0**")
    st.write("Mean NDCG@5: **0.982**")

    st.divider()
    st.subheader("Example Bugs")
    st.caption("Optional: click one to autofill the form.")

    for example_name, example_data in example_queries.items():
        if st.button(example_name):
            st.session_state.bug_title = example_data["title"]
            st.session_state.bug_description = example_data["description"]


st.divider()

st.subheader("Enter Bug Report")

bug_title = st.text_input(
    "Bug title:",
    key="bug_title",
    placeholder="Example: Application crashes when opening an invalid file"
)

bug_description = st.text_area(
    "Bug description:",
    key="bug_description",
    height=160,
    placeholder="Describe what happens, when it happens, expected behavior, actual behavior, and any error messages."
)

combined_query = (bug_title + ". " + bug_description).strip()

col1, col2 = st.columns([1, 2])

with col1:
    top_k = st.slider(
        "Number of retrieved bug reports",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    generate_answer = st.checkbox(
        "Generate debugging report with Ollama/Mistral",
        value=True
    )

analyze_button = st.button("Analyze Bug", type="primary")


if analyze_button:
    if combined_query.strip() in ["", "."]:
        st.warning("Please enter a bug title or description.")
    else:
        with st.spinner("Loading FAISS retriever..."):
            retriever = load_retriever()

        with st.spinner("Retrieving similar bug reports..."):
            results = retriever.search_bug_reports(combined_query, top_k=top_k)
            context = retriever.format_results_for_llm(results, max_chars=400)

        st.subheader("Retrieved Similar Bug Reports")
        st.caption("FAISS IndexFlatL2 is used, so lower distance scores mean higher semantic similarity.")

        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            score = result.get("score", 0.0)

            with st.expander(f"Result {i}: {title}  |  Distance: {score:.4f}", expanded=(i == 1)):
                st.write(f"**Bug ID:** {result.get('bug_id')}")
                st.write(f"**Title:** {title}")
                st.write(f"**FAISS Distance Score:** {score:.4f}")
                st.write(f"**Label:** {result.get('label', 'N/A')}")
                st.write("**Description:**")
                st.write(result.get("description", ""))

        if generate_answer:
            st.subheader("Generated Debugging Report")

            with st.spinner("Generating answer with Ollama/Mistral..."):
                try:
                    answer = generate_bug_summary(
                        user_query=combined_query,
                        retrieved_context=context,
                        model="mistral"
                    )
                    st.write(answer)

                except Exception as error:
                    st.error("Ollama/Mistral generation failed.")
                    st.write(error)
                    st.info(
                        "Make sure Ollama is running and the mistral model is installed. "
                        "You can also uncheck the generation option and use only FAISS retrieval."
                    )
