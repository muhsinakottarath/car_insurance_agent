import streamlit as st

from query import build_pipeline, ask

st.set_page_config(page_title="Car Insurance Assistant", page_icon="🚗")
st.title("🚗 Car Insurance Assistant")
st.caption("Learning project. Answers come only from the knowledge base.")


@st.cache_resource
def load_pipeline():
    """Build the pipeline once, not on every message."""
    return build_pipeline()


pipeline = load_pipeline()

# Keep the conversation between messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat box at the bottom
question = st.chat_input("Ask a question about your car insurance")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            answer, documents = ask(pipeline, question)
        st.markdown(answer)

        with st.expander("Sources used"):
            for doc in documents:
                name = doc.meta.get("file_path", "unknown").split("/")[-1]
                st.write(f"**{name}** (score {doc.score:.3f})")
                st.caption(doc.content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})