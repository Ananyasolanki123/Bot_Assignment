import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import api_client

st.set_page_config(page_title="Enterprise Chatbot UX", page_icon="🤖", layout="wide")

# Initialize session state
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "conversations_list" not in st.session_state:
    st.session_state.conversations_list = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_docs" not in st.session_state:
    st.session_state.pending_docs = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "OPEN_CHAT"


def load_conversations():
    st.session_state.conversations_list = api_client.list_conversations()

def switch_conversation(conv_id):
    st.session_state.current_conversation_id = conv_id
    history = api_client.get_conversation_history(conv_id)
    if history:
        st.session_state.messages = history.get("messages", [])
        st.session_state.current_mode = history.get("mode", "OPEN_CHAT")
    else:
        st.session_state.messages = []
    
    # Rerun to update UI
    st.rerun()

def delete_conversation_callback(conv_id):
    if api_client.delete_conversation(conv_id):
        if st.session_state.current_conversation_id == conv_id:
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
        load_conversations()
        st.rerun()

def app():
    st.markdown("""
    <style>
    /* A seamless central radial glow that expands left and right in Cyberpunk Cyan & Navy */
    .stApp {
        background: radial-gradient(ellipse at center, #0ea5e9 0%, #0f172a 40%, #000000 100%);
        background-attachment: fixed;
    }
    
    /* Modify sidebar to perfectly match the Navy aesthetic */
    [data-testid="stSidebar"] {
        background-color: #050a15 !important;
        border-right: 1px solid #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("DocMind ⚡")
    
    # Load conversations on startup
    if not st.session_state.conversations_list:
        load_conversations()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("New Conversation")
        new_mode = st.selectbox("Conversation Mode", ["OPEN_CHAT", "RAG_CHAT"])
        
        # Depending on mode, maybe show docs to include
        selected_docs = []
        if new_mode == "RAG_CHAT" and st.session_state.pending_docs:
            st.write("Available Pending Documents:")
            for doc in st.session_state.pending_docs:
                if st.checkbox(f"{doc['filename']} ({doc['id'][:8]}...)", key=f"sel_{doc['id']}"):
                    selected_docs.append(doc['id'])
        
        if st.button("Start New Conversation"):
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.session_state.current_mode = new_mode
            st.session_state.start_mode = new_mode
            st.session_state.start_docs = selected_docs
            # We don't actually create it on the backend until the first message is sent
            st.success(f"Started new {new_mode} conversation! Type a message to begin.")

        st.divider()

        st.header("Recent Conversations")
        if st.button("🔄 Refresh List"):
            load_conversations()
        
        for conv in st.session_state.conversations_list:
            col1, col2 = st.columns([4, 1])
            with col1:
                title = conv.get("title", f"Chat {conv['conversation_id'][:8]}...")
                if st.button(title, key=f"btn_{conv['conversation_id']}"):
                    switch_conversation(conv['conversation_id'])
            with col2:
                if st.button("❌", key=f"del_{conv['conversation_id']}", help="Delete Conversation"):
                    delete_conversation_callback(conv['conversation_id'])

    
    # --- MAIN CONTENT ---
    tab1, tab2 = st.tabs(["Chat Interface", "Document Management"])

    with tab2:
        st.header("Upload Documents")
        st.markdown("Upload PDFs to use in **RAG_CHAT** mode.")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if st.button("Upload & Process"):
            if uploaded_file is not None:
                with st.spinner("Uploading and processing..."):
                    result = api_client.upload_document(uploaded_file)
                    if result and "document_id" in result:
                        st.success(f"Successfully processed {uploaded_file.name}!")
                        st.session_state.pending_docs.append({
                            "id": result["document_id"],
                            "filename": uploaded_file.name
                        })
                    else:
                        st.error("Failed to process document.")
            else:
                st.warning("Please select a file first.")
        
        if st.session_state.pending_docs:
            st.subheader("Pending Documents (Ready for new RAG chats)")
            for doc in st.session_state.pending_docs:
                st.text(f"- {doc['filename']} (ID: {doc['id']})")
                
            if st.session_state.current_conversation_id and st.session_state.current_mode == "RAG_CHAT":
                if st.button("Link all pending docs to CURRENT conversation"):
                    doc_ids_to_link = [d['id'] for d in st.session_state.pending_docs]
                    res = api_client.link_docs_to_conversation(doc_ids_to_link, st.session_state.current_conversation_id)
                    if res:
                        st.success(f"Linked {len(doc_ids_to_link)} documents to current conversation.")
                        # Clear pending docs as they are linked
                        st.session_state.pending_docs = []
                        st.rerun()
                    else:
                        st.error("Failed to link documents.")

    with tab1:
        mode_label = "💬 Open Chat" if st.session_state.current_mode == "OPEN_CHAT" else "🧠 RAG Chat"
        st.subheader(mode_label)
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            role = message["role"].lower()
            if role == "system": continue # Don't display system prompts
            
            # Use sleek theme-matching avatars instead of standard ones
            avatar_icon = "👤" if role == "user" else "⚡"
            with st.chat_message("user" if role == "user" else "assistant", avatar=avatar_icon):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            # Add user message to local state temporarily
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                if st.session_state.current_conversation_id is None:
                    # Starting a completely new conversation
                    mode = st.session_state.get('start_mode', 'OPEN_CHAT')
                    docs = st.session_state.get('start_docs', [])
                    
                    new_conv = api_client.start_new_conversation(prompt, mode, docs)
                    if new_conv:
                        st.session_state.current_conversation_id = new_conv['conversation_id']
                        # The response will contain the full history including the first assistant reply
                        st.session_state.messages = new_conv.get('messages', [])
                        
                        # Remove linked docs from pending
                        st.session_state.pending_docs = [d for d in st.session_state.pending_docs if d['id'] not in docs]
                        
                        load_conversations() # refresh sidebar
                        st.rerun()
                    else:
                        st.error("Failed to start conversation. Please check the backend.")
                        # Remove the temporary user message
                        st.session_state.messages.pop()
                else:
                    # Continuing an existing conversation
                    reply = api_client.continue_conversation(st.session_state.current_conversation_id, prompt)
                    if reply:
                        # Append the assistant's reply
                        st.session_state.messages.append({"role": "assistant", "content": reply.get("content", "Error parsing response")})
                        st.rerun()
                    else:
                        st.error("Failed to get response.")
                        st.session_state.messages.pop()

if __name__ == "__main__":
    app()
