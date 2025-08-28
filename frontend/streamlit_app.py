import streamlit as st
import requests
import pandas as pd
import time
import json
import uuid  # Import the UUID library to generate unique IDs

# --- CONFIGURATION ---
BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="DocQuery AI-Wasserstoff",
    page_icon="📄",
    layout="wide"
)

# --- SESSION ID MANAGEMENT ---
# Generate a unique session ID for the user if one doesn't exist.
# This ID will persist as long as the user keeps the browser tab open.
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Create a dictionary of headers that will be sent with every API request.
# The backend will use "X-Session-ID" to identify the user's private data.
headers = {"X-Session-ID": st.session_state.session_id}

# --- HELPER FUNCTION ---
def convert_df_to_csv(df):
    """Converts a dataframe to a CSV string."""
    return df.to_csv(index=False).encode('utf-8')

# --- HEADER ---
st.title("📄 DocQuery AI - Your Private AI Chatbot")

# Add a button at the top to clear the session
if st.button("🗑️ Clear Session & Start Over"):
    try:
        # Call the new cleanup endpoint
        requests.post(f"{BACKEND_URL}/admin/session/end", headers=headers)
        # Clear the local session state as well
        st.session_state.clear()
        st.success("Session cleared! You can now start over by ingesting new documents.")
        # Rerun the app to get a new session_id and a clean UI
        st.rerun()
    except Exception as e:
        st.error(f"Could not clear session: {e}")

st.markdown("""
Welcome! This application is your private document analyst. Upload your documents, and the AI will answer questions and identify themes based *only* on the content you provide. 
Your data is temporary and will be cleared when you end your session.
""")


# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["1. Ingest Documents", "2. Ask Questions", "3. Explore Themes"])

# --- TAB 1: DOCUMENT INGESTION ---
with tab1:
    st.header("Upload and Process Your Documents")
    st.write("Upload multiple documents to build your temporary, private knowledge base.")
    uploaded_files = st.file_uploader(
        "Choose your documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg']
    )
    if st.button("Ingest Documents") and uploaded_files:
        with st.spinner("Processing documents..."):
            files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
            try:
                # Add the 'headers=headers' parameter to the request
                response = requests.post(f"{BACKEND_URL}/ingest/batch", files=files_to_upload, headers=headers)
                if response.status_code == 200:
                    st.success("Ingestion complete!")
                    st.json(response.json())
                else:
                    st.error(f"Error during ingestion: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

# --- TAB 2: ASK QUESTIONS ---
with tab2:
    st.header("Query the Knowledge Base")
    st.write("Ask a question about the content of your ingested documents.")
    query = st.text_input("Enter your question:")
    if st.button("Get Answer") and query:
        with st.spinner("Searching for answers..."):
            payload = {"query": query, "top_k": 20, "final_n": 5}
            try:
                # Add the 'headers=headers' parameter to the request
                response = requests.post(f"{BACKEND_URL}/ask", json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("🧠 Synthesized Answer")
                    st.write(data.get("answer", "No answer could be generated."))
                    st.subheader("📑 Supporting Passages")
                    for chunk in data.get("supporting_chunks", []):
                        citation = f"Source: {chunk.get('doc_id', 'N/A')}, Page: {chunk.get('page', 'N/A')}"
                        with st.expander(citation):
                            st.write(chunk.get("text", ""))
                else:
                    st.error(f"Error from backend: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

# --- TAB 3: EXPLORE THEMES ---
with tab3:
    st.header("Comprehensive Theme Analysis")
    st.write("Generate a report with descriptive names and summaries for all unique themes.")
    if 'themes_analysis_data' not in st.session_state:
        st.session_state.themes_analysis_data = None
    if st.button("Analyze All Themes"):
        count = 0
        try:
            # Add the 'headers=headers' parameter to the request
            count_response = requests.get(f"{BACKEND_URL}/themes/count", headers=headers)
            if count_response.status_code == 200:
                count = count_response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to get theme count: {e}")
        
        if count > 0:
            st.info(f"Found {count} unique themes. Starting live analysis...")
            results_container = st.empty()
            all_results = []
            try:
                # Add the 'headers=headers' parameter to the request
                with requests.post(f"{BACKEND_URL}/themes/analyze-stream", stream=True, timeout=300, headers=headers) as response:
                    # ... (rest of the streaming logic is the same)
                    for line in response.iter_lines():
                         if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                try:
                                    data = json.loads(line_str[6:])
                                    if "error" in data:
                                        st.error(f"Backend error: {data['error']}")
                                        break
                                    all_results.append(data)
                                    with results_container.container():
                                        st.success(f"Analyzed {len(all_results)} of {count} themes...")
                                        for row in all_results:
                                            with st.expander(f"### {row['name']}"):
                                                st.write(f"**Summary:** {row['summary']}")
                                except json.JSONDecodeError:
                                    continue
            except requests.exceptions.RequestException as e:
                st.error(f"Connection to backend stream failed: {e}")
            st.session_state.themes_analysis_data = pd.DataFrame(all_results)
        else:
            st.warning("No themes found to analyze.")
            
            
    # Display the download button if analysis is complete
    if 'themes_analysis_data' in st.session_state and st.session_state.themes_analysis_data is not None and not st.session_state.themes_analysis_data.empty:
        df_to_download = st.session_state.themes_analysis_data[['name', 'summary']]
        df_to_download['citations_count'] = st.session_state.themes_analysis_data['citations'].apply(len)
        csv_data = convert_df_to_csv(df_to_download)
        
        st.download_button(
           label="📥 Download Report as CSV",
           data=csv_data,
           file_name='full_theme_analysis.csv',
           mime='text/csv',
        )