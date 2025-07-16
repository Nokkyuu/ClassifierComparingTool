import streamlit as st

# def reset():
#     st.session_state.clear()
#     st.rerun()

def sidebar_builder(logger):
    """Builds the sidebar for the Streamlit app.

    Args:
        logger (logging.Logger): Logger instance for logging.

    Returns:
        Uploaded File: The uploaded file if any, otherwise None.
        str: The selected scoring method.
    """
    st.sidebar.header("Data Upload")
    #TODO: add option to choose delimiter in UI
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    logger.info("File uploaded: %s", uploaded_file.name if uploaded_file else "No file uploaded")
    st.sidebar.header("Classifier Parameters")

    scoring = st.sidebar.selectbox(
            "Select scoring method",
            ["accuracy", "f1_macro", "recall_macro"])
    if st.sidebar.button("Reset"):
        st.session_state.clear()
        st.rerun()
        logger.info("app reset")
    
                
    return uploaded_file,scoring
