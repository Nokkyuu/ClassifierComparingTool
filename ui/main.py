import streamlit as st
import pandas as pd
import os

MODEL_DIR = "models"

def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        "results": [],
        "models": [],
        "columns": None,
        "features": None,
        "target": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_results():
    """Reset results and models in session state."""
    st.session_state["results"] = []
    st.session_state["models"] = []

def render_column_selection(data_handler, logger, columns):
    """Render the column selection UI.
    
    Args:        
        data_handler (DataHandler): Instance of DataHandler to manage data.
        logger (logging.Logger): Logger instance for logging.
        columns (list): List of column names from the uploaded data.
    """
    st.session_state["columns"] = columns
    
    feature_cols = st.multiselect(
        "Select feature columns", 
        st.session_state["columns"],
        key="feature_cols"
    )
    available_target_cols = [col for col in st.session_state["columns"] if col not in feature_cols]
    target_col = st.selectbox(
        "Select target column", 
        available_target_cols,
        key="target_col"
    )
    
    if st.button("Confirm selection"):
        st.session_state["features"] = feature_cols
        st.session_state["target"] = target_col
        data_handler.feature_select(feature_cols)
        data_handler.target_select(target_col)
        logger.info(f"Features: {feature_cols}, Target: {target_col}")
        st.success(f"Features: {', '.join(st.session_state['features'])} - Target: {st.session_state['target']}")

def render_classifier_comparison(classifiers, data_handler, scoring):
    """Render the classifier comparison UI.
    Args:
        classifiers (dict): Dictionary of classifier handlers.
        data_handler (DataHandler): Instance of DataHandler to manage data.
        scoring (str): Scoring method for model evaluation.
    """
    
    #st.write(f")
    
    
    selected_classifier = st.multiselect(
        "Select Classifiers to compare",
        list(classifiers.keys()),
        key="selected_classifier"
    )
    
    if st.button("Run Classifier comparison"):
        reset_results()
        results = st.session_state["results"]
        models = st.session_state["models"]
        
        if selected_classifier:
            for classifier_name in selected_classifier:
                classifier = classifiers[classifier_name]
                best_model, best_score, best_para = classifier.grid_search(
                    data_handler.features, data_handler.target, scoring=scoring
                )
                results.append({
                    "classifier": classifier_name,
                    f"{scoring}": best_score,
                    "best_params": best_para
                })
                models.append({
                    "classifier": classifier_name,
                    "model": best_model
                })
        else:
            st.error("Please select at least one classifier to compare.")

def render_results():
    """Render comparison results."""
    results = st.session_state["results"]
    if results:
        st.subheader("📋 Comparison Results")
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True)

def render_model_downloads(classifiers):
    """Render model download buttons.
    Args:
        classifiers (dict): Dictionary of classifier handlers."""
    models = st.session_state["models"]
    if models:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        for model_info in models:
            classifier = classifiers[model_info["classifier"]]
            model_path = f"{MODEL_DIR}/{model_info['classifier']}_model.pkl"
            
            try:
                classifier.save_model(model_info["model"], model_path)
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {model_info['classifier']} Model",
                            data=f,
                            file_name=f"{model_info['classifier']}_model.pkl",
                        )
                else:
                    st.error(f"Model file {model_path} does not exist.")
            except Exception as e:
                st.error(f"Failed to save {model_info['classifier']} model: {e}")

def main_ui_builder(logger, classifiers, data_handler, scoring, columns):
    """Main UI builder function.
    
    Args:
        logger (logging.Logger): Logger instance for logging.
        classifiers (dict): Dictionary of classifier handlers.
        data_handler (DataHandler): Instance of DataHandler to manage data.
        scoring (str): Scoring method for model evaluation.
        columns (list): List of column names from the uploaded data.
    """
    initialize_session_state()
    
    if columns is not None:
        render_column_selection(data_handler, logger, columns)
        
        if st.session_state.get("features") and st.session_state.get("target"):
            render_classifier_comparison(classifiers, data_handler, scoring)
            render_results()
            render_model_downloads(classifiers)
    else:
        st.error("Failed to load data. Please check the file format.")