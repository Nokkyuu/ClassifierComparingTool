import logging
from logging.config import fileConfig
from pathlib import Path
import json
import os
import streamlit as st
from ui import main_ui_builder, sidebar_builder
from utils import DataHandler
from utils import DecisionTreeHandler, RandomForestHandler, KNNHandler, LogisticRegressionHandler

os.chdir(Path(__file__).parent)
fileConfig("./logging.ini")

with open("config.json", "r") as f:
    config = json.load(f)

logger = logging.getLogger(config["logger"])

#keeping the handlers in session state to avoid reinitialization
if "tree_handler" not in st.session_state:
    st.session_state["tree_handler"] = DecisionTreeHandler(logger)
if "rf_handler" not in st.session_state:
    st.session_state["rf_handler"] = RandomForestHandler(logger)
if "knn_handler" not in st.session_state:
    st.session_state["knn_handler"] = KNNHandler(logger)
if "logistic_handler" not in st.session_state:
    st.session_state["logistic_handler"] = LogisticRegressionHandler(logger)

classifiers = {
    "Decision Tree": st.session_state["tree_handler"],
    "Random Forest": st.session_state["rf_handler"],
    "KNN": st.session_state["knn_handler"],
    "Logistic Regression": st.session_state["logistic_handler"]
}

if "data_handler" not in st.session_state:
    st.session_state["data_handler"] = DataHandler(logger)
data_handler = st.session_state["data_handler"]

st.title("👀 Classifier Comparing Tool")

uploaded_file, scoring = sidebar_builder(logger)

if uploaded_file is not None:
    columns = data_handler.load_data(uploaded_file)
    main_ui_builder(logger, classifiers, data_handler, scoring, columns) #Builds the main UI with the provided parameters



