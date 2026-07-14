from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.batch import normalize_input_columns, predict_input_frame
from src.data_loader import prepare_dataset
from src.io_utils import read_csv_with_fallback
from src.prediction import train_predictor


DEFAULT_TRAINING_DATA = Path("data/selected_thermodynamic_properties.csv")
DEFAULT_INPUT_DATA = Path("data/example_input_compositions.csv")


st.set_page_config(
    page_title="Thermodynamic Property Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Thermodynamic Property Predictor")

    with st.sidebar:
        st.header("Training")
        training_source = st.radio(
            "Training data",
            ["Default CSV", "Upload CSV/Excel"],
            index=0,
        )
        uploaded_training = None
        if training_source == "Upload CSV/Excel":
            uploaded_training = st.file_uploader(
                "Upload training data",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="training_upload",
            )

        validation = st.selectbox("Validation", ["kfold", "loo"], index=0)
        model_set = st.selectbox("Model set", ["fast", "expanded", "full"], index=1)
        st.caption("`expanded` is the practical default. `full` can be slow with leave-one-out.")

        train_clicked = st.button("Train models", type="primary", use_container_width=True)

    training_path = _resolve_training_path(training_source, uploaded_training)
    if training_path is None:
        st.info("Upload a training CSV or Excel file to begin.")
        return

    try:
        dataset = prepare_dataset(training_path)
    except Exception as error:
        st.error(f"Training data could not be loaded: {error}")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows", len(dataset.frame))
    summary_cols[1].metric("Features", len(dataset.feature_columns))
    summary_cols[2].metric("Targets", len(dataset.target_columns))
    summary_cols[3].metric("Validation", validation)

    with st.expander("Training Data Preview", expanded=False):
        preview_columns = [
            c for c in ["Group", "Subgroup", "Name", "Chemical_formula"] + dataset.feature_columns + dataset.target_columns
            if c in dataset.frame.columns
        ]
        st.dataframe(dataset.frame[preview_columns], use_container_width=True, height=340)

    if train_clicked or "predictor" not in st.session_state:
        with st.spinner("Training and comparing candidate models..."):
            try:
                st.session_state.predictor = train_predictor(
                    dataset,
                    validation=validation,
                    model_set=model_set,
                )
                st.session_state.training_signature = (str(training_path), validation, model_set)
            except Exception as error:
                st.error(f"Model training failed: {error}")
                return

    predictor = st.session_state.get("predictor")
    if predictor is None:
        st.info("Click Train models to compare candidates.")
        return

    st.subheader("Model Comparison")
    model_table = _model_results_frame(predictor)
    top_cols = st.columns([1, 2, 2])
    top_cols[0].metric("Selected model", predictor.model_name)
    top_cols[1].metric("Best score", f"{model_table.iloc[0]['score']:.5f}")
    top_cols[2].metric("Candidate models", len(model_table))
    st.dataframe(model_table, use_container_width=True, height=380)

    st.subheader("Prediction Input")
    input_mode = st.radio(
        "Input source",
        ["Example input file", "Upload input file"],
        horizontal=True,
    )
    input_frame = None
    if input_mode == "Example input file":
        input_frame = normalize_input_columns(pd.read_csv(DEFAULT_INPUT_DATA))
    else:
        uploaded_input = st.file_uploader(
            "Upload prediction input",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="input_upload",
        )
        if uploaded_input is not None:
            input_frame = _read_uploaded_table(uploaded_input)

    if input_frame is None:
        st.info("Upload an input file with `Formula`, `Composition`, or element columns.")
        return

    st.dataframe(input_frame, use_container_width=True, height=180)

    output_path = st.text_input("Save result as", value="prediction_results.csv")
    predict_clicked = st.button("Predict and save result", type="primary")
    if predict_clicked:
        with st.spinner("Predicting properties..."):
            try:
                output = predict_input_frame(predictor, input_frame)
                output.to_csv(output_path, index=False, encoding="utf-8-sig")
                st.session_state.output_frame = output
                st.session_state.output_path = output_path
            except Exception as error:
                st.error(f"Prediction failed: {error}")
                return

    output = st.session_state.get("output_frame")
    if output is not None:
        st.subheader("Prediction Results")
        st.success(f"Saved result: {st.session_state.get('output_path', output_path)}")
        st.dataframe(output, use_container_width=True, height=360)
        st.download_button(
            "Download result CSV",
            data=_to_csv_bytes(output),
            file_name=Path(st.session_state.get("output_path", output_path)).name,
            mime="text/csv",
            use_container_width=True,
        )


def _resolve_training_path(training_source: str, uploaded_training) -> Path | None:
    if training_source == "Default CSV":
        return DEFAULT_TRAINING_DATA
    if uploaded_training is None:
        return None
    suffix = Path(uploaded_training.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_training.getbuffer())
        return Path(handle.name)


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = BytesIO(uploaded_file.getvalue())
    if suffix == ".csv":
        return normalize_input_columns(read_csv_with_fallback(data))
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return normalize_input_columns(pd.read_excel(data))
    raise ValueError(f"Unsupported input file extension: {suffix}")


def _model_results_frame(predictor) -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(predictor.results, start=1):
        row = {
            "rank": rank,
            "model": result.name,
            "score": result.score,
        }
        for target, value in result.rmse.items():
            row[f"rmse_{target}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


if __name__ == "__main__":
    main()
