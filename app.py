import os
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# Try import predictor (robust to module layout)
try:
    from src.predict import predict_insurance_renewal
except Exception:
    try:
        from predict import predict_insurance_renewal
    except Exception:
        predict_insurance_renewal = None

st.set_page_config(page_title="Insurance renewal predection model Ananlysis", layout="wide")
st.title("Insurance renewal predection model Ananlysis")

menu = st.sidebar.radio(
    "Left side menu",
    ("Upload the training dataset", "Upload test Dataset", "User Experiance")
)

PLOTS_DIR = Path("src/Visualizations/Plots")
RESULTS_DIR = Path("src/Visualizations/Results")
DATA_DIR = Path("src/Dataset") if Path("src/Dataset").exists() else Path("Dataset")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def list_plot_files():
    if not PLOTS_DIR.exists():
        return []
    exts = (".png", ".jpg", ".jpeg", ".gif")
    files = sorted([p for p in PLOTS_DIR.iterdir() if p.suffix.lower() in exts])
    return files

def show_plots_grid(plot_files, cols_per_row=3):
    if not plot_files:
        st.warning(f"No plots found in {PLOTS_DIR}.")
        return
    cols = st.columns(cols_per_row)
    for i, p in enumerate(plot_files):
        try:
            img = Image.open(p)
            with cols[i % cols_per_row]:
                st.image(img, caption=p.name, use_column_width=True)
        except Exception as e:
            st.write(f"Cannot display {p.name}: {e}")

def show_model_comparison():
    comp_path = RESULTS_DIR / "model_comparison_metrics.csv"
    if comp_path.exists():
        try:
            df_comp = pd.read_csv(comp_path)
            st.subheader("Model comparison metrics")
            st.dataframe(df_comp.round(4))
        except Exception as e:
            st.write("Failed to load model comparison table:", e)
    else:
        st.info("No model comparison metrics found (Visualizations/Results/model_comparison_metrics.csv).")

if menu == "Upload the training dataset":
    st.header("Upload the training dataset")
    uploaded = st.file_uploader("Upload training CSV", type=["csv"], key="train_upload")
    if uploaded is not None:
        st.write("File uploaded:", uploaded.name)
        save_path = DATA_DIR / "train_uploaded.csv"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved training file to {save_path}")

    if st.button("Submit - show saved Visualizations/Plots and model comparison"):
        plot_files = list_plot_files()
        st.subheader("Plots from Visualizations/Plots")
        show_plots_grid(plot_files, cols_per_row=3)
        # also show model comparison table (if present)
        show_model_comparison()

elif menu == "Upload test Dataset":
    st.header("Upload test Dataset")
    uploaded_test = st.file_uploader("Upload test CSV", type=["csv"], key="test_upload")
    if uploaded_test is not None:
        st.write("File uploaded:", uploaded_test.name)
        save_path = DATA_DIR / "test_uploaded.csv"
        with open(save_path, "wb") as f:
            f.write(uploaded_test.getbuffer())
        st.success(f"Saved test file to {save_path}")
    if st.button("Submit test dataset"):
        st.info("Test dataset saved. Use training notebook / pipeline to evaluate models with this test set.")

else:  # User Experiance
    st.header("User Experiance")
    with st.form("user_experience_form"):
        premium = st.number_input("Premium", min_value=0.0, value=1000.0, step=1.0)
        late_payments = st.number_input("Late payments (count)", min_value=0, value=0, step=1)
        application_underwriting = st.number_input("Application underwriting score", min_value=0, value=700, step=1)
        age = st.number_input("Age (years)", min_value=0, value=30, step=1)
        submitted = st.form_submit_button("Submit for prediction")

    if submitted:
        st.write("Input values:")
        st.write(dict(premium=premium, late_payments=late_payments, application_underwriting=application_underwriting, age=age))
        if predict_insurance_renewal is None:
            st.error("Prediction function `predict_insurance_renewal` not found. See src/predict.py.")
        else:
            try:
                result = predict_insurance_renewal(premium, late_payments, application_underwriting, age)
            except TypeError:
                try:
                    result = predict_insurance_renewal(premium, late_payments, application_underwriting, 0, age)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    result = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                result = None

            if result is not None:
                st.success("Prediction result:")
                st.json(result)
