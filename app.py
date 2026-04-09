import pandas as pd
import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="🎓",
    layout="centered",
)


def _inject_css() -> None:
    st.markdown(
        """
<style>
  .app-title {
    font-size: 2.0rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
  }
  .app-subtitle {
    color: rgba(250,250,250,0.75);
    margin-top: 0rem;
    margin-bottom: 1.25rem;
  }
  .card {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 16px 16px;
    background: rgba(255,255,255,0.03);
  }
  .metric {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
  }
  .metric-sub {
    margin-top: -0.25rem;
    color: rgba(250,250,250,0.75);
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### How it works")
        st.write(
            "This app loads the trained model from `artifacts/model.pkl` "
            "and the preprocessing pipeline from `artifacts/preprocessor.pkl`."
        )
        st.markdown("### Setup checklist")
        st.write("- Train once: `python -m src.pipeline.train_pipeline`")
        st.write("- Run app: `streamlit run app.py`")


def main() -> None:
    _inject_css()
    _sidebar()

    st.markdown('<div class="app-title">Student Math Score Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Predict expected <b>math score</b> using demographics and reading/writing scores.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["female", "male"])
            race_ethnicity = st.selectbox(
                "Race / ethnicity",
                ["group A", "group B", "group C", "group D", "group E"],
            )
            lunch = st.selectbox("Lunch", ["free/reduced", "standard"])

        with c2:
            parental_level_of_education = st.selectbox(
                "Parental level of education",
                [
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree",
                ],
            )
            test_preparation_course = st.selectbox(
                "Test preparation course",
                ["none", "completed"],
            )
            reading_score = st.number_input("Reading score", min_value=0, max_value=100, value=70, step=1)
            writing_score = st.number_input("Writing score", min_value=0, max_value=100, value=70, step=1)

        st.markdown("</div>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    with col_btn2:
        st.caption("Tip: train the model first so `artifacts/` exists.")

    if predict_clicked:
        try:
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=int(reading_score),
                writing_score=int(writing_score),
            )

            features: pd.DataFrame = data.get_data_as_data_frame()
            preds = PredictPipeline().predict(features)
            pred = float(preds[0])

            st.markdown("### Result")
            st.markdown(
                f"""
<div class="card">
  <p class="metric">{pred:.1f}</p>
  <p class="metric-sub">Predicted math score (0–100)</p>
</div>
""",
                unsafe_allow_html=True,
            )

        except FileNotFoundError:
            st.error(
                "Model artifacts not found. Train first:\n\n"
                "`python -m src.pipeline.train_pipeline`"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()

