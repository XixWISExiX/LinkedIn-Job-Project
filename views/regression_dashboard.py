import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from regression_task.regression_algo import perform_regression, predict_salary, mean_squared_error, perform_r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go


random_seed = 42

skill_selection = set(st.session_state.job_market_filtered_df.columns[31:])
regression_columns = set(st.session_state.job_market_filtered_df.columns[31:])
regression_columns.update(['company_name', 'location', 'title', 'normalized_salary'])
regression_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(regression_columns)]

max_sample = regression_df.shape[0]


def train_regression_model(regression_df, k_folds, skill_inputs, company_input, location_input, title_input):
    df = regression_df.dropna().sort_values(by="normalized_salary").copy()

    # Figure out which categorical features we can actually use
    cat_features = []
    mp = {}
    mp["company_name"] = company_input
    mp["location"] = location_input
    mp["title"] = title_input
    if company_input is not None:
        cat_features.append("company_name")
    if location_input is not None:
        cat_features.append("location")
    if title_input is not None:
        cat_features.append("title")


    # ----- SKILLS: if any skill is entered, use *all* skills as features -----
    # === All Skills ===
    #skills_used = [s for s in skill_selection if s in df.columns]
    # === Just Filtered Skill ===
    skills_used = [s for s in skill_inputs if s in df.columns]

    # === TOP 10 SKILLS ===
    #all_skills = [s for s in skill_selection if s in df.columns]
    #skills_used = []
    #if all_skills:
    #    # drop skills that are extremely rare or constant
    #    # tune min_support as you like (e.g., 10, 20, etc.)
    #    min_support = 10
    #    support = df[all_skills].sum()
    #    frequent_skills = support[support >= min_support].index.tolist()

    #    if frequent_skills:
    #        nunique = df[frequent_skills].nunique()
    #        skills_used = [s for s in frequent_skills if nunique[s] > 1]

    if not cat_features and not skills_used:
        st.warning("Please provide at least one of: company, location, title, or skill.")
        return None  # nothing to train on

    # Build raw feature frame
    feature_cols_raw = cat_features + skills_used
    x_values = df[feature_cols_raw].copy()
    y = df["normalized_salary"]

    # One-hot encode the categorical columns we decided to use
    data_encoder = {}
    for col in cat_features:
        dummies = pd.get_dummies(x_values[col], prefix=col)
        data_encoder[col] = dummies.columns.tolist()
        x_values = pd.concat([x_values.drop(columns=[col]), dummies], axis=1)

        # === One hot encode only selected column ===
        #selected_val = mp.get(col)
        #if selected_val is None:
        #    x_values = x_values.drop(columns=[col])
        #    data_encoder[col] = []
        #    continue
        #dummy_col_name = f"{col}_{selected_val}"
        #x_values[dummy_col_name] = (x_values[col] == selected_val).astype(int)
        #data_encoder[col] = [dummy_col_name]
        #x_values = x_values.drop(columns=[col])

    # Design matrix with intercept
    X_matrix = np.c_[np.ones((len(x_values), 1)), x_values.to_numpy(dtype=float)]
    y_vector = y.values.reshape(-1, 1)

    if len(regression_df) < k_folds:
        st.warning(
            f"Not enough samples ({regression_df.shape[0]}) for {k_folds}-fold CV. "
            "Lower K or increase Max Samples."
        )
        st.stop()

    # ----- 1) SPLIT FOR EVALUATION -----
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    r2_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X_matrix):
        X_train, X_test = X_matrix[train_idx], X_matrix[test_idx]
        y_train, y_test = y_vector[train_idx], y_vector[test_idx]

        beta = perform_regression(X_train, y_train)
        y_pred = predict_salary(beta, X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    #float(np.mean(r2_scores)),
    #float(np.mean(mse_scores)),
    #float(np.std(r2_scores)),
    #float(np.std(mse_scores)),

    # Fit on train only for evaluation metrics
    beta_split = perform_regression(X_train, y_train)
    y_test_pred = predict_salary(beta_split, X_test)

    val_mse = mean_squared_error(y_test, y_test_pred)
    val_r2 = perform_r2_score(y_test, y_test_pred)

    # ----- 2) REFIT ON ALL DATA FOR FINAL MODEL -----
    beta_full = perform_regression(X_matrix, y_vector)
    y_full_pred = predict_salary(beta_full, X_matrix)

    intercept = float(beta_full[0, 0])
    coefficients = beta_full[1:].flatten()

    model = {
        "beta": beta_full,               # final model (trained on ALL data)
        "feature_cols": x_values.columns.tolist(),    # names of encoded features
        "data_encoder": data_encoder,    # for encoding new inputs
        "intercept": intercept,
        "coefficients": coefficients,
        "cat_features": cat_features,
        "skills_used": skills_used,

        # evaluation metrics from held-out test
        "mse": np.mean(mse_scores),
        "r2": np.mean(r2_scores),
        "mse_std": np.std(mse_scores),
        "r2_std": np.std(r2_scores),

        # full-data predictions for parity plot, etc.
        "y_vector": y_vector,
        "y_pred": y_full_pred,
    }

    return model



def build_feature_vector(company_input, location_input, title_input,
                         skill_inputs, model):

    feature_cols = model["feature_cols"]
    cat_features = model["cat_features"]
    skills_used  = model["skills_used"]

    row = {col: 0.0 for col in feature_cols}

    # --- categorical: company_name, location, title ---
    for col, value in [
        ("company_name", company_input),
        ("location", location_input),
        ("title", title_input),
    ]:
        if value is None:
            continue
        dummy_col = f"{col}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1.0  # one-hot

    # --- skills: 1 if selected, else 0 ---
    for skill in skill_selection:
        if skill in row:  # skill is one of the columns
            row[skill] = 1.0 if skill in skill_inputs else 0.0

    # put in correct order and add intercept at front
    x_vec = [row[c] for c in feature_cols]
    X_new = np.array([[1.0] + x_vec])  # shape (1, n_features+1)

    return X_new




# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

st.caption(
    "Estimate normalized salary from job attributes and skill signals. "
    "This page trains a lightweight linear regression model using your current filters "
    "and evaluates stability with K-fold cross-validation. "
    "For responsiveness, training may use a capped random sample of the filtered data."
)

with st.expander("ℹ️ What is this page showing?", expanded=False):
    st.markdown(
        """
        This page runs a **linear regression** over your filtered job postings to estimate **normalized salary**.

        **You can:**
        - Select **skills** and optionally **company, location, and title** inputs.
        - Control the **number of K-folds** used for evaluation.
        - View how reliable the model is via **R², MSE, and their standard deviations**.
        - Control the **maximum training sample size** for faster live modeling.

        **How the model is built:**
        - We build features from the inputs you provide.
        - Categorical fields (company/location/title) are **one-hot encoded**.
        - Skills are treated as binary indicators.
        - The model is evaluated with **K-fold cross-validation** and then refit on all data
          to produce the final coefficients used for prediction.

        **Tabs:**
        - **Regression Model Metrics** – quick reliability summary with R²/MSE.
        - **Regression Parity Plot** – predicted vs actual to spot bias/variance issues.
        - **Top-K Model Features** – the strongest coefficients in the current model.
        - **Help** – definitions and interpretation tips.

        **Notable Information:**
        - To keep the dashboard responsive, training is capped by **Max Samples** and uses a fixed random seed.
        - K-fold splitting also uses a fixed seed for reproducible metrics during demos.
        """
    )



top_col1, top_col2, top_col3  = st.columns([8, 2, 2], gap="small")

with top_col1:
    skill_inputs = st.multiselect("Types of Skills", sorted(skill_selection), default = [], placeholder= "All skills", help = "Select one or more skills to filter via dropdown or typing. Select the x in the red selection to delete a filter. Press the x next to the dropdown to delete all filters.")

with top_col2:
    k_folds = st.number_input('K-Folds', min_value=2, value=3, step=1, max_value=30, help="Number of cross-validation folds used to estimate model reliability. Higher K gives a more thorough estimate but can be slower. We use a fixed random seed for reproducible results in the dashboard.")

with top_col3:
    num_samples = st.number_input('Max Samples to Train Model', min_value=5, value=min(100, max_sample), step=(((max_sample//40 + 25) // 50) * 50), max_value=max_sample, help="Limits how many rows are used to train the local regression model for responsiveness. Smaller sample sizes run faster but can make R²/MSE noisier and less reliable. Larger sample sizes are slower but usually provide more stable and representative results. If the filtered dataset is larger than this value, a random sample is used.")

# Get all the comanies, locations, and titles for user inputs.
company_column = st.session_state.job_market_filtered_df['company_name']
job_location_column = st.session_state.job_market_filtered_df['location']
job_title_column = st.session_state.job_market_filtered_df['title']

# Get a unique list of the company, the title and the location for the user input.
company_list = list(set(company_column.tolist()))
job_location_list = list(set(job_location_column.tolist()))
job_title_list = list(set(job_title_column.tolist()))

# remove any floating point NaN's.
company_selection = [item for item in company_list if not (isinstance(item, float) and math.isnan(item))]
job_location_selection = [item for item in job_location_list if not (isinstance(item, float) and math.isnan(item))]
job_title_selection = [item for item in job_title_list if not (isinstance(item, float) and math.isnan(item))]

col1, col2, col3 = st.columns([4, 4, 4], gap="small")

with col1:
    company_input = st.selectbox("Company Name", sorted(company_selection), index = None, placeholder = "Select a company...", help = "Click the dropdown box or type to find the company.")

with col2:
    location_input = st.selectbox("Location", sorted(job_location_selection), index = None, placeholder = "Select a location...", help = "Click the dropdown box or type to find the job location.")

with col3:
    title_input = st.selectbox("Title", sorted(job_title_selection), index = None, placeholder = "Select a job title...", help = "Click the dropdown box or type to find the job title.")


if regression_df.shape[0] > num_samples:
    regression_df = regression_df.sample(n=num_samples, random_state=random_seed)

model = train_regression_model(regression_df, k_folds, skill_inputs, company_input, location_input, title_input)

#st.write(regression_df.shape)


if skill_inputs or company_input or location_input or title_input:
    X_new = build_feature_vector(
        company_input,
        location_input,
        title_input,
        skill_inputs,
        model
    )

    estimated_salary = predict_salary(model["beta"], X_new)[0, 0]

    st.markdown(f"### Estimated normalized salary: **${estimated_salary:,.0f}**")


    #st.write(model["beta"])


    #### TABS ####

    tabs = st.tabs(
        ["📊 Regression Model Metrics",
        "📈 Regression Parity Plot",
        "🧠 Top-K Model Features",
        "❓ Help"]
    )


    with tabs[0]:
        st.header("Model Metrics")
        st.caption(
            "Cross-validated performance summary for the current feature set. "
            "R² and MSE are reported as mean ± variation across K folds."
        )

        m1, m2, m3, m4, m5 = st.columns([2, 2, 2, 2, 4], gap="small")

        with m1:
            st.metric("R²", f"{model['r2']:.3f}")

        with m2:
            st.metric("R² std", f"{model['r2_std']:.3f}")

        with m3:
            st.metric("MSE", f"{model['mse']:.2e}")  # sci notation usually easier

        with m4:
            st.metric("MSE std", f"{model['mse_std']:.2e}")  # sci notation usually easier

        with m5:
            st.metric("Train samples", f"{len(model['y_vector'])}")

        with st.expander("Show regression coefficients (advanced)"):
            st.write(f"Intercept (b₀): `{model['intercept']:.2f}`")

            coef_df = (
                pd.DataFrame({
                    "feature": model["feature_cols"],
                    "coefficient": model["coefficients"],
                    "abs_coeff": np.abs(model["coefficients"]),
                })
                .sort_values("abs_coeff", ascending=False)
            )

            st.dataframe(coef_df[["feature", "coefficient"]], use_container_width=True)



    # Tab for the regression summary of entire dataset.
    with tabs[1]:
        st.header("Regression Parity Plot")
        st.caption(
            "Compares predicted vs actual normalized salaries. "
            "Points close to the dashed y=x line indicate better calibration."
        )

        y_true = model["y_vector"].flatten()
        y_pred = model["y_pred"].flatten()

        min_val = float(min(y_true.min(), y_pred.min()))
        max_val = float(max(y_true.max(), y_pred.max()))

        fig_parity = go.Figure()

        # Scatter: predicted vs actual
        fig_parity.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Predictions",
                opacity=0.8,
            )
        )

        # Ideal y = x line
        fig_parity.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal",
                line=dict(dash="dash"),
            )
        )

        fig_parity.update_layout(
            title="Parity Plot: Predicted vs Actual Normalized Salaries",
            xaxis_title="Actual normalized salary",
            yaxis_title="Predicted normalized salary",
            xaxis=dict(scaleanchor="y", scaleratio=1),  # keep 45° angle visually correct
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_parity, use_container_width=True)

    with tabs[2]:
        st.header("Regression Feature Importance Plot")
        st.caption(
            "Shows the most influential features by coefficient magnitude in the current linear model. "
            "Positive values increase predicted salary; negative values decrease it."
        )

        coef_df = pd.DataFrame({
                "feature": model["feature_cols"],
                "coefficient": model["coefficients"],
            })
        coef_df["abs_coeff"] = coef_df["coefficient"].abs()

        # Slider to choose K
        max_k = int(min(50, len(coef_df)))   # cap to keep plot readable
        K_feat = st.slider(
            "Top K features by |coefficient|",
            min_value=5,
            max_value=max_k,
            #value=min(10, max_k),
            key="k_feat_importance",
        )

        top_coef_df = (
            coef_df.sort_values("abs_coeff", ascending=False)
                .head(K_feat)
                .sort_values("coefficient", ascending=True)  # nicer for horizontal bar
        )

        fig_importance = px.bar(
            top_coef_df,
            x="coefficient",
            y="feature",
            orientation="h",
            title=f"Top {K_feat} Features by |Coefficient|",
        )
        fig_importance.update_layout(
            xaxis_title="Coefficient (impact on salary)",
            yaxis_title="Feature",
            margin=dict(l=100, r=20, t=60, b=40),
        )

        st.plotly_chart(fig_importance, use_container_width=True)


    with tabs[3]:
        st.subheader("How to read these metrics")
        st.markdown(
            r"""
            **R² (Coefficient of Determination)**

            - Measures how much variance in salary is explained by your selected features.
            - Range: (-∞, 1]. Higher is better.
            - A low R² here does not always mean your data is bad — it can mean the chosen features
            are weak proxies for compensation.

            **R² std (across folds)**

            - Indicates how stable your R² is across different train/test splits.
            - Lower is better (more reliable signal).

            **MSE (Mean Squared Error)**

            $$
            \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
            $$

            - Penalizes large prediction errors more heavily.
            - Useful for comparing models trained with the same target scaling.

            **MSE std (across folds)**

            - Measures how sensitive your error is to the data split.
            - High values often suggest overfitting or sparse/high-cardinality features.

            **Parity Plot**

            - Each point compares an actual salary to your model’s prediction.
            - The dashed line is the ideal **y = x** line.
            - Systematic curvature or wide scatter suggests missing features or non-linear effects.

            **Coefficients**

            - Positive coefficient ⇒ feature increases predicted salary.
            - Negative coefficient ⇒ feature decreases predicted salary.
            - For one-hot categories, coefficients reflect the effect relative to the omitted baseline.

            ---
            **Tip:**  
            For the most stable results, compare feature sets using similar input scope
            (e.g., “skills-only” vs “skills + title”) before drawing conclusions about model quality.
            """
        )
