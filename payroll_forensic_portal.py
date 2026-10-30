"""
IBS AI Payroll Anomaly Detection & Forensic Audit Portal
Run locally:
    pip install streamlit pandas numpy matplotlib scikit-learn
    streamlit run payroll_forensic_portal.py

Optional image files (place beside this script):
    efcc_logo.jpeg, coat_of_arms.jpeg, school_logo.png
    developer 1.jpeg ... developer 12.jpeg

CSV guidance:
- Upload one staff/payroll record per row.
- Include numeric salary fields such as basicsalary, allowances,
  deductions and/or netsalary.
- Optional identity/location fields: employeeid, name, state, lga.
- This tool flags unusual records for human review; a flag is not proof of fraud.
"""

import io
import os
import zipfile
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IBS Payroll Forensic Audit Portal",
    page_icon="🇳🇬",
    layout="wide",
)

GREEN = "#008751"
DARK_GREEN = "#1b5e20"
RED = "#d32f2f"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #f7f9f8; }}
    .nigeria-banner {{ background:{GREEN}; height:10px; border-radius:4px; margin:10px 0 15px; }}
    .top-header {{
        background:#fff; padding:15px; border-radius:12px;
        border:2px solid {GREEN}; box-shadow:0 4px 15px rgba(0,135,81,.15);
        text-align:center; margin-bottom:15px;
    }}
    .top-title {{ color:{GREEN}; font-weight:900; font-size:22px; margin:4px 0; }}
    .top-sub {{ color:{DARK_GREEN}; font-size:13px; font-weight:700; margin:0; }}
    .stButton>button {{ background:{GREEN}; color:white; font-weight:bold; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- Utility functions -------------------------
def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and common payroll field aliases."""
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    aliases = {
        "employee_id": "employeeid",
        "staff_id": "staffid",
        "staff_name": "name",
        "basic_salary": "basicsalary",
        "basic_pay": "basicsalary",
        "net_salary": "netsalary",
        "net_pay": "netsalary",
        "allowance": "allowances",
        "total_allowance": "allowances",
        "deduction": "deductions",
        "total_deduction": "deductions",
        "local_government": "lga",
        "local_govt": "lga",
    }
    data = data.rename(columns={k: v for k, v in aliases.items() if k in data.columns})
    return data


def make_sample_data(n=500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "employeeid": [f"EMP{i:03d}" for i in range(1, n + 1)],
            "name": [f"Employee_{i:03d}" for i in range(1, n + 1)],
            "state": rng.choice(["Kogi State", "Kwara State", "Ekiti State"], n),
            "lga": rng.choice(
                ["Okene", "Kabba", "Lokoja", "Ado-Ekiti", "Ijumu", "Yagba East"], n
            ),
            "basicsalary": np.maximum(rng.normal(85000, 15000, n), 30000).round(2),
            "allowances": np.maximum(rng.normal(18000, 5000, n), 0).round(2),
            "deductions": np.maximum(rng.normal(6000, 1800, n), 0).round(2),
        }
    )
    data["netsalary"] = (
        data["basicsalary"] + data["allowances"] - data["deductions"]
    ).round(2)

    # Deliberate synthetic examples for demonstrating anomaly detection.
    data.loc[10, "basicsalary"] = 450000
    data.loc[25, "allowances"] = 300000
    data.loc[50, "deductions"] = 0
    data["netsalary"] = (
        data["basicsalary"] + data["allowances"] - data["deductions"]
    ).round(2)
    return data


def get_numeric_features(data: pd.DataFrame):
    excluded = {
        "id", "employeeid", "staffid", "recordnumber", "bank_account",
        "account_number", "phone", "contact",
    }
    numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c.lower() not in excluded]


def run_anomaly_detection(data: pd.DataFrame, contamination: float):
    """Fit Isolation Forest and return a result frame plus model metadata."""
    features = get_numeric_features(data)
    if not features:
        raise ValueError(
            "No usable numeric payroll columns were found. Include fields such as "
            "basicsalary, allowances, deductions or netsalary."
        )

    x = data[features].replace([np.inf, -np.inf], np.nan)
    # Drop features with no usable values or no variation.
    x = x.dropna(axis=1, how="all")
    x = x.loc[:, x.nunique(dropna=True) > 1]
    if x.shape[1] == 0:
        raise ValueError("The numeric columns contain no variation for analysis.")

    x = x.fillna(x.median()).fillna(0)
    if len(x) < 10:
        raise ValueError("At least 10 payroll records are required.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    predictions = model.fit_predict(x_scaled)

    output = data.copy()
    output["anomalystatus"] = np.where(
        predictions == -1, "Flagged for Review", "No Anomaly Flag"
    )
    output["flagged"] = predictions == -1
    output["anomalyscore"] = model.decision_function(x_scaled)

    # PCA requires at least two features. Supply a zero second axis if only one exists.
    if x_scaled.shape[1] >= 2:
        components = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
        output["pc1"], output["pc2"] = components[:, 0], components[:, 1]
    else:
        output["pc1"] = x_scaled[:, 0]
        output["pc2"] = 0.0

    return output, features


def safe_chart(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def chart_pca(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    normal = data[~data["flagged"]]
    flagged = data[data["flagged"]]
    ax.scatter(normal["pc1"], normal["pc2"], s=15, label="Normal", alpha=.65)
    ax.scatter(flagged["pc1"], flagged["pc2"], s=35, marker="x", label="Flagged")
    ax.set(title="Payroll Records (PCA View)", xlabel="Principal Component 1",
           ylabel="Principal Component 2")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_lga(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "lga" in data.columns:
        counts = data.loc[data["flagged"], "lga"].fillna("Unknown").value_counts()
        if not counts.empty:
            ax.barh(counts.index.astype(str), counts.values)
        else:
            ax.text(.5, .5, "No flagged records", ha="center", va="center")
    else:
        ax.text(.5, .5, "LGA column not supplied", ha="center", va="center")
    ax.set(title="Flagged Records by LGA", xlabel="Flagged count")
    fig.tight_layout()
    return fig


def chart_status(data, pie=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = data["anomalystatus"].value_counts().reindex(
        ["No Anomaly Flag", "Flagged for Review"], fill_value=0
    )
    if pie:
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Anomaly Share")
    else:
        bars = ax.bar(counts.index, counts.values)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(int(bar.get_height())), ha="center", va="bottom")
        ax.set_title("Anomaly Status Count")
        ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    return fig


def chart_salary(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "netsalary" in data.columns:
        values = pd.to_numeric(data["netsalary"], errors="coerce").dropna()
        ax.hist(values, bins=15, edgecolor="white")
        ax.set(xlabel="Net salary (₦)", ylabel="Employee count",
               title="Net Salary Distribution")
    else:
        ax.text(.5, .5, "netsalary column not supplied", ha="center", va="center")
    fig.tight_layout()
    return fig


def chart_allowance_salary(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    if {"allowances", "netsalary"}.issubset(data.columns):
        normal = data[~data["flagged"]]
        flagged = data[data["flagged"]]
        ax.scatter(normal["allowances"], normal["netsalary"], label="Normal", alpha=.6)
        ax.scatter(flagged["allowances"], flagged["netsalary"], label="Flagged", marker="X")
        ax.set(xlabel="Allowances (₦)", ylabel="Net salary (₦)",
               title="Allowances vs Net Salary")
        ax.legend()
    else:
        ax.text(.5, .5, "allowances and netsalary columns are required",
                ha="center", va="center")
    fig.tight_layout()
    return fig


def logo_or_text(paths, width=80, fallback=""):
    for path in paths:
        if os.path.exists(path):
            st.image(path, width=width)
            return
    st.write(fallback)


def initialize_state():
    defaults = {
        "auth_verified": False,
        "full_name": "",
        "address": "",
        "contact": "",
        "step": 1,
        "dev_index": 0,
        "df": None,
        "results": None,
        "flagged_records": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(step):
    st.session_state.step = step
    st.rerun()


initialize_state()

# ------------------------------ Header ------------------------------
top_left, top_center, top_right = st.columns([1, 5, 1])
with top_left:
    logo_or_text(["coat_of_arms.jpeg", "coat_of_arms.jpg"], 75, "🇳🇬")
with top_center:
    st.markdown(
        """
        <div class="top-header">
            <p class="top-title">International Business School</p>
            <p class="top-sub">AI Payroll Anomaly Detection & Forensic Audit Portal</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_right:
    logo_or_text(
        ["school_logo.jpeg", "school_logo.jpg", "school_logo.png",
         "logo.jpeg", "logo.jpg", "logo.png"],
        75, "🏫 IBS",
    )
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# --------------------------- Navigation ----------------------------
st.markdown("### 📌 Navigation Menu")
step_names = {
    1: "Authorization",
    2: "Data Input",
    3: "AI Analysis",
    4: "Visualizations",
    5: "Q&A & Kudos",
    6: "Developers",
}
nav_cols = st.columns(6)
for number, name in step_names.items():
    with nav_cols[number - 1]:
        if st.button(f"{'👉 ' if st.session_state.step == number else ''}{number}. {name}",
                     key=f"nav_{number}", use_container_width=True):
            if number >= 2 and (
                not st.session_state.auth_verified
                or not st.session_state.full_name.strip()
            ):
                st.warning("Access locked. Complete authorization first.")
            elif number >= 3 and st.session_state.df is None:
                st.warning("Load or generate a dataset on Step 2 first.")
            elif number >= 4 and st.session_state.results is None:
                st.warning("Run AI analysis on Step 3 first.")
            else:
                go_to(number)

st.caption(f"Current section: {st.session_state.step}. {step_names[st.session_state.step]}")
st.divider()

# ------------------------ Step 1: Authorization ---------------------
if st.session_state.step == 1:
    st.header("🔑 Investigator Authorization & Registration")
    left, right = st.columns(2)
    with left:
        st.subheader("Authorization Check")
        choice = st.radio("Do you have official authorization?",
                          ["Please select", "YES", "NO"], key="auth_choice")
        st.session_state.auth_verified = choice == "YES"
        if choice == "YES":
            st.success("Authorization confirmed for this demonstration session.")
        elif choice == "NO":
            st.error("Access denied. Official authorization is required.")
        else:
            st.info("Select your authorization status.")
    with right:
        st.subheader("Investigator Details")
        st.session_state.full_name = st.text_input(
            "Full Name", value=st.session_state.full_name
        )
        st.session_state.address = st.text_input(
            "Official Address", value=st.session_state.address
        )
        st.session_state.contact = st.text_input(
            "Contact Number", value=st.session_state.contact
        )

    if st.session_state.auth_verified and st.session_state.full_name.strip():
        st.success(f"Session ready. Welcome, {st.session_state.full_name}.")
    if st.button("Continue to Data Input ➡️", use_container_width=True):
        if not st.session_state.auth_verified:
            st.error("Confirm that you have official authorization.")
        elif not st.session_state.full_name.strip():
            st.error("Enter your full name.")
        else:
            go_to(2)

# ------------------------ Step 2: Data Input ------------------------
elif st.session_state.step == 2:
    if not st.session_state.auth_verified or not st.session_state.full_name.strip():
        st.error("Complete Step 1 authorization first.")
        if st.button("Return to Authorization"):
            go_to(1)
        st.stop()

    st.header("📂 Payroll Data Input & Cleaning")
    source = st.radio(
        "Select dataset source",
        ["Generate Sample Payroll Dataset (500 Records)", "Upload External Payroll CSV"],
        horizontal=True,
        key="dataset_source",
    )

    if source.startswith("Generate"):
        if st.button("Generate 500-record sample dataset"):
            st.session_state.df = make_sample_data()
            st.session_state.results = None
            st.session_state.flagged_records = None
            st.success("Synthetic sample dataset generated.")
    else:
        uploaded = st.file_uploader("Upload payroll CSV file", type=["csv"])
        if uploaded is not None:
            try:
                loaded = pd.read_csv(uploaded)
                loaded = normalize_columns(loaded)
                if loaded.empty:
                    st.error("The uploaded CSV contains no records.")
                else:
                    st.session_state.df = loaded
                    st.session_state.results = None
                    st.session_state.flagged_records = None
                    st.success("CSV file loaded successfully.")
            except Exception as exc:
                st.error(f"Could not read the CSV file: {exc}")

    if st.session_state.df is not None:
        st.subheader("Dataset Preview")
        st.write(f"Rows: {len(st.session_state.df):,} | Columns: {len(st.session_state.df.columns)}")
        st.dataframe(st.session_state.df.head(15), use_container_width=True)
        st.download_button(
            "Download cleaned input CSV",
            st.session_state.df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_payroll_input.csv",
            mime="text/csv",
        )

    back, forward = st.columns(2)
    with back:
        if st.button("⬅️ Back to Authorization", use_container_width=True):
            go_to(1)
    with forward:
        if st.button("Continue to AI Analysis ➡️",
                     disabled=st.session_state.df is None,
                     use_container_width=True):
            go_to(3)

# ------------------------- Step 3: AI Analysis -----------------------
elif st.session_state.step == 3:
    if not st.session_state.auth_verified or st.session_state.df is None:
        st.error("Complete authorization and load a dataset first.")
        if st.button("Return to Data Input"):
            go_to(2)
        st.stop()

    st.header("🤖 AI Model Training & Anomaly Detection")
    sensitivity = st.slider("Expected anomaly sensitivity (%)", 1, 20, 7) / 100

    if st.button("Run AI Anomaly Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Analysing payroll records..."):
                results, used_features = run_anomaly_detection(
                    st.session_state.df, sensitivity
                )
            st.session_state.results = results
            st.session_state.flagged_records = results[results["flagged"]].copy()
            st.session_state.used_features = used_features
            st.success("Analysis completed.")
        except Exception as exc:
            st.session_state.results = None
            st.session_state.flagged_records = None
            st.error(str(exc))

    if st.session_state.results is not None:
        results = st.session_state.results
        flagged = st.session_state.flagged_records
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records analysed", f"{len(results):,}")
        m2.metric("Flagged for review", f"{len(flagged):,}")
        m3.metric("Not flagged", f"{len(results) - len(flagged):,}")
        m4.metric("Flagged percentage", f"{len(flagged) / len(results) * 100:.2f}%")
        st.caption("Numeric features used: " + ", ".join(st.session_state.used_features))
        st.subheader("🚩 Flagged Payroll Records")
        st.dataframe(flagged, use_container_width=True)
        st.download_button(
            "Download flagged records CSV",
            flagged.to_csv(index=False).encode("utf-8"),
            file_name="flagged_payroll_records.csv",
            mime="text/csv",
        )

    back, forward = st.columns(2)
    with back:
        if st.button("⬅️ Back to Data Input", use_container_width=True):
            go_to(2)
    with forward:
        if st.button("Continue to Visualizations ➡️",
                     disabled=st.session_state.results is None,
                     use_container_width=True):
            go_to(4)

# --------------------- Step 4: Visualizations ------------------------
elif st.session_state.step == 4:
    if st.session_state.results is None:
        st.error("Run the AI analysis first.")
        if st.button("Return to AI Analysis"):
            go_to(3)
        st.stop()

    st.header("📊 Payroll Analytics & Forensic Graphs")
    results = st.session_state.results

    st.subheader("1. PCA View and Regional Risk")
    c1, c2 = st.columns(2)
    with c1:
        safe_chart(chart_pca(results))
    with c2:
        safe_chart(chart_lga(results))

    st.divider()
    st.subheader("2. Anomaly Counts and Proportion")
    c3, c4 = st.columns(2)
    with c3:
        safe_chart(chart_status(results))
    with c4:
        safe_chart(chart_status(results, pie=True))

    st.divider()
    st.subheader("3. Salary Distribution and Financial Relationship")
    c5, c6 = st.columns(2)
    with c5:
        safe_chart(chart_salary(results))
    with c6:
        safe_chart(chart_allowance_salary(results))

    st.divider()
    st.subheader("📦 Download Audit Package")
    flagged = st.session_state.flagged_records
    summary = pd.DataFrame(
        {
            "ReportItem": [
                "System", "DateTime", "Investigator", "Contact",
                "TotalRecords", "FlaggedRecords",
            ],
            "Value": [
                "IBS AI Payroll Anomaly Detection Portal",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                st.session_state.full_name,
                st.session_state.contact,
                len(results),
                len(flagged),
            ],
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payroll_summary.csv", summary.to_csv(index=False))
        archive.writestr("full_analysis_results.csv", results.to_csv(index=False))
        archive.writestr("flagged_records.csv", flagged.to_csv(index=False))
    st.download_button(
        "📥 Download Audit ZIP Package",
        data=buffer.getvalue(),
        file_name=f"IBS_Payroll_Audit_{datetime.now():%Y%m%d_%H%M%S}.zip",
        mime="application/zip",
        use_container_width=True,
    )

    back, forward = st.columns(2)
    with back:
        if st.button("⬅️ Back to AI Analysis", use_container_width=True):
            go_to(3)
    with forward:
        if st.button("Continue to Q&A & Kudos ➡️", use_container_width=True):
            go_to(5)

# --------------------------- Step 5: Q&A -----------------------------
elif st.session_state.step == 5:
    if st.session_state.results is None:
        st.error("Complete the analysis before opening this section.")
        if st.button("Return to AI Analysis"):
            go_to(3)
        st.stop()

    st.header("❓ Project Q&A Assistant & User Kudos")
    question = st.text_input("Ask a question about the system")
    if question:
        q = question.lower()
        if "algorithm" in q or "isolation forest" in q:
            answer = (
                "The system uses Isolation Forest, an unsupervised machine-learning "
                "algorithm that identifies records that differ from common patterns."
            )
        elif "training" in q or "train" in q:
            answer = (
                "Isolation Forest learns patterns from the supplied numeric payroll "
                "features. It does not require pre-labelled fraud examples."
            )
        elif "contamination" in q or "sensitivity" in q:
            answer = (
                "The sensitivity setting is the expected proportion of unusual records "
                "the model should flag. It is a modelling setting, not a fraud rate."
            )
        elif "report" in q or "download" in q:
            answer = (
                "The audit ZIP contains a summary, the full analysis results and "
                "the records flagged for review."
            )
        elif "fraud" in q or "flag" in q:
            answer = (
                "A flagged record is an exception for investigation, not proof of fraud. "
                "Verify it against source documents and applicable payroll rules."
            )
        else:
            answer = "I do not have a prepared answer. Please consult the project team."
        st.info(answer)

    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        k_name = st.text_input("Your name or department", value=st.session_state.full_name)
        k_rating = st.select_slider(
            "Rate system performance",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Excellent",
        )
        k_comment = st.text_area("Comments", value="Great application!")
        submitted = st.form_submit_button("Submit Kudos")
        if submitted:
            st.success("Thank you for your feedback.")
            st.caption(f"Rating: {k_rating}. (This demo does not save feedback permanently.)")

    back, forward = st.columns(2)
    with back:
        if st.button("⬅️ Back to Visualizations", use_container_width=True):
            go_to(4)
    with forward:
        if st.button("Continue to Developers ➡️", use_container_width=True):
            go_to(6)

# ------------------------ Step 6: Developers -------------------------
elif st.session_state.step == 6:
    if st.session_state.results is None:
        st.error("Complete the analysis before opening this section.")
        if st.button("Return to AI Analysis"):
            go_to(3)
        st.stop()

    st.header("👥 Meet the Development Team")
    developers = [
        ("A. O. JUWON", "Ejuku, Yagba East LGA", "Kogi State",
         "NCE, B.Sc., M.Sc. (In View)", "Team Lead and Project Problem Analyzer", "Married"),
        ("ADEYEMI MARY MARVELOUS", "Odo-Ere, Yagba West LGA", "Kogi State",
         "B.Sc.", "AI Team", "Married"),
        ("IBITOYE MAYOMIKUN JONATHAN", "Isanlu, Yagba East LGA", "Kogi State",
         "B.Sc. (Engineering), University of Ilorin", "Data Collection Team", "Single"),
        ("OLUKAIYE OLUGBENGA TIMOTHY", "Bolorunduro, Kabba-Bunu LGA", "Kogi State",
         "B.Sc. Engineering, CUSTECH Osara", "Problem Solving Team", "Single"),
        ("KOMOLAFE BABATUNDE RICHARD", "Ayetoro-Gbede, Ijumu LGA", "Kogi State",
         "B.Sc.", "Demonstration Team", "Single"),
        ("ADEWUMI BISIRIYU TUNDE", "Ife-Olukotun, Yagba East LGA", "Kogi State",
         "B.Sc., CNA, NBA (in view)", "Data Collection Team", "Married"),
        ("BABALOLA MARVELOUS ITUNUOLUWA", "Egbe, Yagba West", "Kogi State",
         "B.Sc. Nursing, Federal University Lokoja", "Demonstration Team Member", "Single"),
        ("AKANKU TAIYE OLUWAPEMISIRE", "It edo Isanlu, Yagba East LGA", "Kogi State",
         "B.Sc. Economics", "Demonstration Team Member", "Married"),
        ("VICTOR IWALAYE", "Iyara, Ijumu LGA", "Kogi State",
         "National Diploma", "AI Team", "Single"),
        ("RACHEAL OLORUNMAIYE (Shellie)", "Kabba-Bunu LGA", "Kogi State",
         "B.Sc.", "Data Collection Team", "N/A"),
        ("ISREAL ADEOLA OLUMAIYEGUN", "Odo-Ere, Yagba West LGA", "Kogi State",
         "B.Sc. Economics, KSU Kabba", "Demonstration Team Member", "Single"),
        ("OLUWATOSIN MOSES O", "Odo-Ere, Yagba West", "Kogi State",
         "B.Sc. Sociology", "AI Team", "Married"),
    ]

    idx = st.session_state.dev_index % len(developers)
    previous, _, next_col = st.columns([1, 2, 1])
    with previous:
        if st.button("⬅️ Previous Developer", use_container_width=True):
            st.session_state.dev_index = (idx - 1) % len(developers)
            st.rerun()
    with next_col:
        if st.button("Next Developer ➡️", use_container_width=True):
            st.session_state.dev_index = (idx + 1) % len(developers)
            st.rerun()

    name, address, state, education, role, marital = developers[idx]
    image_name = f"developer {idx + 1}.jpeg"
    left, right = st.columns([1, 2])
    with left:
        alternate = image_name.replace(".jpeg", ".jpg")
        if os.path.exists(image_name):
            st.image(image_name, use_container_width=True)
        elif os.path.exists(alternate):
            st.image(alternate, use_container_width=True)
        else:
            st.info(f"Add {image_name} to the app folder to display this photo.")
    with right:
        st.subheader(name)
        st.markdown(f"📍 **Address:** {address}")
        st.markdown(f"🏛️ **State:** {state}")
        st.markdown(f"🎓 **Education:** {education}")
        st.markdown(f"💼 **Role:** {role}")
        st.markdown(f"💍 **Marital status:** {marital}")
        st.caption(f"Developer {idx + 1} of {len(developers)}")

    if st.button("⬅️ Back to Q&A Assistant & Kudos", use_container_width=True):
        go_to(5)

st.divider()
st.caption(
    "International Business School • AI Payroll Anomaly Detection Demonstration Portal"
)
st.warning(
    "Audit caution: Isolation Forest identifies statistical outliers only. "
    "A flagged payroll record must be independently verified; the model does not "
    "establish fraud, misconduct or a financial loss."
)
