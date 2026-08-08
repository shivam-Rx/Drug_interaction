import streamlit as st
import pandas as pd

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Drug Interaction Prediction Tool",
    page_icon="💊",
    layout="wide"
)

# ----------------------------------------------------
# Load Databases
# ----------------------------------------------------

interaction_df = pd.read_csv("database/interactions.csv")
drug_df = pd.read_csv("database/drugs.csv")
brand_df = pd.read_csv("database/brands.csv")


# ----------------------------------------------------
# Brand → Generic Conversion
# ----------------------------------------------------

def get_generic(drug_name):

    if drug_name is None:
        return None

    match = brand_df[
        brand_df["Brand"].str.strip().str.lower()
        == drug_name.strip().lower()
    ]

    if not match.empty:
        return match.iloc[0]["Generic"]

    return drug_name


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("💊 Drug Interaction Tool")

total_drugs = len(
    set(
        interaction_df["Drug1"].tolist()
        + interaction_df["Drug2"].tolist()
    )
)

st.sidebar.metric("💊 Drugs", total_drugs)
st.sidebar.metric("🔗 Interactions", len(interaction_df))

st.sidebar.metric(
    "🔴 Major",
    len(interaction_df[interaction_df["Severity"] == "Major"])
)

st.sidebar.metric(
    "🟡 Moderate",
    len(interaction_df[interaction_df["Severity"] == "Moderate"])
)

st.sidebar.metric(
    "⚫ Contraindicated",
    len(interaction_df[interaction_df["Severity"] == "Contraindicated"])
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍🎓 Developer: Shivam")
st.sidebar.write("B.Pharm Minor Project")
st.sidebar.success("Version 2.0")


# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.title("💊 AI-Assisted Drug Interaction Prediction Tool")

st.info(
    "Clinical Decision Support System for checking drug-drug interactions."
)

st.markdown("---")


# ----------------------------------------------------
# Drug List
# ----------------------------------------------------

generic_list = list(
    set(
        interaction_df["Drug1"].tolist()
        + interaction_df["Drug2"].tolist()
    )
)

brand_list = brand_df["Brand"].tolist()

drug_list = sorted(
    list(
        set(generic_list + brand_list)
    )
)


# ----------------------------------------------------
# Drug Selection
# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    drug1 = st.selectbox(
        "💊 Select First Drug",
        drug_list,
        index=None,
        placeholder="Choose a drug..."
    )

with col2:

    drug2 = st.selectbox(
        "💊 Select Second Drug",
        drug_list,
        index=None,
        placeholder="Choose a drug..."
    )
    


# ----------------------------------------------------
# Drug Information
# ----------------------------------------------------

st.markdown("## 💊 Drug Information")

info_col1, info_col2 = st.columns(2)
# ----------------------------------------------------
# Drug Information - Drug 1
# ----------------------------------------------------

with info_col1:

    if drug1 is not None:

        generic1 = get_generic(drug1)

        info1 = drug_df[
            drug_df["Drug"].str.lower() == generic1.lower()
        ]

        st.subheader(drug1)

        if generic1 != drug1:
            st.caption(f"Generic Name: {generic1}")

        if not info1.empty:

            st.write("**Class:**", info1.iloc[0]["Class"])
            st.write("**Indication:**", info1.iloc[0]["Indication"])
            st.write("**Route:**", info1.iloc[0]["Route"])

        else:

            st.warning("Drug information not available.")


# ----------------------------------------------------
# Drug Information - Drug 2
# ----------------------------------------------------

with info_col2:

    if drug2 is not None:

        generic2 = get_generic(drug2)

        info2 = drug_df[
            drug_df["Drug"].str.lower() == generic2.lower()
        ]

        st.subheader(drug2)

        if generic2 != drug2:
            st.caption(f"Generic Name: {generic2}")

        if not info2.empty:

            st.write("**Class:**", info2.iloc[0]["Class"])
            st.write("**Indication:**", info2.iloc[0]["Indication"])
            st.write("**Route:**", info2.iloc[0]["Route"])

        else:

            st.warning("Drug information not available.")

st.markdown("---")


# ----------------------------------------------------
# Interaction Checker
# ----------------------------------------------------

if st.button("🔍 Check Interaction"):

    if drug1 is None or drug2 is None:

        st.warning("⚠ Please select both drugs.")
        st.stop()

    generic1 = get_generic(drug1)
    generic2 = get_generic(drug2)

    drug1_lower = generic1.strip().lower()
    drug2_lower = generic2.strip().lower()

    interaction_df["Drug1_lower"] = (
        interaction_df["Drug1"].str.strip().str.lower()
    )

    interaction_df["Drug2_lower"] = (
        interaction_df["Drug2"].str.strip().str.lower()
    )

    result = interaction_df[
        (
            (interaction_df["Drug1_lower"] == drug1_lower)
            &
            (interaction_df["Drug2_lower"] == drug2_lower)
        )
        |
        (
            (interaction_df["Drug1_lower"] == drug2_lower)
            &
            (interaction_df["Drug2_lower"] == drug1_lower)
        )
    ]

    if not result.empty:

        row = result.iloc[0]

        severity = row["Severity"]
                # -----------------------------
        # Severity
        # -----------------------------

        if severity == "Major":
            risk = 90
            st.error("🔴 MAJOR INTERACTION")

        elif severity == "Moderate":
            risk = 60
            st.warning("🟡 MODERATE INTERACTION")

        elif severity == "Minor":
            risk = 30
            st.success("🟢 MINOR INTERACTION")

        elif severity == "Contraindicated":
            risk = 100
            st.error("⚫ CONTRAINDICATED")

        else:
            risk = 0

        # -----------------------------
        # Risk Meter
        # -----------------------------

        st.subheader("⚠ Clinical Risk")

        st.progress(risk)

        st.write(f"### {risk}% Risk Level")

        st.markdown("---")

        # -----------------------------
        # Interaction Details
        # -----------------------------

        st.subheader("📋 Interaction Details")

        st.write("**Drug 1:**", row["Drug1"])
        st.write("**Drug 2:**", row["Drug2"])
        st.write("**Severity:**", row["Severity"])
        st.write("**Type:**", row["Type"])
        st.write("**Mechanism:**", row["Mechanism"])
        st.write("**Clinical Effect:**", row["Clinical Effect"])

        st.success(
            f"**Recommendation:** {row['Recommendation']}"
        )

        if "Reference" in row.index:
            st.info(
                f"📚 Reference: {row['Reference']}"
            )

        # -----------------------------
        # Patient Counseling
        # -----------------------------

        st.markdown("---")
        st.subheader("👨‍⚕️ Patient Counseling")

        if severity == "Major":

            st.warning("""
• Watch for unusual bleeding or bruising.

• Inform your doctor immediately if symptoms occur.

• Avoid self-medication.

• Attend follow-up appointments.
""")

        elif severity == "Moderate":

            st.info("""
• Continue medicines as prescribed.

• Monitoring may be required.

• Report unusual symptoms.
""")

        elif severity == "Minor":

            st.success("""
• Usually safe.

• Continue therapy as advised.
""")

        elif severity == "Contraindicated":

            st.error("""
🚫 These medicines should NEVER be used together.

Seek immediate medical advice.
""")

    else:

        st.info("✅ No interaction found in the database.")


# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")
st.caption("Developed using Python • Streamlit • Pandas")