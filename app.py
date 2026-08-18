import os
import re
from io import BytesIO
from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(
    page_title="DrugSafe",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")

REQUIRED_FILES = {
    "drugs": "drugs.csv",
    "interactions": "interactions.csv",
    "brands": "brands.csv",
    "contraindications": "contraindications.csv",
    "precautions": "precautions.csv",
    "monitoring": "monitoring.csv",
    "duplicate_therapy": "duplicate_therapy.csv",
    "patient_safety": "patient_safety.csv",
}

for key, default in {
    "analysis_done": False,
    "analysis_drugs": [],
    "report_bytes": None,
    "scroll_to_analysis": False,
    "active_module": "Interactions",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def normalize(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def compact(value):
    return re.sub(r"[^a-z0-9]", "", normalize(value))


def clean_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def paragraph_text(value):
    return escape(clean_value(value)).replace("\n", "<br/>")


@st.cache_data(show_spinner=False)
def load_database():
    loaded, missing = {}, []
    for key, filename in REQUIRED_FILES.items():
        path = os.path.join(DATABASE_DIR, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        try:
            loaded[key] = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            raise RuntimeError(f"Could not read {filename}: {exc}") from exc
    if missing:
        raise FileNotFoundError(
            "The following database file(s) were not found:\n\n" +
            "\n".join(missing)
        )
    return loaded


try:
    db = load_database()
    drug_df = db["drugs"].copy()
    interaction_df = db["interactions"].copy()
    brand_df = db["brands"].copy()
    contraindication_df = db["contraindications"].copy()
    precaution_df = db["precautions"].copy()
    monitoring_df = db["monitoring"].copy()
    duplicate_df = db["duplicate_therapy"].copy()
    patient_safety_df = db["patient_safety"].copy()
except Exception as exc:
    st.error("DrugSafe could not load its database.")
    st.code(str(exc))
    st.info("Make sure app.py is beside a database folder containing all 8 CSV files.")
    st.stop()

drug_column = "Drug" if "Drug" in drug_df.columns else (drug_df.columns[0] if not drug_df.empty else None)
if not drug_column:
    st.error("drugs.csv is empty.")
    st.stop()

if "Drug1" not in interaction_df.columns or "Drug2" not in interaction_df.columns:
    st.error("interactions.csv must contain Drug1 and Drug2 columns.")
    st.stop()


@st.cache_data(show_spinner=False)
def prepare_indexes(drugs, brands, interactions, duplicate_therapy):
    canonical_records = {}
    alias_to_canonical = {}

    for _, row in drugs.iterrows():
        drug_name = clean_value(row.get("Drug", ""))
        generic_name = clean_value(row.get("Generic Name", ""))
        canonical = drug_name or generic_name
        if not canonical:
            continue
        canonical_records[canonical] = row.to_dict()
        for alias in [drug_name, generic_name]:
            if alias:
                alias_to_canonical[normalize(alias)] = canonical
                alias_to_canonical[compact(alias)] = canonical

    brand_records = {}
    brand_to_generic = {}
    brand_to_canonical = {}

    if not brands.empty:
        brand_col = generic_col = None
        for column in brands.columns:
            n = normalize(column)
            if n in {"brand", "brand name"}:
                brand_col = column
            if n in {"generic", "generic name"}:
                generic_col = column

        if brand_col:
            for _, row in brands.iterrows():
                brand = clean_value(row.get(brand_col, ""))
                generic = clean_value(row.get(generic_col, "")) if generic_col else ""
                if not brand:
                    continue
                brand_records[brand] = row.to_dict()
                if generic:
                    brand_to_generic[normalize(brand)] = generic
                    brand_to_generic[compact(brand)] = generic
                    canonical = (
                        alias_to_canonical.get(normalize(generic))
                        or alias_to_canonical.get(compact(generic))
                    )
                    if canonical:
                        brand_to_canonical[normalize(brand)] = canonical
                        brand_to_canonical[compact(brand)] = canonical

    interaction_working = interactions.copy()
    for col, source in [("_D1", "Drug1"), ("_D2", "Drug2")]:
        interaction_working[col] = interaction_working[source].map(normalize)
    interaction_working["_D1C"] = interaction_working["Drug1"].map(compact)
    interaction_working["_D2C"] = interaction_working["Drug2"].map(compact)

    duplicate_working = duplicate_therapy.copy()
    if "Drug1" in duplicate_working.columns:
        duplicate_working["_D1"] = duplicate_working["Drug1"].map(normalize)
        duplicate_working["_D1C"] = duplicate_working["Drug1"].map(compact)
    if "Drug2" in duplicate_working.columns:
        duplicate_working["_D2"] = duplicate_working["Drug2"].map(normalize)
        duplicate_working["_D2C"] = duplicate_working["Drug2"].map(compact)

    drug_options = sorted(canonical_records.keys(), key=lambda x: x.lower())

    return (
        canonical_records, alias_to_canonical, brand_to_generic,
        brand_to_canonical, brand_records, interaction_working,
        duplicate_working, drug_options
    )


(
    canonical_records, alias_to_canonical, brand_to_generic,
    brand_to_canonical, brand_records, interaction_working,
    duplicate_working, drug_options
) = prepare_indexes(drug_df, brand_df, interaction_df, duplicate_df)


def canonical_drug(value):
    text = clean_value(value)
    if not text:
        return ""
    if " — " in text:
        text = text.split(" — ", 1)[0].strip()
    normal, short = normalize(text), compact(text)
    return (
        alias_to_canonical.get(normal)
        or alias_to_canonical.get(short)
        or brand_to_canonical.get(normal)
        or brand_to_canonical.get(short)
        or text
    )


@st.cache_data(show_spinner=False)
def cached_aliases(canonical, record_tuple, brand_alias_tuple):
    aliases_normal = {normalize(canonical)}
    aliases_compact = {compact(canonical)}
    record = dict(record_tuple)

    for field in ("Drug", "Generic Name"):
        value = clean_value(record.get(field, ""))
        if value:
            aliases_normal.add(normalize(value))
            aliases_compact.add(compact(value))

    for brand in brand_alias_tuple:
        if brand:
            aliases_normal.add(normalize(brand))
            aliases_compact.add(compact(brand))

    return aliases_normal, aliases_compact


def aliases_for_drug(value):
    canonical = canonical_drug(value)
    record = canonical_records.get(canonical, {})
    brands_for_drug = [
        brand for brand, mapped in brand_to_canonical.items()
        if mapped == canonical
    ]
    return cached_aliases(canonical, tuple(record.items()), tuple(brands_for_drug))


def canonicalize_selection(selected_labels):
    result, seen = [], set()
    for label in selected_labels:
        canonical = canonical_drug(label)
        key = compact(canonical)
        if canonical and key not in seen:
            result.append(canonical)
            seen.add(key)
    return result


def find_interactions(selected_drugs):
    results, signatures = [], set()
    if len(selected_drugs) < 2 or interaction_working.empty:
        return results

    for i in range(len(selected_drugs)):
        a = selected_drugs[i]
        a_normal, a_compact = aliases_for_drug(a)

        for j in range(i + 1, len(selected_drugs)):
            b = selected_drugs[j]
            b_normal, b_compact = aliases_for_drug(b)

            mask = (
                (interaction_working["_D1"].isin(a_normal) & interaction_working["_D2"].isin(b_normal))
                | (interaction_working["_D1"].isin(b_normal) & interaction_working["_D2"].isin(a_normal))
                | (interaction_working["_D1C"].isin(a_compact) & interaction_working["_D2C"].isin(b_compact))
                | (interaction_working["_D1C"].isin(b_compact) & interaction_working["_D2C"].isin(a_compact))
            )

            for _, row in interaction_working[mask].iterrows():
                d1, d2 = normalize(row.get("Drug1", "")), normalize(row.get("Drug2", ""))
                signature = (
                    frozenset([d1, d2]),
                    normalize(row.get("Severity", "")),
                    normalize(row.get("Type", "")),
                    normalize(row.get("Mechanism", "")),
                    normalize(row.get("Clinical Effect", "")),
                    normalize(row.get("Recommendation", "")),
                )
                if signature in signatures:
                    continue
                signatures.add(signature)
                results.append({"Drug1": a, "Drug2": b, "data": row.to_dict()})
    return results


def find_duplicate_therapy(selected_drugs):
    results, signatures = [], set()
    if len(selected_drugs) < 2 or duplicate_working.empty:
        return results
    if "_D1" not in duplicate_working.columns or "_D2" not in duplicate_working.columns:
        return results

    for i in range(len(selected_drugs)):
        a = selected_drugs[i]
        a_normal, a_compact = aliases_for_drug(a)

        for j in range(i + 1, len(selected_drugs)):
            b = selected_drugs[j]
            b_normal, b_compact = aliases_for_drug(b)

            mask = (
                (duplicate_working["_D1"].isin(a_normal) & duplicate_working["_D2"].isin(b_normal))
                | (duplicate_working["_D1"].isin(b_normal) & duplicate_working["_D2"].isin(a_normal))
                | (duplicate_working["_D1C"].isin(a_compact) & duplicate_working["_D2C"].isin(b_compact))
                | (duplicate_working["_D1C"].isin(b_compact) & duplicate_working["_D2C"].isin(a_compact))
            )

            for _, row in duplicate_working[mask].iterrows():
                d1, d2 = normalize(row.get("Drug1", "")), normalize(row.get("Drug2", ""))
                payload = tuple(sorted(
                    (str(k), normalize(v))
                    for k, v in row.to_dict().items()
                    if not str(k).startswith("_") and k not in {"Drug1", "Drug2"}
                ))
                signature = (frozenset([d1, d2]), payload)
                if signature in signatures:
                    continue
                signatures.add(signature)
                results.append({"Drug1": a, "Drug2": b, "data": row.to_dict()})
    return results


def merge_rows(rows):
    combined = {}
    for _, row in rows.iterrows():
        for column in rows.columns:
            if str(column).startswith("_"):
                continue
            value = clean_value(row.get(column, ""))
            if not value:
                continue
            if column not in combined:
                combined[column] = value
                continue
            existing_parts = [normalize(x) for x in str(combined[column]).split(" | ")]
            if normalize(value) not in existing_parts:
                combined[column] = f"{combined[column]} | {value}"
    return combined


def search_safety_database(database, selected_drugs):
    results = []
    if database.empty:
        return results

    match_columns = [
        col for col in ["Drug", "Medicine", "Generic", "Generic Name", "Medication"]
        if col in database.columns
    ]
    if not match_columns:
        return results

    normalized_lookup = {col: database[col].map(normalize) for col in match_columns}
    compact_lookup = {col: database[col].map(compact) for col in match_columns}

    for drug in selected_drugs:
        normal_aliases, compact_aliases = aliases_for_drug(drug)
        mask = pd.Series(False, index=database.index)
        for col in match_columns:
            mask |= normalized_lookup[col].isin(normal_aliases)
            mask |= compact_lookup[col].isin(compact_aliases)
        matched_rows = database[mask]
        if not matched_rows.empty:
            results.append({"Drug": drug, "data": merge_rows(matched_rows)})
    return results


def get_patient_safety(selected_drugs):
    return search_safety_database(patient_safety_df, selected_drugs)


def process_patient_safety(raw_results, renal, hepatic, pregnancy, lactation, age):
    patient_results = []
    for item in raw_results:
        safety, warnings = item["data"], []
        for active, column in [
            (renal, "Renal Impairment"),
            (hepatic, "Hepatic Impairment"),
            (pregnancy, "Pregnancy"),
            (lactation, "Lactation"),
            (age >= 65, "Older Adults"),
        ]:
            if active:
                value = clean_value(safety.get(column, ""))
                if value:
                    warnings.append((column, value))
        if warnings:
            patient_results.append({"Drug": item["Drug"], "warnings": warnings})
    return patient_results


def get_drug_information(selected_drugs):
    return {
        medicine: canonical_records.get(canonical_drug(medicine), {})
        for medicine in selected_drugs
    }


def scroll_to_analysis():
    components.html(
        """
        <script>
        setTimeout(function() {
            try {
                const doc = window.parent.document;
                const target = doc.getElementById("drug-safe-analysis-start");
                if (target) {
                    target.scrollIntoView({behavior: "smooth", block: "start"});
                }
            } catch (e) {}
        }, 250);
        </script>
        """,
        height=0,
    )


def generate_pdf(
    patient_data, selected_drugs, interaction_results,
    duplicate_results, patient_results, contraindication_results,
    precaution_results, monitoring_results, drug_information
):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DrugSafeTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, spaceAfter=8)
    subtitle_style = ParagraphStyle("DrugSafeSubtitle", parent=styles["Heading3"], alignment=TA_CENTER, fontSize=11, spaceAfter=16)
    heading_style = ParagraphStyle("DrugSafeHeading", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=7)
    normal_style = ParagraphStyle("DrugSafeNormal", parent=styles["Normal"], fontSize=8.7, leading=12)
    small_style = ParagraphStyle("DrugSafeSmall", parent=styles["Normal"], fontSize=7.7, leading=10)

    story = [
        Paragraph("DrugSafe", title_style),
        Paragraph("Medication Safety & Drug Interaction Report", subtitle_style),
        Paragraph("1. Patient Information", heading_style)
    ]

    patient_rows = [
        ["Patient Name", patient_data.get("name", "")],
        ["Patient ID", patient_data.get("id", "")],
        ["Age", patient_data.get("age", "")],
        ["Gender", patient_data.get("gender", "")],
        ["Weight", f"{patient_data.get('weight', '')} kg"],
        ["Height", f"{patient_data.get('height', '')} cm"],
        ["Known Allergies", patient_data.get("allergies", "")],
        ["Medical Conditions", patient_data.get("conditions", "")],
        ["Previous Medication History", patient_data.get("previous_medications", "")],
    ]

    table = Table(
        [[Paragraph(paragraph_text(a), small_style), Paragraph(paragraph_text(b), small_style)] for a, b in patient_rows],
        colWidths=[150, 360]
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 8)])

    story.append(Paragraph("2. Patient Safety Factors", heading_style))
    factors = [
        label for key, label in [
            ("renal", "Renal impairment"),
            ("hepatic", "Hepatic impairment"),
            ("pregnancy", "Pregnancy"),
            ("lactation", "Lactation"),
        ] if patient_data.get(key)
    ]
    story.append(Paragraph(paragraph_text(", ".join(factors) if factors else "None selected"), normal_style))

    story.append(Paragraph("3. Selected Medicines", heading_style))
    for medicine in selected_drugs:
        story.append(Paragraph("• " + paragraph_text(medicine), normal_style))

    story.append(Paragraph("4. Drug Interactions", heading_style))
    if interaction_results:
        for result in interaction_results:
            data = result["data"]
            story.append(Paragraph(f"<b>{paragraph_text(result['Drug1'])} + {paragraph_text(result['Drug2'])}</b>", normal_style))
            for key in ["Severity", "Type", "Mechanism", "Clinical Effect", "Recommendation", "Monitoring", "Dose Dependence", "Alternative", "Reference"]:
                value = clean_value(data.get(key, ""))
                if value:
                    story.append(Paragraph(f"<b>{paragraph_text(key)}:</b> {paragraph_text(value)}", small_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No matching interaction was found in the current DrugSafe database. This does not prove that no interaction exists.", normal_style))

    story.append(Paragraph("5. Duplicate Therapy", heading_style))
    if duplicate_results:
        for result in duplicate_results:
            story.append(Paragraph(f"<b>{paragraph_text(result['Drug1'])} + {paragraph_text(result['Drug2'])}</b>", normal_style))
            for key, value in result["data"].items():
                if str(key).startswith("_") or key in {"Drug1", "Drug2"}:
                    continue
                value = clean_value(value)
                if value:
                    story.append(Paragraph(f"<b>{paragraph_text(key)}:</b> {paragraph_text(value)}", small_style))
    else:
        story.append(Paragraph("No duplicate therapy record was identified.", normal_style))

    story.append(Paragraph("6. Patient-Specific Safety", heading_style))
    if patient_results:
        for item in patient_results:
            story.append(Paragraph(f"<b>{paragraph_text(item['Drug'])}</b>", normal_style))
            for label, value in item["warnings"]:
                story.append(Paragraph(f"<b>{paragraph_text(label)}:</b> {paragraph_text(value)}", small_style))
    else:
        story.append(Paragraph("No patient-specific warning was triggered from the selected safety factors.", normal_style))

    def add_database_section(title, records, empty_text):
        story.append(Paragraph(title, heading_style))
        if not records:
            story.append(Paragraph(paragraph_text(empty_text), normal_style))
            return
        for item in records:
            story.append(Paragraph(f"<b>{paragraph_text(item.get('Drug', ''))}</b>", normal_style))
            for key, value in item.get("data", {}).items():
                if key in {"Drug", "Medicine", "Generic", "Generic Name", "Medication"}:
                    continue
                value = clean_value(value)
                if value:
                    story.append(Paragraph(f"<b>{paragraph_text(key)}:</b> {paragraph_text(value)}", small_style))

    add_database_section("7. Contraindications", contraindication_results, "No matching contraindication record was found.")
    add_database_section("8. Precautions", precaution_results, "No matching precaution record was found.")
    add_database_section("9. Monitoring Parameters", monitoring_results, "No matching monitoring record was found.")

    story.append(Paragraph("10. Drug Information", heading_style))
    for medicine, data in drug_information.items():
        story.append(Paragraph(f"<b>{paragraph_text(medicine)}</b>", normal_style))
        if data:
            for key, value in data.items():
                value = clean_value(value)
                if value:
                    story.append(Paragraph(f"<b>{paragraph_text(key)}:</b> {paragraph_text(value)}", small_style))
        else:
            story.append(Paragraph("Drug information not found in drugs.csv.", small_style))

    story.append(Paragraph("Important Disclaimer", heading_style))
    story.append(Paragraph(
        "DrugSafe is a student medication-safety decision-support project. "
        "It is based on the current DrugSafe databases and should not replace professional clinical "
        "judgement, prescribing information, pharmacists, physicians, or authoritative references. "
        "A missing database result does not establish that a medicine or combination is safe.",
        normal_style
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# MODERN UI
# ============================================================

st.markdown("""
<style>
/* ============================================================
   DRUGSAFE VISUAL SYSTEM
   Background is self-contained: no external image/file needed.
   ============================================================ */

:root {
    --ds-bg: #06111f;
    --ds-bg-2: #081a2d;
    --ds-panel: rgba(10, 25, 43, 0.82);
    --ds-panel-strong: rgba(9, 22, 38, 0.94);
    --ds-border: rgba(56, 189, 248, 0.22);
    --ds-border-strong: rgba(56, 189, 248, 0.42);
    --ds-text: #f1f7ff;
    --ds-muted: #9fb4c9;
    --ds-blue: #22b8ff;
    --ds-blue-2: #008fe8;
    --ds-cyan: #48e0ff;
    --ds-green: #34d399;
}

/* Force the background onto Streamlit's actual app containers. */
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(0, 174, 255, 0.24) 0,
            rgba(0, 174, 255, 0.10) 18%,
            transparent 38%
        ),
        radial-gradient(
            circle at 92% 18%,
            rgba(34, 211, 238, 0.17) 0,
            rgba(34, 211, 238, 0.06) 20%,
            transparent 42%
        ),
        radial-gradient(
            circle at 72% 88%,
            rgba(14, 165, 233, 0.12) 0,
            transparent 34%
        ),
        linear-gradient(
            135deg,
            #04101d 0%,
            #071728 42%,
            #061221 70%,
            #03101b 100%
        ) !important;
    color: var(--ds-text) !important;
}

/* Pharmaceutical-style technical grid. */
[data-testid="stAppViewContainer"] {
    position: relative !important;
    min-height: 100vh;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.34;
    background-image:
        linear-gradient(
            rgba(56, 189, 248, 0.065) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(56, 189, 248, 0.065) 1px,
            transparent 1px
        );
    background-size: 44px 44px;
    mask-image: linear-gradient(
        to bottom,
        rgba(0,0,0,0.95),
        rgba(0,0,0,0.50) 55%,
        transparent 100%
    );
}

/* Large decorative medical/science glow. */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    width: 620px;
    height: 620px;
    right: -220px;
    top: 80px;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
    border: 1px solid rgba(72, 224, 255, 0.13);
    box-shadow:
        0 0 0 70px rgba(72, 224, 255, 0.025),
        0 0 0 140px rgba(72, 224, 255, 0.018),
        inset 0 0 100px rgba(34, 184, 255, 0.06);
}

/* Keep actual content above the decorative layer. */
[data-testid="stAppViewContainer"] .main {
    position: relative;
    z-index: 1;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* ============================================================
   BRAND / HEADER
   ============================================================ */

.drug-hero {
    position: relative;
    padding: 10px 0 20px 0;
}

.drug-title {
    font-size: 44px;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -1.8px;
    color: #f8fbff !important;
    text-shadow: 0 0 28px rgba(34, 184, 255, 0.18);
}

.drug-title span {
    color: var(--ds-cyan) !important;
}

.drug-subtitle {
    color: #a9bfd4 !important;
    font-size: 15px;
    margin-top: 8px;
}

/* ============================================================
   PANELS / CARDS
   ============================================================ */

.dashboard-panel {
    position: relative;
    background:
        linear-gradient(
            135deg,
            rgba(12, 31, 51, 0.90),
            rgba(7, 20, 35, 0.82)
        ) !important;
    border: 1px solid var(--ds-border) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin: 10px 0 !important;
    box-shadow:
        0 18px 55px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255,255,255,0.035) !important;
    backdrop-filter: blur(16px);
}

.section-card {
    display: inline-block;
    width: auto;
    max-width: 100%;
    background: rgba(11, 28, 47, 0.84) !important;
    border: 1px solid rgba(56, 189, 248, 0.16) !important;
    border-radius: 10px !important;
    padding: 10px 13px !important;
    margin: 5px 0 !important;
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
}

.compact-result {
    display: inline-block;
    max-width: 100%;
    background: rgba(8, 32, 52, 0.88) !important;
    border: 1px solid rgba(72, 224, 255, 0.30) !important;
    border-radius: 11px;
    padding: 10px 14px;
    margin: 5px 0;
    color: #eaf8ff !important;
}

.analysis-head {
    margin-top: 6px;
    margin-bottom: 14px;
}

.module-label {
    color: #64d9ff !important;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 7px 0 12px;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background:
        radial-gradient(
            circle at 20% 5%,
            rgba(0, 174, 255, 0.16),
            transparent 34%
        ),
        linear-gradient(
            180deg,
            #071525 0%,
            #06111f 55%,
            #040c16 100%
        ) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.28) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

[data-testid="stSidebar"] * {
    color: #eaf4ff !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background:
        linear-gradient(
            135deg,
            rgba(15, 39, 64, 0.96),
            rgba(8, 24, 41, 0.96)
        ) !important;
    border: 1px solid rgba(56, 189, 248, 0.23) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #8fb1c9 !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(56, 189, 248, 0.20) !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #8fa9bf !important;
}

/* ============================================================
   INPUTS
   ============================================================ */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(7, 24, 40, 0.94) !important;
    color: #eef8ff !important;
    border: 1px solid rgba(56, 189, 248, 0.20) !important;
    border-radius: 9px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(72, 224, 255, 0.75) !important;
    box-shadow: 0 0 0 1px rgba(72,224,255,0.22) !important;
}

[data-testid="stMultiSelect"] {
    background: rgba(7, 24, 40, 0.94) !important;
    border-radius: 9px !important;
}

[data-baseweb="select"] > div {
    background: rgba(7, 24, 40, 0.94) !important;
    border-color: rgba(56, 189, 248, 0.20) !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    background: #081a2d !important;
}

[data-baseweb="popover"] * {
    color: #eef8ff !important;
}

/* ============================================================
   EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    background: rgba(8, 24, 40, 0.86) !important;
    border: 1px solid rgba(56, 189, 248, 0.20) !important;
    border-radius: 12px !important;
    margin: 7px 0 !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] details summary {
    color: #edf7ff !important;
}

[data-testid="stExpander"] details summary:hover {
    background: rgba(34, 184, 255, 0.055) !important;
}

/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 42px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    background: linear-gradient(
        135deg,
        rgba(15, 39, 64, 0.98),
        rgba(9, 27, 46, 0.98)
    ) !important;
    color: #eef8ff !important;
    font-weight: 650 !important;
    transition: all .16s ease !important;
    box-shadow: 0 5px 18px rgba(0,0,0,.15) !important;
}

.stButton > button:hover {
    border-color: rgba(72, 224, 255, 0.72) !important;
    background: linear-gradient(
        135deg,
        #0b3a5b,
        #0b5279
    ) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(0, 174, 255, .16) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #0077c8 0%,
        #00aeea 55%,
        #26c6da 100%
    ) !important;
    border: 1px solid rgba(120, 235, 255, .55) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 28px rgba(0, 174, 255, .26) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(
        135deg,
        #0089df,
        #16c7f0,
        #39d8e7
    ) !important;
}

/* ============================================================
   BADGES / ALERTS / DIVIDERS
   ============================================================ */

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 750;
    margin-right: 6px;
}

.badge-major {
    background: rgba(127, 29, 29, .55);
    color: #fda4af !important;
}

.badge-moderate {
    background: rgba(120, 53, 15, .55);
    color: #fcd34d !important;
}

.badge-other {
    background: rgba(8, 67, 98, .65);
    color: #7dd3fc !important;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
}

hr {
    border-color: rgba(56, 189, 248, 0.17) !important;
}

/* ============================================================
   TEXT
   ============================================================ */

.stMarkdown,
.stCaption,
p,
label,
[data-testid="stMarkdownContainer"] {
    color: #e5eff9;
}

small,
[data-testid="stCaptionContainer"] {
    color: #9bb0c4 !important;
}
</style>
""", unsafe_allow_html=True)



# ============================================================
# FINAL DRUGSAFE UI OVERRIDES
# ============================================================

st.markdown(
    """
    <style>
    /* Keep Streamlit's header container alive because the sidebar
       expand button lives inside it. Hide the visual header itself. */
    [data-testid="stHeader"] {
        display: block !important;
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        z-index: 999999 !important;
    }

    /* The native Streamlit sidebar-open control must remain visible. */
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 1000000 !important;
        width: auto !important;
        height: auto !important;
    }

    [data-testid="stExpandSidebarButton"] button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 38px !important;
        height: 38px !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        color: #dbe7f5 !important;
        background: rgba(8, 24, 41, 0.94) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.28) !important;
        backdrop-filter: blur(10px) !important;
    }

    [data-testid="stExpandSidebarButton"] button:hover {
        color: #ffffff !important;
        background: #102944 !important;
        border-color: #48e0ff !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu, footer {
        visibility: hidden !important;
    }

    section.main > div {
        padding-top: 0 !important;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08111f 0%, #0b1627 55%, #09111d 100%) !important;
        border-right: 1px solid #24364d !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }

    /* ========================================================
       SIDEBAR SCROLL FIX
       Keep the sidebar in its existing position, but make the
       entire sidebar content independently scrollable so every
       item (including the top branding and bottom information)
       can be reached.
       ======================================================== */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        scrollbar-color: #3b6b8f transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        width: 7px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
        background: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: #315a78 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
        background: #48e0ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-bottom: 28px !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: .65rem !important;
        padding-bottom: .8rem !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #dbe7f5 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(20, 32, 50, .88) !important;
        border: 1px solid #2b405a !important;
        border-radius: 10px !important;
        padding: 7px 9px !important;
        margin: 0 0 7px 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #8fa6bf !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #29405b !important;
        margin: 10px 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #111d2e !important;
        border: 1px solid #2b405a !important;
    }

    
    /* Existing sidebar collapse button */
    [data-testid="stSidebarCollapseButton"] button {

        color: #dbe7f5 !important;
    }

    [data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid #2b405a !important;
    }

    .stButton > button {
        border-radius: 9px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="drug-hero">
    <div class="drug-title">💊 Drug<span>Safe</span></div>
    <div class="drug-subtitle">
        Medication Safety & Drug Interaction Decision-Support Tool
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.markdown(
        """
        <div id="sidebar-brand" style="
            padding: 2px 0 12px 0;
            margin: 0 0 12px 0;
            border-bottom: 1px solid #29405b;
        ">
            <div style="
                font-size: 25px;
                font-weight: 800;
                color: #60a5fa;
                letter-spacing: -0.6px;
            ">💊 DrugSafe</div>
            <div style="
                margin-top: 2px;
                color: #8fa6bf;
                font-size: 11px;
            ">Medication Safety System</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:10px;
            padding:4px 2px 14px 2px;
            margin-bottom:8px;
            border-bottom:1px solid rgba(56,189,248,.16);
        ">
            <div style="
                width:38px;
                height:38px;
                border-radius:11px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:linear-gradient(135deg,#0077c8,#22c6ef);
                box-shadow:0 0 24px rgba(34,198,239,.22);
                font-size:20px;
            ">💊</div>
            <div>
                <div style="font-size:18px;font-weight:800;color:#f8fbff;">
                    Drug<span style="color:#48e0ff;">Safe</span>
                </div>
                <div style="font-size:10px;letter-spacing:1.2px;color:#8fb1c9;">
                    MEDICATION SAFETY
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## 📊 Database")
    st.metric("Medicines", len(drug_df))
    st.metric("Interactions", len(interaction_df))
    st.metric("Brands", len(brand_df))
    st.divider()
    st.markdown("### Safety Modules")
    st.write(f"🚫 Contraindications  ·  {len(contraindication_df)}")
    st.write(f"⚠️ Precautions  ·  {len(precaution_df)}")
    st.write(f"📋 Monitoring  ·  {len(monitoring_df)}")
    st.write(f"🔁 Duplicate Therapy  ·  {len(duplicate_df)}")
    st.write(f"🛡️ Patient Safety  ·  {len(patient_safety_df)}")
    st.divider()

    st.markdown(
        '''
        <div class="ds-sidebar-info">
            <div class="ds-sidebar-version">DrugSafe <span>v3.0</span></div>
            <div class="ds-sidebar-info-text">
                Medication Safety &amp; Drug Interaction Decision-Support Tool
            </div>
            <div class="ds-sidebar-info-text">
                student research project
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.caption(
        "Does not replace professional clinical judgement."
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
st.markdown("### 👤 Patient Information")

c1, c2, c3 = st.columns(3)

with c1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])

with c2:
    patient_id = st.text_input("Patient ID")
    weight = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=0.0, value=0.0, step=0.1)

with c3:
    allergies = st.text_area("Known Allergies")
    conditions = st.text_area("Medical Conditions")
    previous_medications = st.text_area("Previous Medication History")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
st.markdown("### ⚕️ Patient Safety Factors")

s1, s2, s3, s4 = st.columns(4)
with s1:
    renal = st.checkbox("Renal impairment")
with s2:
    hepatic = st.checkbox("Hepatic impairment")
with s3:
    pregnancy = st.checkbox("Pregnancy")
with s4:
    lactation = st.checkbox("Lactation")

if patient_age >= 65:
    st.caption("👴 Age ≥ 65: older-adult considerations will be checked where available.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MEDICINE SELECTOR
# ============================================================

st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
st.markdown("### 💊 Select Medicines")
st.caption("Search and select medicines from the DrugSafe database.")

selected_labels = st.multiselect(
    "Search medicines",
    options=drug_options,
    key="selected_drugs",
    placeholder="Type a medicine name..."
)

selected_drugs = canonicalize_selection(selected_labels)

if selected_drugs:
    st.markdown(
        '<div class="compact-result"><b>Selected:</b> ' +
        " &nbsp;•&nbsp; ".join(escape(x) for x in selected_drugs) +
        "</div>",
        unsafe_allow_html=True
    )

button_col1, button_col2 = st.columns([1.25, 4])

with button_col1:
    button_text = "🔎 Check Interaction" if len(selected_drugs) >= 2 else "🔎 Check Safety"
    check_button = st.button(
        button_text,
        type="primary",
        use_container_width=True,
        disabled=not selected_drugs
    )

with button_col2:
    if st.session_state.analysis_done and st.session_state.analysis_drugs != selected_drugs:
        st.caption("Medicine selection changed. Click the check button to run a new analysis.")

st.markdown("</div>", unsafe_allow_html=True)


if check_button:
    st.session_state.analysis_done = True
    st.session_state.analysis_drugs = selected_drugs.copy()
    st.session_state.report_bytes = None
    st.session_state.scroll_to_analysis = True
    st.session_state.active_module = "Interactions"


analysis_active = st.session_state.analysis_done and bool(st.session_state.analysis_drugs)

if analysis_active:
    analysis_drugs = st.session_state.analysis_drugs

    st.markdown('<div id="drug-safe-analysis-start"></div>', unsafe_allow_html=True)

    if st.session_state.scroll_to_analysis:
        scroll_to_analysis()
        st.session_state.scroll_to_analysis = False

    st.divider()
    st.markdown('<div class="analysis-head">', unsafe_allow_html=True)
    st.markdown("## 🧪 Safety Analysis")
    st.caption("Choose a module below. Only the selected module is displayed.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Calculate all datasets once; only one is rendered.
    interaction_results = find_interactions(analysis_drugs) if len(analysis_drugs) >= 2 else []
    duplicate_results = find_duplicate_therapy(analysis_drugs)

    raw_patient_safety = get_patient_safety(analysis_drugs)
    patient_results = process_patient_safety(
        raw_patient_safety, renal, hepatic, pregnancy, lactation, patient_age
    )

    contraindication_results = search_safety_database(
        contraindication_df, analysis_drugs
    )
    precaution_results = search_safety_database(
        precaution_df, analysis_drugs
    )
    monitoring_results = search_safety_database(
        monitoring_df, analysis_drugs
    )
    drug_information = get_drug_information(analysis_drugs)

    # --------------------------------------------------------
    # MODULE BUTTONS
    # --------------------------------------------------------

    module_names = [
        ("🔎", "Interactions"),
        ("🚫", "Contraindications"),
        ("⚠️", "Precautions"),
        ("🔁", "Duplicate Therapy"),
        ("🛡️", "Patient Safety"),
        ("📋", "Monitoring"),
        ("📚", "Drug Information"),
    ]

    cols = st.columns(4)

    for idx, (icon, name) in enumerate(module_names):
        with cols[idx % 4]:
            label = f"{icon} {name}"
            if st.button(
                label,
                key=f"module_{name}",
                use_container_width=True
            ):
                st.session_state.active_module = name

    st.markdown(
        f'<div class="module-label">ACTIVE MODULE · {escape(st.session_state.active_module.upper())}</div>',
        unsafe_allow_html=True
    )

    active = st.session_state.active_module

    # --------------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------------

    if active == "Interactions":
        st.markdown("### 🔎 Drug Interactions")

        if len(analysis_drugs) < 2:
            st.info("Select at least two medicines to perform drug-drug interaction checking.")
        elif interaction_results:
            st.success(f"{len(interaction_results)} unique interaction record(s) found.")

            for result in interaction_results:
                data = result["data"]
                severity = normalize(data.get("Severity", ""))

                if severity == "major":
                    badge = '<span class="badge badge-major">MAJOR</span>'
                    icon = "🚨"
                elif severity == "moderate":
                    badge = '<span class="badge badge-moderate">MODERATE</span>'
                    icon = "⚠️"
                else:
                    badge = '<span class="badge badge-other">INFO</span>'
                    icon = "ℹ️"

                with st.expander(
                    f"{icon} {result['Drug1']} + {result['Drug2']}  ",
                    expanded=False
                ):
                    st.markdown(badge, unsafe_allow_html=True)

                    fields = [
                        "Type", "Mechanism", "Clinical Effect",
                        "Recommendation", "Monitoring",
                        "Dose Dependence", "Alternative"
                    ]

                    for key in fields:
                        value = clean_value(data.get(key, ""))
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(key)}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )

                    reference = clean_value(data.get("Reference", ""))
                    if reference:
                        st.caption(f"Reference: {reference}")
        else:
            st.success("No matching interaction found in the current DrugSafe database.")
            st.caption("This does not prove that no interaction exists.")

    # --------------------------------------------------------
    # CONTRAINDICATIONS
    # --------------------------------------------------------

    elif active == "Contraindications":
        st.markdown("### 🚫 Contraindications")

        if contraindication_results:
            for result in contraindication_results:
                with st.expander(f"🚫 {result['Drug']}", expanded=False):
                    for key, value in result["data"].items():
                        if normalize(key) in {
                            "drug", "medicine", "generic", "generic name", "medication"
                        }:
                            continue
                        value = clean_value(value)
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(str(key))}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )
        else:
            st.success("No matching contraindication record was found.")

    # --------------------------------------------------------
    # PRECAUTIONS
    # --------------------------------------------------------

    elif active == "Precautions":
        st.markdown("### ⚠️ Precautions")

        if precaution_results:
            for result in precaution_results:
                with st.expander(f"⚠️ {result['Drug']}", expanded=False):
                    for key, value in result["data"].items():
                        if normalize(key) in {
                            "drug", "medicine", "generic", "generic name", "medication"
                        }:
                            continue
                        value = clean_value(value)
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(str(key))}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )
        else:
            st.success("No matching precaution record was found.")

    # --------------------------------------------------------
    # DUPLICATE THERAPY
    # --------------------------------------------------------

    elif active == "Duplicate Therapy":
        st.markdown("### 🔁 Duplicate Therapy")

        if duplicate_results:
            for result in duplicate_results:
                with st.expander(
                    f"🔁 {result['Drug1']} + {result['Drug2']}",
                    expanded=False
                ):
                    for key, value in result["data"].items():
                        if str(key).startswith("_") or key in {"Drug1", "Drug2"}:
                            continue
                        value = clean_value(value)
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(str(key))}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )
        else:
            st.success("No duplicate therapy record was identified.")

    # --------------------------------------------------------
    # PATIENT SAFETY
    # --------------------------------------------------------

    elif active == "Patient Safety":
        st.markdown("### 🛡️ Patient-Specific Safety")

        if patient_results:
            for item in patient_results:
                with st.expander(f"⚕️ {item['Drug']}", expanded=False):
                    for label, value in item["warnings"]:
                        st.markdown(
                            f'<div class="section-card"><b>{escape(label)}</b><br>{escape(value)}</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.success(
                "No patient-specific warning was triggered from the selected safety factors "
                "and available database records."
            )

    # --------------------------------------------------------
    # MONITORING
    # --------------------------------------------------------

    elif active == "Monitoring":
        st.markdown("### 📋 Monitoring Parameters")

        if monitoring_results:
            for result in monitoring_results:
                with st.expander(f"📋 {result['Drug']}", expanded=False):
                    for key, value in result["data"].items():
                        if normalize(key) in {
                            "drug", "medicine", "generic", "generic name", "medication"
                        }:
                            continue
                        value = clean_value(value)
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(str(key))}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )
        else:
            st.success("No matching monitoring record was found.")

    # --------------------------------------------------------
    # DRUG INFORMATION
    # --------------------------------------------------------

    elif active == "Drug Information":
        st.markdown("### 📚 Drug Information")

        for medicine, data in drug_information.items():
            with st.expander(f"💊 {medicine}", expanded=False):
                if not data:
                    st.warning("Drug information not found in drugs.csv.")
                else:
                    for column, value in data.items():
                        value = clean_value(value)
                        if value:
                            st.markdown(
                                f'<div class="section-card"><b>{escape(str(column))}</b><br>{escape(value)}</div>',
                                unsafe_allow_html=True
                            )

    # --------------------------------------------------------
    # PDF REPORT
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 📄 Generate Report")

    patient_data = {
        "name": patient_name,
        "id": patient_id,
        "age": patient_age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "allergies": allergies,
        "conditions": conditions,
        "previous_medications": previous_medications,
        "renal": renal,
        "hepatic": hepatic,
        "pregnancy": pregnancy,
        "lactation": lactation,
    }

    if st.button("📄 Prepare PDF Report", type="secondary", use_container_width=True):
        with st.spinner("Preparing DrugSafe report..."):
            st.session_state.report_bytes = generate_pdf(
                patient_data,
                analysis_drugs,
                interaction_results,
                duplicate_results,
                patient_results,
                contraindication_results,
                precaution_results,
                monitoring_results,
                drug_information,
            )

    if st.session_state.report_bytes:
        st.download_button(
            "📥 Download DrugSafe PDF Report",
            data=st.session_state.report_bytes,
            file_name="DrugSafe_Medication_Safety_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


st.divider()
st.caption(
    "DrugSafe is a  student medication-safety decision-support project. "
    "It is database-based and does not replace professional clinical judgement. "
    "A missing result does not establish that a medicine or combination is safe."
)