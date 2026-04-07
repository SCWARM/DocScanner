import streamlit as st
import io
import base64
import pandas as pd
import fitz
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# ── Page setup ────────────────────────────────────────────────────────────────
# This is the very first thing Streamlit needs — sets the browser tab title,
# icon, and tells it to use the full screen width instead of a narrow column.
st.set_page_config(
    page_title="834 Enrollment Extractor",
    page_icon="🏥",
    layout="wide"
)

# ── Custom styling ─────────────────────────────────────────────────────────────
# Streamlit lets you inject raw CSS. This makes the app look more professional:
# - hides the default Streamlit menu and footer
# - styles the upload box and the download button
st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }
    .stFileUploader { border: 2px dashed #4A90D9; border-radius: 10px; padding: 10px; }
    .stDownloadButton > button {
        background-color: #1a7f37; color: white;
        border-radius: 8px; padding: 10px 20px;
        font-size: 16px; width: 100%;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 20px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏥 834 Enrollment Form Extractor")
st.caption("Upload scanned 834 PDFs or images — AI reads them and builds your Excel table automatically.")
st.divider()

# =============================================================================
# DATA MODELS
# These Pydantic classes define exactly what fields we want the AI to extract.
# Think of them as a template: the AI reads the form and fills in this template.
# If a field is missing on the form, it defaults to "N/A".
# =============================================================================

class Dependent(BaseModel):
    first_name:   str = "N/A"
    last_name:    str = "N/A"
    relationship: str = "N/A"
    dob:          str = Field(default="N/A", description="MM/DD/YYYY")
    ssn:          str = "N/A"
    gender:       str = "N/A"

class CoverageLine(BaseModel):
    coverage_type: str = Field(default="N/A", description="Medical, Dental, Vision, Life")
    plan_selected: str = "N/A"
    coverage_tier: str = Field(default="N/A", description="Employee Only, EE+Spouse, EE+Child, Family")

class Form834(BaseModel):
    first_name:  str = "N/A"
    last_name:   str = "N/A"
    ssn:         str = "N/A"
    dob:         str = Field(default="N/A", description="MM/DD/YYYY")
    gender:      str = "N/A"
    phone:       str = "N/A"
    email:       str = "N/A"
    address:     str = "N/A"
    city:        str = "N/A"
    state:       str = "N/A"
    zip_code:    str = "N/A"
    job_title:   str = "N/A"
    department:  str = "N/A"
    hire_date:   str = Field(default="N/A", description="MM/DD/YYYY")
    coverages:   List[CoverageLine] = Field(default_factory=list)
    dependents:  List[Dependent]    = Field(default_factory=list)

# =============================================================================
# AI CLIENT
# We use Google Gemini as our AI model, accessed through the OpenAI-compatible
# API. The `instructor` library wraps it so the AI returns structured data
# (our Form834 object) instead of plain text.
# =============================================================================

def get_client(api_key: str):
    return instructor.from_openai(
        OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        ),
        mode=instructor.Mode.JSON,
    )

# =============================================================================
# FILE READING HELPERS
# Two functions to handle different types of input:
# 1. read_pdf_text — fast path for digital PDFs that have a real text layer
# 2. pdf_to_images — for scanned PDFs, we convert each page to an image
#    so the AI can "see" it visually (vision mode)
# =============================================================================

def read_pdf_text(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(p.get_text() for p in doc)
    except Exception:
        return ""

def pdf_to_images(file_bytes: bytes) -> List[str]:
    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc[:10]:
            pix = page.get_pixmap(dpi=200)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images

# =============================================================================
# EXTRACTION
# This is the core function. It:
# 1. Checks if the file is a digital PDF (has text) or a scanned image
# 2. Sends the content to Gemini AI with a clear prompt
# 3. Gets back a structured Form834 object with all the fields filled in
# =============================================================================

PROMPT = (
    "Extract only the fields that a customer filled in by hand or typing on this "
    "834 enrollment form — personal info, contact details, employment, coverage "
    "selections, and dependents. Use 'N/A' for anything blank or missing."
)

def extract(file_bytes: bytes, filename: str, client) -> Form834 | None:
    ext  = filename.lower().rsplit(".", 1)[-1]
    text = read_pdf_text(file_bytes) if ext == "pdf" else ""

    # Route A: digital PDF — send the text directly (faster + cheaper)
    if len(text.strip()) > 50:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user",   "content": text}
        ]
    # Route B: scanned PDF or image — send as base64 images for vision
    else:
        images = pdf_to_images(file_bytes) if ext == "pdf" else [base64.b64encode(file_bytes).decode()]
        if not images:
            return None
        content = [{"type": "text", "text": PROMPT}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images
        ]
        messages = [{"role": "user", "content": content}]

    try:
        return client.chat.completions.create(
            model="gemini-2.5-flash",
            response_model=Form834,
            messages=messages,
            max_retries=2
        )
    except Exception as e:
        st.error(f"❌ Failed to process {filename}: {e}")
        return None

# =============================================================================
# FLATTEN TO ROWS
# The AI returns one Form834 object per document.
# But a subscriber with 3 dependents needs 3 Excel rows.
# This function "flattens" the nested structure into flat rows:
#   - subscriber info repeats on every row
#   - each row gets a different dependent's info
# =============================================================================

def to_rows(form: Form834, filename: str) -> List[dict]:
    coverages = " | ".join(
        f"{c.coverage_type}: {c.plan_selected} ({c.coverage_tier})"
        for c in form.coverages if c.coverage_type != "N/A"
    ) or "N/A"

    base = {
        "Source File": filename,
        "First Name":  form.first_name,
        "Last Name":   form.last_name,
        "SSN":         form.ssn,
        "DOB":         form.dob,
        "Gender":      form.gender,
        "Phone":       form.phone,
        "Email":       form.email,
        "Address":     form.address,
        "City":        form.city,
        "State":       form.state,
        "Zip":         form.zip_code,
        "Job Title":   form.job_title,
        "Department":  form.department,
        "Hire Date":   form.hire_date,
        "Coverages":   coverages,
    }

    deps = [d for d in form.dependents if d.first_name != "N/A"]
    if deps:
        rows = []
        for d in deps:
            row = base.copy()
            row.update({
                "Dep. First Name":   d.first_name,
                "Dep. Last Name":    d.last_name,
                "Dep. Relationship": d.relationship,
                "Dep. DOB":          d.dob,
                "Dep. SSN":          d.ssn,
                "Dep. Gender":       d.gender,
            })
            rows.append(row)
        return rows
    else:
        base.update({k: "N/A" for k in [
            "Dep. First Name", "Dep. Last Name", "Dep. Relationship",
            "Dep. DOB", "Dep. SSN", "Dep. Gender"
        ]})
        return [base]

# =============================================================================
# EXCEL GENERATOR
# Converts the pandas DataFrame into an Excel file stored in memory (BytesIO).
# We never write to disk — everything lives in RAM so the user can
# download it directly from the browser.
# =============================================================================

def generate_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="834 Enrollments")
        ws = writer.sheets["834 Enrollments"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 50)
    return output.getvalue()

# =============================================================================
# SIDEBAR — API KEY + INSTRUCTIONS
# The sidebar is always visible on the left. We ask for the API key here
# so it never gets hardcoded into the script or accidentally shared.
# =============================================================================

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your key here...")
    st.markdown("[Get a free key →](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("""
    **How to use:**
    1. Paste your Gemini API key above
    2. Upload your 834 PDF or image files
    3. Click **Extract Data**
    4. Preview the results on screen
    5. Download the Excel file

    **Supported formats:**
    - PDF (scanned or digital)
    - PNG, JPG, JPEG

    **Notes:**
    - One row per dependent per subscriber
    - Missing fields show as N/A
    - Up to 10 pages per PDF
    """)

# =============================================================================
# MAIN AREA — FILE UPLOAD + PROCESSING
# =============================================================================

uploaded_files = st.file_uploader(
    "Drop your 834 forms here",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can upload multiple files at once"
)

# Show file count badge once files are uploaded
if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} file(s) ready — click Extract Data when ready.")

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    run = st.button("⚡ Extract Data", type="primary", use_container_width=True)

st.divider()

# =============================================================================
# PROCESSING — runs when the button is clicked
# =============================================================================

if run:
    # Validate inputs before doing anything
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file.")
        st.stop()

    client  = get_client(api_key)
    records = []
    failed  = []

    # Summary metrics at the top
    m1, m2, m3 = st.columns(3)
    m1.metric("Files uploaded", len(uploaded_files))
    status_placeholder  = m2.empty()
    records_placeholder = m3.empty()

    progress = st.progress(0, text="Starting...")

    # Process each file one by one
    for i, uploaded_file in enumerate(uploaded_files):
        filename   = uploaded_file.name
        file_bytes = uploaded_file.read()

        progress.progress((i + 1) / len(uploaded_files), text=f"Processing {filename}...")
        status_placeholder.metric("Processing", f"{i+1} / {len(uploaded_files)}")

        form = extract(file_bytes, filename, client)

        if form:
            new_rows = to_rows(form, filename)
            records.extend(new_rows)
            records_placeholder.metric("Rows extracted", len(records))
        else:
            failed.append(filename)

    progress.empty()

    # ── Results ────────────────────────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)

        # Success banner
        st.success(f"✅ Done! Extracted **{len(records)} row(s)** from **{len(uploaded_files) - len(failed)} file(s)**.")

        if failed:
            st.warning(f"⚠️ {len(failed)} file(s) could not be processed: {', '.join(failed)}")

        # Data preview
        st.subheader("📊 Extracted Data Preview")
        st.dataframe(df, use_container_width=True, height=300)

        # Per-file summary
        with st.expander("📁 Per-file breakdown"):
            summary = df.groupby("Source File").size().reset_index(name="Rows extracted")
            st.dataframe(summary, use_container_width=True)

        # Download button
        st.subheader("📥 Download")
        st.download_button(
            label="Download Excel File",
            data=generate_excel(df),
            file_name="master_834_database.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error("❌ No data could be extracted. Check your files and API key and try again.")
