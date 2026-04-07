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
st.set_page_config(page_title="834 Enrollment Extractor", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }
    .stFileUploader { border: 2px dashed #4A90D9; border-radius: 10px; padding: 10px; }
    .stDownloadButton > button {
        background-color: #1a7f37; color: white;
        border-radius: 8px; padding: 10px 20px;
        font-size: 16px; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 834 Enrollment Form Extractor")
st.caption("Upload scanned 834 PDFs or images — AI reads them and builds your Excel tables automatically.")
st.divider()

# =============================================================================
# DATA MODELS
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
# ID GENERATOR
# Employees get sequential IDs: C001, C002, C003 ...
# Each dependent gets the employee ID + their position index:
#   Employee C009 → dependents C0091, C0092, C0093 ...
# =============================================================================

def make_employee_id(counter: int) -> str:
    return f"C{counter:03d}"          # C001, C002 ... C999

def make_dependent_id(employee_id: str, dep_index: int) -> str:
    return f"{employee_id}{dep_index}"  # C0091, C0092 ...

# =============================================================================
# AI CLIENT
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
# =============================================================================

PROMPT = (
    "Extract only the fields that a customer filled in by hand or typing on this "
    "834 enrollment form — personal info, contact details, employment, coverage "
    "selections, and dependents. Use 'N/A' for anything blank or missing."
)

def extract(file_bytes: bytes, filename: str, client) -> Form834 | None:
    ext  = filename.lower().rsplit(".", 1)[-1]
    text = read_pdf_text(file_bytes) if ext == "pdf" else ""

    if len(text.strip()) > 50:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user",   "content": text}
        ]
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
# SPLIT INTO TWO TABLES
#
# Instead of one flat table, we now produce two separate tables:
#
# EMPLOYEES TABLE — one row per file/subscriber
#   Columns: Employee ID | Source File | First Name | Last Name | SSN | ...
#
# DEPENDENTS TABLE — one row per dependent
#   Columns: Dependent ID | Employee ID | First Name | Last Name | Relationship | ...
#
# The Employee ID links the two tables together (like a foreign key in a database).
# Example:
#   Employee:  C009 | Mitchell, James | ...
#   Dependent: C0091 | C009 | Sarah | Mitchell | Spouse
#   Dependent: C0092 | C009 | Oliver | Mitchell | Child
#   Dependent: C0093 | C009 | Emma | Mitchell | Child
# =============================================================================

def build_tables(forms_with_files: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes a list of (Form834, filename) tuples.
    Returns (employees_df, dependents_df).
    """
    employee_rows  = []
    dependent_rows = []

    for counter, (form, filename) in enumerate(forms_with_files, start=1):
        emp_id = make_employee_id(counter)

        # Coverage summary string
        coverages = " | ".join(
            f"{c.coverage_type}: {c.plan_selected} ({c.coverage_tier})"
            for c in form.coverages if c.coverage_type != "N/A"
        ) or "N/A"

        # ── One employee row ──────────────────────────────────────────────────
        employee_rows.append({
            "Employee ID":  emp_id,
            "Source File":  filename,
            "First Name":   form.first_name,
            "Last Name":    form.last_name,
            "SSN":          form.ssn,
            "DOB":          form.dob,
            "Gender":       form.gender,
            "Phone":        form.phone,
            "Email":        form.email,
            "Address":      form.address,
            "City":         form.city,
            "State":        form.state,
            "Zip":          form.zip_code,
            "Job Title":    form.job_title,
            "Department":   form.department,
            "Hire Date":    form.hire_date,
            "Coverages":    coverages,
        })

        # ── One row per dependent ─────────────────────────────────────────────
        valid_deps = [d for d in form.dependents if d.first_name != "N/A"]
        for i, dep in enumerate(valid_deps, start=1):
            dep_id = make_dependent_id(emp_id, i)   # C0091, C0092 ...
            dependent_rows.append({
                "Dependent ID":  dep_id,
                "Employee ID":   emp_id,             # foreign key back to employee
                "First Name":    dep.first_name,
                "Last Name":     dep.last_name,
                "Relationship":  dep.relationship,
                "DOB":           dep.dob,
                "SSN":           dep.ssn,
                "Gender":        dep.gender,
            })

    employees_df  = pd.DataFrame(employee_rows)
    dependents_df = pd.DataFrame(dependent_rows) if dependent_rows else pd.DataFrame(
        columns=["Dependent ID", "Employee ID", "First Name", "Last Name",
                 "Relationship", "DOB", "SSN", "Gender"]
    )

    return employees_df, dependents_df

# =============================================================================
# EXCEL GENERATOR — two sheets in one file
# =============================================================================

def generate_excel(employees_df: pd.DataFrame, dependents_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Employees
        employees_df.to_excel(writer, index=False, sheet_name="Employees")
        ws_emp = writer.sheets["Employees"]
        ws_emp.freeze_panes = "A2"
        for col in ws_emp.columns:
            width = max(len(str(c.value or "")) for c in col)
            ws_emp.column_dimensions[col[0].column_letter].width = min(width + 2, 50)

        # Sheet 2: Dependents
        dependents_df.to_excel(writer, index=False, sheet_name="Dependents")
        ws_dep = writer.sheets["Dependents"]
        ws_dep.freeze_panes = "A2"
        for col in ws_dep.columns:
            width = max(len(str(c.value or "")) for c in col)
            ws_dep.column_dimensions[col[0].column_letter].width = min(width + 2, 50)

    return output.getvalue()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your key here...")
    st.markdown("[Get a free key →](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("""
    **How it works:**
    1. Each uploaded form becomes one **Employee** row with a unique ID (C001, C002...)
    2. Each dependent gets its own row in the **Dependents** sheet, linked back to the employee by ID (C0011, C0012...)
    3. Both sheets are in the same Excel file

    **Supported formats:** PDF, PNG, JPG

    **How to use:**
    1. Paste your API key above
    2. Upload your 834 forms
    3. Click Extract Data
    4. Preview both tables
    5. Download Excel
    """)

# =============================================================================
# MAIN UI
# =============================================================================

uploaded_files = st.file_uploader(
    "Drop your 834 forms here",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can upload multiple files at once"
)

if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} file(s) ready — click Extract Data when ready.")

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    run = st.button("⚡ Extract Data", type="primary", use_container_width=True)

st.divider()

# =============================================================================
# PROCESSING
# =============================================================================

if run:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file.")
        st.stop()

    client          = get_client(api_key)
    forms_with_files = []
    failed          = []

    m1, m2, m3 = st.columns(3)
    m1.metric("Files uploaded", len(uploaded_files))
    status_metric = m2.empty()
    rows_metric   = m3.empty()

    progress = st.progress(0, text="Starting...")

    for i, uploaded_file in enumerate(uploaded_files):
        filename   = uploaded_file.name
        file_bytes = uploaded_file.read()

        progress.progress((i + 1) / len(uploaded_files), text=f"Processing {filename}...")
        status_metric.metric("Processing", f"{i+1} / {len(uploaded_files)}")

        form = extract(file_bytes, filename, client)
        if form:
            forms_with_files.append((form, filename))
            rows_metric.metric("Forms extracted", len(forms_with_files))
        else:
            failed.append(filename)

    progress.empty()

    # ── Results ────────────────────────────────────────────────────────────────
    if forms_with_files:
        employees_df, dependents_df = build_tables(forms_with_files)

        st.success(f"✅ Done! **{len(employees_df)} employee(s)** and **{len(dependents_df)} dependent(s)** extracted.")

        if failed:
            st.warning(f"⚠️ Could not process: {', '.join(failed)}")

        # Preview both tables side by side
        tab1, tab2 = st.tabs([f"👤 Employees ({len(employees_df)} rows)", f"👨‍👩‍👧 Dependents ({len(dependents_df)} rows)"])

        with tab1:
            st.dataframe(employees_df, use_container_width=True, height=300)

        with tab2:
            st.dataframe(dependents_df, use_container_width=True, height=300)

            # Show how dependents link back to employees
            if not dependents_df.empty:
                with st.expander("🔗 How the IDs connect"):
                    link_df = dependents_df[["Dependent ID", "Employee ID", "First Name", "Last Name", "Relationship"]].copy()
                    emp_names = employees_df[["Employee ID", "First Name", "Last Name"]].rename(
                        columns={"First Name": "Emp. First", "Last Name": "Emp. Last"}
                    )
                    link_df = link_df.merge(emp_names, on="Employee ID", how="left")
                    st.dataframe(link_df, use_container_width=True)

        st.subheader("📥 Download")
        st.download_button(
            label="Download Excel File (2 sheets)",
            data=generate_excel(employees_df, dependents_df),
            file_name="master_834_database.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error("❌ No data could be extracted. Check your files and API key and try again.")
