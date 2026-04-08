import streamlit as st
import io
import base64
import pandas as pd
import fitz
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

st.set_page_config(page_title="834 Enrollment Extractor", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.6rem; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("834 Enrollment Form Extractor")
st.caption("Upload scanned 834 PDFs or images. The AI reads them and outputs two linked tables — one for employees, one for dependents.")
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
# ID GENERATION
# Employees: C001, C002, C003 ...
# Dependents: C0011, C0012, C0021 ... (employee ID + dependent index)
# =============================================================================

def employee_id(n: int) -> str:
    return f"C{n:03d}"

def dependent_id(emp_id: str, i: int) -> str:
    return f"{emp_id}{i}"

# =============================================================================
# AI
# =============================================================================

def get_client(api_key: str):
    return instructor.from_openai(
        OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        ),
        mode=instructor.Mode.JSON,
    )

PROMPT = (
    "Extract only the fields that a customer filled in by hand or typing on this "
    "834 enrollment form - personal info, contact details, employment, coverage "
    "selections, and dependents. Use 'N/A' for anything blank or missing."
) 

def read_pdf_text(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            raw_text = "".join(p.get_text() for p in doc)
            # Sanitize the text: strip out weird hidden PDF typography characters
            return raw_text.encode("ascii", errors="ignore").decode("ascii")
    except Exception:
        return ""

def pdf_to_images(file_bytes: bytes) -> List[str]:
    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc[:10]:
            pix = page.get_pixmap(dpi=200)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images

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
            model="gemini-1.5-flash-8b", 
            response_model=Form834,
            messages=messages,
            max_retries=2
        )
    except Exception as e:
        st.error(f"Extraction failed for {filename}: {e}")
        return None

# =============================================================================
# BUILD TABLES
# =============================================================================

def build_tables(forms: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    emp_rows = []
    dep_rows = []

    for n, (form, filename) in enumerate(forms, start=1):
        eid = employee_id(n)

        coverages = " | ".join(
            f"{c.coverage_type}: {c.plan_selected} ({c.coverage_tier})"
            for c in form.coverages if c.coverage_type != "N/A"
        ) or "N/A"

        emp_rows.append({
            "Employee ID": eid,
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
        })

        for i, dep in enumerate([d for d in form.dependents if d.first_name != "N/A"], start=1):
            dep_rows.append({
                "Dependent ID": dependent_id(eid, i),
                "Employee ID":  eid,
                "First Name":   dep.first_name,
                "Last Name":    dep.last_name,
                "Relationship": dep.relationship,
                "DOB":          dep.dob,
                "SSN":          dep.ssn,
                "Gender":       dep.gender,
            })

    emp_df = pd.DataFrame(emp_rows)
    dep_df = pd.DataFrame(dep_rows) if dep_rows else pd.DataFrame(
        columns=["Dependent ID", "Employee ID", "First Name", "Last Name",
                 "Relationship", "DOB", "SSN", "Gender"]
    )
    return emp_df, dep_df

# =============================================================================
# EXCEL — two sheets
# =============================================================================

def to_excel(emp_df: pd.DataFrame, dep_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for df, sheet in [(emp_df, "Employees"), (dep_df, "Dependents")]:
            df.to_excel(writer, index=False, sheet_name=sheet)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                w = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(w + 2, 50)
    return output.getvalue()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### Configuration")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your API key")
    st.caption("[Get a free key](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("""
**Output structure**

Each uploaded form produces one row in the **Employees** sheet with a unique ID (C001, C002...).

Each dependent is a separate row in the **Dependents** sheet, linked to their employee by ID:

- Employee `C001`
- Dependent 1 → `C0011`
- Dependent 2 → `C0012`

**Supported files:** PDF, PNG, JPG
    """)

# =============================================================================
# MAIN
# =============================================================================

uploaded_files = st.file_uploader(
    "Upload 834 forms",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) selected.")

st.divider()

if st.button("Extract Data", type="primary"):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        st.stop()

    client  = get_client(api_key)
    forms   = []
    failed  = []
    progress = st.progress(0)

    for i, f in enumerate(uploaded_files):
        with st.spinner(f"Processing {f.name}..."):
            form = extract(f.read(), f.name, client)
        if form:
            forms.append((form, f.name))
        else:
            failed.append(f.name)
        progress.progress((i + 1) / len(uploaded_files))

    progress.empty()

    if not forms:
        st.error("No data could be extracted. Check your files and API key.")
        st.stop()

    emp_df, dep_df = build_tables(forms)

    st.success(f"Extracted {len(emp_df)} employee(s) and {len(dep_df)} dependent(s).")

    if failed:
        st.warning(f"Could not process: {', '.join(failed)}")

    tab1, tab2 = st.tabs([f"Employees  ({len(emp_df)})", f"Dependents  ({len(dep_df)})"])

    with tab1:
        st.dataframe(emp_df, use_container_width=True, height=350)

    with tab2:
        st.dataframe(dep_df, use_container_width=True, height=350)

    st.divider()
    st.download_button(
        label="Download Excel",
        data=to_excel(emp_df, dep_df),
        file_name="834_enrollment_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
