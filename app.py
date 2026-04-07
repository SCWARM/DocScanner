import streamlit as st
import io
import base64
import pandas as pd
import fitz  # PyMuPDF
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(page_title="834 Form Extractor", page_icon="📄", layout="wide")
st.title("📄 834 Enrollment Form Extractor")
st.write("Upload scanned PDFs or images to extract data into a formatted Excel file.")

# ── Data Schema (Unchanged) ───────────────────────────────────────────────────
class Dependent(BaseModel):
    first_name: str = "N/A"
    last_name: str = "N/A"
    relationship: str = "N/A"
    dob: str = Field(default="N/A", description="MM/DD/YYYY")
    ssn: str = "N/A"
    gender: str = "N/A"

class CoverageLine(BaseModel):
    coverage_type: str = Field(default="N/A", description="Medical, Dental, Vision, Life")
    plan_selected: str = "N/A"
    coverage_tier: str = Field(default="N/A", description="Employee Only, EE+Spouse, EE+Child, Family")

class Form834(BaseModel):
    first_name: str = "N/A"
    last_name: str = "N/A"
    ssn: str = "N/A"
    dob: str = Field(default="N/A", description="MM/DD/YYYY")
    gender: str = "N/A"
    phone: str = "N/A"
    email: str = "N/A"
    address: str = "N/A"
    city: str = "N/A"
    state: str = "N/A"
    zip_code: str = "N/A"
    job_title: str = "N/A"
    department: str = "N/A"
    hire_date: str = Field(default="N/A", description="MM/DD/YYYY")
    coverages: List[CoverageLine] = Field(default_factory=list)
    dependents: List[Dependent] = Field(default_factory=list)

# ── API Client ────────────────────────────────────────────────────────────────
def get_client(api_key: str):
    return instructor.from_openai(
        OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key),
        mode=instructor.Mode.JSON,
    )

# ── Web File Processing Helpers ───────────────────────────────────────────────
# Note: We now read from bytes in memory, not file paths!
def pdf_to_images(file_bytes):
    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc[:10]:
            pix = page.get_pixmap(dpi=200)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images

def read_pdf_text(file_bytes):
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(p.get_text() for p in doc)
    except Exception:
        return ""

# ── Extraction Logic ──────────────────────────────────────────────────────────
PROMPT = (
    "Extract only the fields that a customer filled in by hand or typing on this "
    "834 enrollment form — personal info, contact details, employment, coverage "
    "selections, and dependents. Use 'N/A' for anything blank or missing."
)

def extract(file_bytes, filename, client):
    ext = filename.lower().rsplit(".", 1)[-1]
    text = read_pdf_text(file_bytes) if ext == "pdf" else ""

    if len(text.strip()) > 50:
        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]
    else:
        images = pdf_to_images(file_bytes) if ext == "pdf" else [base64.b64encode(file_bytes).decode()]
        if not images: return None
        
        content = [{"type": "text", "text": PROMPT}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images
        ]
        messages = [{"role": "user", "content": content}]

    try:
        return client.chat.completions.create(
            model="gemini-2.5-flash", response_model=Form834, messages=messages, max_retries=2
        )
    except Exception as e:
        st.error(f"Error processing {filename}: {e}")
        return None

# ── Data Flattening ───────────────────────────────────────────────────────────
def to_rows(form, filename):
    coverages = " | ".join(
        f"{c.coverage_type}: {c.plan_selected} ({c.coverage_tier})"
        for c in form.coverages if c.coverage_type != "N/A"
    ) or "N/A"

    base = {
        "Source File": filename, "First Name": form.first_name, "Last Name": form.last_name,
        "SSN": form.ssn, "DOB": form.dob, "Gender": form.gender, "Phone": form.phone,
        "Email": form.email, "Address": form.address, "City": form.city, "State": form.state,
        "Zip": form.zip_code, "Job Title": form.job_title, "Department": form.department,
        "Hire Date": form.hire_date, "Coverages": coverages,
    }

    deps = [d for d in form.dependents if d.first_name != "N/A"]
    if deps:
        rows = []
        for d in deps:
            row = base.copy()
            row.update({
                "Dep. First Name": d.first_name, "Dep. Last Name": d.last_name,
                "Dep. Relationship": d.relationship, "Dep. DOB": d.dob,
                "Dep. SSN": d.ssn, "Dep. Gender": d.gender,
            })
            rows.append(row)
        return rows
    else:
        base.update({k: "N/A" for k in ["Dep. First Name", "Dep. Last Name", "Dep. Relationship", "Dep. DOB", "Dep. SSN", "Dep. Gender"]})
        return [base]

# ── Excel Generator ───────────────────────────────────────────────────────────
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="834 Enrollments")
        ws = writer.sheets["834 Enrollments"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 50)
    return output.getvalue()

# ── UI Layout ─────────────────────────────────────────────────────────────────
# Ask user for their own API key so you don't leak yours on a public site!
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.markdown("[Get a free key here](https://aistudio.google.com/app/apikey)")

uploaded_files = st.file_uploader("Upload 834 Forms", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Extract Data", type="primary"):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to continue.")
    elif not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        client = get_client(api_key)
        records = []
        
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            
            with st.spinner(f"Processing {filename}..."):
                form = extract(file_bytes, filename, client)
                if form:
                    records.extend(to_rows(form, filename))
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        if records:
            st.success(f"Successfully extracted {len(records)} rows!")
            df = pd.DataFrame(records)
            st.dataframe(df) # Show a preview on the web page!
            
            # Create the download button
            excel_data = generate_excel(df)
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name="master_834_database.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )