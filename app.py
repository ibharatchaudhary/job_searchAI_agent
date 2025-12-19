import streamlit as st
import google.generativeai as genai
import io
import PyPDF2
import docx
import json
import time
import requests
import re
import os
import urllib.parse
from streamlit.components.v1 import html as st_html

# =========================
# --- Page / Config
# =========================
st.set_page_config(page_title="Automated Job Search Agent", page_icon="🤖", layout="wide")

# token-saver knobs
PREFILTER_TOP_N = 20    # keep top N by cheap overlap before AI
AI_SCORE_TOP_N  = 10    # call AI on only top K
JOB_DESC_CAP    = 2000  # max chars of JD sent to AI
MAX_RETRIES     = 3

# =========================
# --- Helper: File Parsing
# =========================

def extract_text_from_pdf(file_bytes):
    """Extracts text from an in-memory PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_docx(file_bytes):
    """Extracts text from an in-memory DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None

# =========================
# --- Helper: JSON Parsing
# =========================

def clean_and_parse_json(raw_text):
    """
    Extract only the JSON part from an AI response and parse it.
    Handles responses wrapped in markdown code blocks.
    """
    if not raw_text:
        return None
    match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', raw_text, re.DOTALL)
    json_str = match.group(1) if match else raw_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON. Error: {e}. Response text (truncated): {json_str[:500]}...")
        return None

# =========================
# --- Helper: Secrets
# =========================

def _get_optional_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None

# =========================
# --- Gemini Setup Helpers
# =========================

PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]

def init_gemini_model(api_key: str, generation_config, system_instruction: str, model_list=None):
    """
    Initialize a GenerativeModel with stable model IDs.
    Falls back to any available model that supports generateContent.
    """
    genai.configure(api_key=api_key)
    last_error = None
    model_list = model_list or PREFERRED_MODELS

    # Try preferred models
    for name in model_list:
        try:
            return genai.GenerativeModel(
                name,
                generation_config=generation_config,
                system_instruction=system_instruction,
            )
        except Exception as e:
            last_error = e
            continue

    # Dynamic fallback: discover a model that supports generateContent
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                try:
                    return genai.GenerativeModel(
                        m.name,
                        generation_config=generation_config,
                        system_instruction=system_instruction,
                    )
                except Exception as e:
                    last_error = e
                    continue
    except Exception as e:
        last_error = e

    st.error(f"Failed to initialize Gemini model. Last error: {last_error}")
    return None

# =========================
# --- Refining Agent
# =========================

def refine_skills_ai(raw_skills, api_key):
    """Ask Gemini to refine skills list to only keep relevant professional skills."""
    if not raw_skills:
        return []
    try:
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        system_instruction = (
            "You are a career expert. From the given list of extracted 'skills', "
            "remove irrelevant items like social media apps, email providers, or non-skills. "
            "Keep only valid professional, technical, or soft skills. "
            "Return them as a JSON array of strings."
        )
        model = init_gemini_model(api_key, generation_config, system_instruction)
        if not model:
            return raw_skills

        skills_text = ", ".join(raw_skills)
        response = model.generate_content([f"Skills: {skills_text}"])
        refined = clean_and_parse_json(getattr(response, "text", None))
        return refined or raw_skills
    except Exception as e:
        st.warning(f"Skill refinement failed: {e}")
        return raw_skills

# =========================
# --- Gemini Parsing / Matching
# =========================

def get_gemini_response(text, api_key, data_type="resume"):
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"API Key Configuration Error: {e}")
        return None

    if data_type == "resume":
        output_schema = {
            "type": "OBJECT",
            "properties": {
                "personalInformation": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "email": {"type": "STRING"},
                        "phone": {"type": "STRING"},
                    },
                },
                "summary": {"type": "STRING"},
                "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "experience": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING"},
                            "company": {"type": "STRING"},
                            "dates": {"type": "STRING"},
                            "description": {"type": "STRING"},
                        },
                    },
                },
                "projects": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "name": {"type": "STRING"},
                            "description": {"type": "STRING"},
                        },
                    },
                },
                "education": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "degree": {"type": "STRING"},
                            "institution": {"type": "STRING"},
                            "university": {"type": "STRING"},
                            "dates": {"type": "STRING"},
                        },
                    },
                },
            },
        }
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=output_schema
        )
        system_instruction = (
            "You are an expert HR and technical recruiter. Extract key information from the resume. "
            "The 'personalInformation' section with name, email, and phone is mandatory. "
            "Identify all relevant technical and soft skills as a flat list. "
            "Return the full data as a structured JSON object."
        )
        prompt = f"Parse this resume:\n\n{text}"

    elif data_type == "match":
        output_schema = {
            "type": "OBJECT",
            "properties": {
                "match_score": {"type": "NUMBER"},
                "justification": {"type": "STRING"},
            },
        }
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=output_schema
        )
        system_instruction = (
            "You are a job matching expert. Compare the user's skills with the job description "
            "and return a match score and justification as JSON."
        )
        prompt = text
    else:
        return None

    model = init_gemini_model(api_key, generation_config, system_instruction)
    if model is None:
        return None

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content([prompt])
            response_text = getattr(response, "text", None)

            if not response_text and hasattr(response, "candidates"):
                try:
                    parts = response.candidates[0].content.parts
                    response_text = "".join(getattr(p, "text", "") for p in parts)
                except Exception:
                    response_text = None

            parsed_json = clean_and_parse_json(response_text)
            return parsed_json
        except Exception as e:
            last_error = e
            if "API key not valid" in str(e):
                st.error("Your Google AI API key is not valid. Please check and re-enter.")
                return None
            if "quota" in str(e).lower():
                st.error("Gemini API quota exceeded. Try again later or use a different key.")
                return None
            if attempt < MAX_RETRIES - 1:
                st.warning(f"API call failed on attempt {attempt + 1}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                st.error(f"API Error after multiple retries: {e}")
                return None
    return None

# =========================
# --- Job Search (SerpApi) + cache
# =========================

@st.cache_data(show_spinner=False, ttl=300)
def search_jobs(serpapi_key, job_title, location):
    params = {
        "engine": "google_jobs",
        "q": job_title,
        "location": location,
        "api_key": serpapi_key,
        "hl": "en",
        "gl": "us",
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            return {"error": payload.get("error"), "jobs": []}
        jobs = payload.get("jobs_results", []) or []
        return {"error": None, "jobs": jobs}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "jobs": []}

# =========================
# --- Apply Links Helpers
# =========================

def extract_apply_links(job: dict):
    """
    Returns a list of (label, url) apply options from a SerpApi job object.
    Priority: apply_options[].link -> share_link/link -> related_links (legacy).
    """
    links = []

    # Preferred: direct ATS / job board links
    for opt in job.get("apply_options", []) or []:
        url = opt.get("link")
        title = opt.get("title") or "Apply"
        if url:
            links.append((title, url))

    # Fallback: Google Jobs listing links
    if not links:
        for k in ("share_link", "link"):
            if job.get(k):
                links.append(("View on Google Jobs", job[k]))
                break

    # Legacy fallback: related_links
    if not links:
        rl = job.get("related_links", [])
        if rl and rl[0].get("link"):
            links.append(("Apply", rl[0]["link"]))

    return links

def clean_url(url):
    """
    Removes Google redirect wrappers like:
    https://www.google.com/url?q=<real link>
    and returns the actual job application link.
    """
    if not url:
        return url
    if "url?q=" in url:
        parsed = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(parsed.query).get("q", [url])[0]
        return q
    return url

def open_same_tab(url: str):
    """Force same-tab navigation (hard redirect)."""
    st_html(f"""
        <script>
            window.location.replace({json.dumps(url)});
        </script>
    """, height=0)

# =========================
# --- Experience Filter Helpers
# =========================

EXPERIENCE_LEVELS = [
    "Any",
    "Internship",
    "Entry (0-1)",
    "Junior (1-3)",
    "Mid (3-5)",
    "Senior (5-8)",
    "Lead (8+)",
]

def bucket_from_years(y):
    if y is None: return "Any"
    if y < 1:     return "Entry (0-1)"
    if y < 3:     return "Junior (1-3)"
    if y < 5:     return "Mid (3-5)"
    if y < 8:     return "Senior (5-8)"
    return "Lead (8+)"

def infer_experience_level(job: dict):
    """
    Heuristic from job title/description:
    - years like '3+ years', '2 years' -> numeric bucket
    - keywords: intern/entry/junior/senior/lead/principal/staff
    Returns one of EXPERIENCE_LEVELS.
    """
    title = (job.get("title") or "").lower()
    desc  = (job.get("description") or "").lower()
    text  = f"{title}\n{desc}"

    # keyword buckets (strong signals)
    if re.search(r'\bintern(ship)?\b', text): return "Internship"
    if re.search(r'\b(entry[- ]?level|graduate|new grad)\b', text): return "Entry (0-1)"
    if re.search(r'\bjunior\b|jr\.', text): return "Junior (1-3)"
    if re.search(r'\bsenior\b|sr\.|senior-level', text): return "Senior (5-8)"
    if re.search(r'\b(principal|staff|lead|manager)\b', text): return "Lead (8+)"

    # numeric years
    m = re.search(r'(\d{1,2})\s*\+?\s*(?:years?|yrs)', text)
    if m:
        years = int(m.group(1))
        return bucket_from_years(years)

    return "Any"

# =========================
# --- Cheap Prefilter Scorer
# =========================

def rough_overlap_score(user_skills_set, job_text):
    if not job_text: return 0
    jt = job_text.lower()
    hits = sum(1 for s in user_skills_set if s in jt)
    return hits

# =========================
# --- Session State
# =========================
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'job_results' not in st.session_state:
    st.session_state.job_results = None

# =========================
# --- Sidebar UI
# =========================
with st.sidebar:
    st.image("https://placehold.co/400x100/3b82f6/ffffff?text=Job+Agent+AI&font=inter", use_column_width=True)
    st.title("1. Configuration")
    google_api_key_input = st.text_input("Google AI API Key", type="password")
    serpapi_key_input = st.text_input("SerpApi Key", type="password")
    google_api_key = google_api_key_input or os.environ.get("GOOGLE_API_KEY", "") or (_get_optional_secret("GOOGLE_API_KEY") or "")
    serpapi_key = serpapi_key_input or os.environ.get("SERPAPI_KEY", "") or (_get_optional_secret("SERPAPI_KEY") or "")

    st.markdown("---")
    st.title("2. Parse Resume")
    uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx"])

    if st.button("Parse Resume", use_container_width=True, type="primary"):
        if google_api_key and uploaded_file:
            with st.spinner("Analyzing resume..."):
                file_bytes = uploaded_file.getvalue()
                raw_text = extract_text_from_pdf(file_bytes) if uploaded_file.type == "application/pdf" else extract_text_from_docx(file_bytes)
                if raw_text:
                    parsed = get_gemini_response(raw_text, google_api_key, "resume")
                    if parsed:
                        parsed["skills"] = refine_skills_ai(parsed.get("skills", []), google_api_key)
                        st.session_state.parsed_data = parsed
                        st.success("Resume parsed and skills refined!")
                    else:
                        st.error("Failed to parse.")
        else:
            st.warning("Please provide API Key and upload a resume.")

    st.markdown("---")
    st.title("3. Find Jobs")
    job_title = st.text_input("Job Title", "Software Engineer")
    location = st.text_input("Location", "United States")
    experience_filter = st.selectbox("Experience Level", EXPERIENCE_LEVELS, index=0)

    if st.button("Find Matching Jobs", use_container_width=True, type="primary"):
        if st.session_state.parsed_data and serpapi_key:
            with st.spinner(f"Searching for '{job_title}' roles..."):
                res = search_jobs(serpapi_key, job_title, location)
            if res["error"]:
                st.error(f"Job search failed: {res['error']}")
            else:
                jobs = res["jobs"]

                # apply experience filter
                if experience_filter != "Any":
                    filtered = []
                    for j in jobs:
                        lvl = infer_experience_level(j)
                        j["_inferred_experience"] = lvl
                        if lvl == experience_filter:
                            filtered.append(j)
                    jobs = filtered

                # cheap prefilter
                job_matches = []
                if jobs:
                    user_skills_list = st.session_state.parsed_data.get("skills", [])
                    user_skills = sorted(list(set([s.lower() for s in user_skills_list])))
                    user_skills_set = set(user_skills)

                    # rough score on all jobs
                    for j in jobs:
                        j["_rough"] = rough_overlap_score(user_skills_set, (j.get("description") or ""))

                    # keep the best N for AI scoring
                    jobs = sorted(jobs, key=lambda x: x.get("_rough", 0), reverse=True)[:PREFILTER_TOP_N]

                    with st.spinner("Calculating match scores with AI..."):
                        if not user_skills:
                            st.warning("Could not find skills in the resume to match jobs against.")
                        else:
                            for idx, job in enumerate(jobs[:AI_SCORE_TOP_N]):
                                job_desc = (job.get('description') or "")[:JOB_DESC_CAP]
                                prompt_for_match = f"USER SKILLS: {', '.join(user_skills)}\n\nJOB DESCRIPTION: {job_desc}"
                                match_result = get_gemini_response(prompt_for_match, google_api_key, "match")
                                if match_result:
                                    job['match_score'] = match_result.get('match_score', 0)
                                    job['justification'] = match_result.get('justification', 'N/A')
                                else:
                                    job['match_score'] = 0
                                    job['justification'] = 'Could not calculate score.'
                                job_matches.append(job)

                st.session_state.job_results = sorted(job_matches, key=lambda x: x.get('match_score', 0), reverse=True)
                st.success(f"Ranked {len(st.session_state.job_results)} jobs!")
        else:
            st.warning("Please parse a resume and enter a SerpApi Key first.")

# =========================
# --- Main UI
# =========================
st.title("Automated Job Search Agent")
st.markdown("#### Complete Job Search Workflow")

tab1, tab2 = st.tabs(["📄 Parsed Resume Details", "🎯 Job Matches"])

with tab1:
    st.header("Extracted Resume Information")
    if st.session_state.parsed_data:
        data = st.session_state.parsed_data
        personal_info = data.get("personalInformation", {})
        name = personal_info.get("name") or data.get("name", "Name Not Found")
        email = personal_info.get("email") or data.get("email", "N/A")
        phone = personal_info.get("phone") or data.get("phone", "N/A")

        st.subheader(name)
        c1, c2 = st.columns(2)
        c1.info(f"**Email:** {email}")
        c2.info(f"**Phone:** {phone}")

        if data.get("summary"):
            with st.expander("Professional Summary", expanded=True):
                st.markdown(data["summary"])

        skills_list = data.get("skills", [])
        if skills_list:
            st.subheader("Skills (Refined)")
            num_skills = len(skills_list)
            num_cols = min(num_skills, 4)
            if num_cols > 0:
                cols = st.columns(num_cols)
                sorted_skills = sorted(skills_list)
                for i, skill in enumerate(sorted_skills):
                    with cols[i % num_cols]:
                        st.markdown(f"- {skill}")

        experience = data.get("experience") or data.get("projects", [])
        if experience:
            st.subheader("Work Experience / Projects")
            for exp in experience:
                title = exp.get('title') or exp.get('name', 'N/A')
                company = exp.get('company')
                if company:
                    st.markdown(f"**{title}** at **{company}**")
                else:
                    st.markdown(f"**{title}**")
                st.markdown(f"*<small>{exp.get('dates', 'N/A')}</small>*", unsafe_allow_html=True)
                if exp.get('description'):
                    st.markdown(f"> {exp['description'].replace(chr(10), ' ')}")
                st.markdown("")

        education = data.get("education", [])
        if education:
            st.subheader("Education")
            for edu in education:
                st.markdown(f"**{edu.get('degree', 'N/A')}**, *{edu.get('institution') or edu.get('university', 'N/A')}*")
                st.markdown(f"<small>{edu.get('dates', 'N/A')}</small>", unsafe_allow_html=True)
                st.markdown("")
    else:
        st.info("Your parsed resume details will appear here after you upload and parse a file.")

with tab2:
    st.header("Top Job Matches")
    if st.session_state.job_results:
        for idx, job in enumerate(st.session_state.job_results):
            with st.container(border=True):
                score = job.get('match_score', 0)
                title = job.get('title', 'Untitled role')
                company = job.get('company_name')
                location = job.get('location')
                lvl = job.get('_inferred_experience', infer_experience_level(job))
                badge = (":green[" if score >= 75 else ":orange[" if score >= 50 else ":red[")
                icon = "✅" if score >= 75 else "🤷" if score >= 50 else "❌"

                st.markdown(f"##### {icon} **{badge}{score}% Match]** - {title}")
                st.write(f"**Company:** {company}")
                st.write(f"**Location:** {location}")
                st.write(f"**Experience Level (inferred):** {lvl}")
                st.caption(f"**AI Justification:** *{job.get('justification', '—')}*")

                apply_links = extract_apply_links(job)
                if apply_links:
                    c1, c2 = st.columns([1, 2])
                    first_label, first_url = apply_links[0]
                    if c1.button("🚀 Instant Apply (same tab)", key=f"instant_{idx}"):
                        # clean and hard-redirect
                        open_same_tab(clean_url(first_url))

                    with c2:
                        for label, url in apply_links:
                            st.link_button(f"Apply on {label}", clean_url(url))
                else:
                    st.warning("No direct apply link found.")
                    fallback = job.get("share_link") or job.get("link") or "#"
                    st.link_button("View Listing", clean_url(fallback))
    else:
        st.info("Your ranked job matches will appear here after you complete a search.")
