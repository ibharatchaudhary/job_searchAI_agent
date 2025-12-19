import streamlit as st
import requests
import json
import re

def fetch_linkedin_jobs(access_token, keywords, location):
    """Fetches job postings from the LinkedIn API."""
    api_url = "https://api.linkedin.com/v2/jobSearch"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Restli-Protocol-Version': '2.0.0'
    }

    params = {
        'q': 'jobSearch',
        'jobSearch': {
            'keywords': keywords,
            'location': location
        },
        'count': 25 # Fetch up to 25 jobs
    }

    try:
        response = requests.get(api_url, headers=headers, params={'json': json.dumps(params)})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        linkedin_jobs = response.json().get('elements', [])
        
        # Format the jobs into the structure our app uses
        formatted_jobs = []
        for job in linkedin_jobs:
            job_details = job.get('jobPosting', {})
            if not job_details:
                continue

            # Extract job ID from the URN to create a link
            job_urn = job_details.get('entityUrn', '')
            job_id = job_urn.split(':')[-1] if job_urn else ''
            
            formatted_jobs.append({
                "title": job_details.get('title', 'N/A'),
                "company": job_details.get('companyName', 'N/A'),
                "location": job_details.get('formattedLocation', 'N/A'),
                "description": job_details.get('description', {}).get('text', 'No description available.'),
                "link": f"https://www.linkedin.com/jobs/view/{job_id}/" if job_id else "https://www.linkedin.com/jobs/"
            })
        return formatted_jobs

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        st.error("Please check your Access Token and network connection. A common issue is an expired or invalid token.")
        return []
    except json.JSONDecodeError:
        st.error("Failed to parse API response. The response may not be in valid JSON format.")
        return []


def calculate_relevance_score(job, preferences):
    """Calculates a relevance score for a job based on user preferences."""
    score = 0
    max_score = 100
    
    # 1. Keywords/Title Matching (40 points)
    title_keywords = [kw.strip().lower() for kw in preferences['keywords'].split(',') if kw.strip()]
    job_title_lower = job['title'].lower()
    job_desc_lower = job['description'].lower()
    
    keyword_matches = 0
    if title_keywords:
        for kw in title_keywords:
            if kw in job_title_lower or kw in job_desc_lower:
                keyword_matches += 1
        score += 40 * (keyword_matches / len(title_keywords))

    # 2. Skills Matching (40 points)
    user_skills = [skill.strip().lower() for skill in preferences['skills'].split(',') if skill.strip()]
    skill_matches = 0
    if user_skills:
        for skill in user_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', job_desc_lower, re.IGNORECASE):
                skill_matches += 1
        score += 40 * (skill_matches / len(user_skills))
        
    # 3. Location Matching (20 points)
    user_location = preferences['location'].lower().strip()
    job_location = job['location'].lower()

    if user_location == 'remote' and 'remote' in job_location:
        score += 20
    elif user_location and user_location in job_location:
        score += 20
    elif user_location and 'remote' in job_location: # User wants a city but job is remote
        score += 10 # Partial match
        
    # 4. Experience Level (Bonus/Penalty)
    user_exp = preferences.get('experience', 'any').lower()
    if user_exp != 'any':
        exp_patterns = {
            'entry-level': r'junior|entry|graduate|new grad',
            'mid-level': r'mid-level|mid|intermediate',
            'senior': r'senior|lead|principal|staff'
        }
        if re.search(exp_patterns[user_exp], job_title_lower) or re.search(exp_patterns[user_exp], job_desc_lower):
            score = min(max_score, score + 10) # Add bonus 10 points
        else:
            is_mismatch = False
            if user_exp == 'entry-level' and (re.search(exp_patterns['senior'], job_title_lower) or re.search(exp_patterns['mid-level'], job_title_lower)):
                is_mismatch = True
            if user_exp == 'senior' and re.search(exp_patterns['entry-level'], job_title_lower):
                is_mismatch = True
            if is_mismatch:
                score = max(0, score - 15) # Penalize more for clear mismatch

    return min(int(score), max_score)


# --- STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Smart Job Search AI")

# --- HEADER ---
st.title("🤖 Smart Job Search AI")
st.subheader("Find the best jobs for your skills automatically using the LinkedIn API")

# --- SIDEBAR FOR USER INPUT ---
with st.sidebar:
    st.header("Your Job Preferences")
    
    with st.expander("🔑 How to get your LinkedIn Access Token", expanded=False):
        st.markdown("""
        To use this app, you need a valid OAuth 2.0 access token from a LinkedIn Developer App. 
        The Client ID and Secret are used to *obtain* this token, not for direct API calls.
        
        **Steps:**
        1.  **Create an App:** Go to the [LinkedIn Developer Portal](https://www.linkedin.com/developers/apps/new) and create a new app.
        2.  **Add Products:** In your app's "Products" tab, add the `Sign In with LinkedIn using OpenID Connect` and `Share on LinkedIn` products. You might need to request access.
        3.  **Configure OAuth:** In the "Auth" tab, add an OAuth 2.0 redirect URL. For local testing, you can use `http://localhost:8501`.
        4.  **Generate Token:** Follow LinkedIn's [3-legged OAuth 2.0 flow](https://learn.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow?tabs=HTTPS1) to authorize your app and get an access token for your own account. This is a multi-step process involving a browser-based consent screen.
        
        This app uses the token directly for simplicity. Storing and refreshing tokens automatically is a more advanced setup.
        """)
    
    linkedin_token = st.text_input("LinkedIn OAuth2 Access Token", type="password", help="Your LinkedIn API token is required to fetch jobs.")

    keywords = st.text_input("Job Title / Keywords", "Software Engineer")
    skills = st.text_input("Skills (comma-separated)", "Python, React, SQL, AWS")
    location = st.text_input("Preferred Location (e.g., 'New York, NY' or 'Remote')", "Remote")
    experience_options = ['Any', 'Entry-Level', 'Mid-Level', 'Senior']
    experience = st.selectbox("Experience Level", options=experience_options, index=0)

    submit_button = st.button("🔍 Find Jobs")
    
# --- MAIN AREA FOR JOB RESULTS ---
if submit_button:
    if not linkedin_token:
        st.sidebar.error("Please enter your LinkedIn Access Token.")
    else:
        user_preferences = {
            "keywords": keywords,
            "skills": skills,
            "location": location,
            "experience": experience.lower().replace('-', '_')
        }

        with st.spinner("Fetching jobs from LinkedIn and ranking them..."):
            # 1. Fetch jobs from API
            live_jobs = fetch_linkedin_jobs(linkedin_token, keywords, location)

            # 2. Calculate scores for all jobs
            results = []
            if live_jobs:
                for job in live_jobs:
                    score = calculate_relevance_score(job, user_preferences)
                    if score >= 60:
                        job_result = job.copy()
                        job_result['score'] = score
                        results.append(job_result)

                # 3. Sort jobs by score in descending order
                sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

                if sorted_results:
                    st.header(f"🏆 Top {len(sorted_results)} Jobs Recommended for You")
                    
                    # Add a download button for the JSON results
                    json_output = json.dumps(sorted_results, indent=2)
                    st.download_button(
                        label="📥 Download Results as JSON",
                        data=json_output,
                        file_name="job_search_results.json",
                        mime="application/json"
                    )

                    for job in sorted_results:
                        with st.expander(f"**{job['title']}** at {job['company']}", expanded=True):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**Company:** {job['company']}")
                                st.markdown(f"**Location:** {job['location']}")
                                
                                tags_html = ""
                                if 'remote' in job['location'].lower():
                                    tags_html += "<span style='background-color:#2A9D8F;color:white;padding:3px 8px;border-radius:15px;margin-right:5px;font-size:12px;'>REMOTE</span>"
                                if job['score'] >= 85:
                                    tags_html += "<span style='background-color:#E76F51;color:white;padding:3px 8px;border-radius:15px;font-size:12px;'>HIGH MATCH</span>"
                                st.markdown(tags_html, unsafe_allow_html=True)

                                st.markdown(f"**Description:** {job['description'][:300]}...")
                                
                            with col2:
                                st.metric(label="Match Score", value=f"{job['score']}%")
                                st.progress(job['score'])
                                st.link_button("Apply on LinkedIn ➔", job['link'], use_container_width=True)
                                
                else:
                    st.warning("No relevant jobs found matching your criteria after filtering (score >= 60%). Try broadening your search!")
            else:
                st.warning("Could not fetch any jobs from LinkedIn. Please check your token and search criteria.")

else:
    st.info("Enter your preferences and LinkedIn token in the sidebar and click 'Find Jobs' to start.")

