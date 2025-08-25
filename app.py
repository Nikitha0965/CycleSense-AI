import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(page_title="Menstrual Health App", page_icon="🌸")

st.title("🌸 AI-powered Menstrual Health App (MVP)")
st.caption("Track periods, get predictions, and view remedies.")

API = st.secrets.get("API_URL", "http://localhost:8000")

with st.expander("Log a new entry"):
    d = st.date_input("Date", value=date.today())
    flow = st.selectbox("Flow", options=[None,0,1,2], format_func=lambda x: {None:'',0:'Light',1:'Medium',2:'Heavy'}[x])
    pain = st.slider("Pain (0..10)", 0, 10, 3)
    mood = st.text_input("Mood", "")
    notes = st.text_area("Notes", "")
    if st.button("Save log"):
        payload = {"date": d.isoformat(), "flow": flow, "pain": pain, "mood": mood, "notes": notes}
        r = requests.post(f"{API}/logs", json=payload)
        if r.ok:
            st.success("Saved")
        else:
            st.error(r.text)

st.subheader("🗓️ Predict Next Period")
dates_csv = st.text_input("Enter past period start dates (YYYY-MM-DD, comma-separated)", "2025-06-01, 2025-06-30, 2025-07-28")
override = st.number_input("Override cycle length (optional)", min_value=0, max_value=90, value=0, step=1)
if st.button("Predict"):
    try:
        period_dates = [x.strip() for x in dates_csv.split(",") if x.strip()]
        payload = {"period_dates": period_dates}
        if override > 0:
            payload["cycle_length_override"] = int(override)
        r = requests.post(f"{API}/predict", json=payload)
        if r.ok:
            data = r.json()
            st.success(f"Next period: {data['next_period_date']} (method: {data['method']})")
            st.info(f"Predicted cycle length: {data['predicted_cycle_length']} days")
            st.write(f"Ovulation window: {data['ovulation_window_start']} — {data['ovulation_window_end']}")
        else:
            st.error(r.text)
    except Exception as e:
        st.error(str(e))

st.subheader("🌿 Remedies")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Add Remedy**")
    title = st.text_input("Title")
    symptom = st.text_input("Symptom", placeholder="cramps, nausea, fatigue, mood")
    desc = st.text_area("Description")
    tags = st.text_input("Tags (comma-separated)", "home,herbal")
    if st.button("Add Remedy"):
        payload = {"title": title, "for_symptom": symptom, "description": desc, "tags": [t.strip() for t in tags.split(",") if t.strip()]}
        r = requests.post(f"{API}/remedies", json=payload)
        if r.ok:
            st.success("Remedy added")
        else:
            st.error(r.text)
with col2:
    st.markdown("**Browse Remedies**")
    q = st.text_input("Filter by symptom (optional)")
    url = f"{API}/remedies"
    if q:
        url += f"?symptom={q}"
    r = requests.get(url)
    if r.ok:
        df = pd.DataFrame(r.json())
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No remedies yet.")
    else:
        st.error(r.text)

st.caption("Backend API docs: /docs  •  This MVP uses a baseline ML and is not medical advice.")
