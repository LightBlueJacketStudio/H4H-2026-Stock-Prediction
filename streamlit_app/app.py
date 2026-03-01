import streamlit as st
import dashboard  # dashboard.py in same folder

st.set_page_config(page_title="Glimpse", layout="centered")

# --------- router state ----------
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "ticker" not in st.session_state:
    st.session_state["ticker"] = ""

def go_dashboard():
    st.session_state["page"] = "dashboard"
    st.rerun()

def go_home():
    st.session_state["page"] = "home"
    st.rerun()

def render_home():
    st.title("Glimpse")
    st.subheader("Powered by AMD Developer Cloud")
    st.divider()

    typed = st.text_input(
        "Enter company ticker",
        placeholder="e.g., AMZN, AAPL, NVDA",
        value=st.session_state["ticker"]
    )
    if typed:
        st.session_state["ticker"] = typed.upper().strip()

    st.markdown("### Supported Companies")
    supported = ["AMZN", "AAPL", "NVDA", "TSLA"]
    cols = st.columns(len(supported))
    for col, company in zip(cols, supported):
        with col:
            if st.button(company, key=f"btn_{company}"):
                st.session_state["ticker"] = company

    st.divider()

    if st.button("Analyze with Glimpse", use_container_width=True):
        ticker = st.session_state["ticker"]
        if not ticker:
            st.warning("Please enter a ticker or choose a supported company.")
        else:
            go_dashboard()

# --------- render ----------
if st.session_state["page"] == "home":
    render_home()
else:
    dashboard.render_dashboard(go_home)