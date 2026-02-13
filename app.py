import os
import re
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------
# Config / filenames (must match your folder)
# ----------------------------
IBOM_FILE = "ibom.html"
SYSTEM_PROMPT_FILE = "troubleshooting_prompt.txt"
CHECKLIST_FILE = "jove_troubleshooting_checklist.json"
EXPECTED_FILE = "expected_measurements_jove_mvp.csv"
BOM_FILE = "bom.csv"

DEFAULT_MODEL = "gpt-4o"  # more capable than 4o-mini

REFDES_RE = re.compile(
    r"\b(VR\d+|U\d+|R\d+|C\d+|P\d+|J\d+|D\d+|Q\d+|TP\d+)\b",
    re.IGNORECASE
)

IC_REWRITE = re.compile(r"\bIC(\d+)\b", re.IGNORECASE)

# ----------------------------
# Helpers: file loading
# ----------------------------
def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore").strip()

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))

def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=";")  # EU-style fallback

def normalize_refdes_text(text: str) -> str:
    """
    Normalize common human naming to PCB refdes.
    Example: IC3 -> U3 (common mismatch between speech and KiCad/iBOM refdes).
    """
    if not text:
        return text
    return IC_REWRITE.sub(lambda m: f"U{m.group(1)}", text)

def extract_refdes(text: str):
    found = set(m.group(1).upper() for m in REFDES_RE.finditer(text or ""))
    def sort_key(x):
        m = re.match(r"([A-Z]+)(\d+)", x)
        if not m:
            return (x, 0)
        return (m.group(1), int(m.group(2)))
    return sorted(found, key=sort_key)

def df_find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if cand in lower[c]:
                return c
    return None

def expected_rows_for_refdes(df_expected: pd.DataFrame, refdes_list):
    if df_expected is None or df_expected.empty or not refdes_list:
        return pd.DataFrame()

    likely_cols = []
    for c in df_expected.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["where", "probe", "point", "node", "ref", "location", "check", "test", "step", "net"]):
            likely_cols.append(c)

    text_cols = [c for c in df_expected.columns if df_expected[c].dtype == "object"]
    search_cols = list(dict.fromkeys(likely_cols + text_cols))
    if not search_cols:
        return pd.DataFrame()

    mask = False
    for ref in refdes_list:
        r = re.escape(ref)
        ref_mask = False
        for c in search_cols:
            ref_mask = ref_mask | df_expected[c].astype(str).str.contains(r, case=False, na=False)
        mask = mask | ref_mask

    return df_expected[mask].copy()

def bom_rows_for_refdes(df_bom: pd.DataFrame, refdes_list):
    if df_bom is None or df_bom.empty or not refdes_list:
        return pd.DataFrame()

    ref_col = df_find_column(df_bom, ["ref", "reference", "designator"])
    if ref_col is None:
        text_cols = [c for c in df_bom.columns if df_bom[c].dtype == "object"]
        if not text_cols:
            return pd.DataFrame()
        mask = False
        for ref in refdes_list:
            r = re.escape(ref)
            ref_mask = False
            for c in text_cols:
                ref_mask = ref_mask | df_bom[c].astype(str).str.contains(r, case=False, na=False)
            mask = mask | ref_mask
        return df_bom[mask].copy()

    mask = False
    for ref in refdes_list:
        r = re.escape(ref)
        mask = mask | df_bom[ref_col].astype(str).str.contains(r, case=False, na=False)
    return df_bom[mask].copy()

def df_to_compact_text(df: pd.DataFrame, max_rows=8, max_cols=10):
    """
    Avoid pandas.to_markdown() (needs tabulate).
    Produces readable plain text for prompt context.
    """
    if df is None or df.empty:
        return ""
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    if d.shape[1] > max_cols:
        d = d.iloc[:, :max_cols]
    return d.to_string(index=False)

def safe_one_step_enforcer(text: str) -> str:
    return text.strip() if text else text

# ----------------------------
# iBOM auto-highlight injection
# ----------------------------
def ibom_with_ref_highlight(ibom_html: str, ref: str | None):
    if not ref:
        return ibom_html

    ref_safe = ref.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    js = f"""
    <script>
    (function() {{
        const ref = "{ref_safe}";

        function forceFB() {{
            const fb = document.getElementById("fb-btn");
            if (!fb) return false;
            fb.click();
            return true;
        }}

        function setRefLookup() {{
            const el = document.getElementById("reflookup");
            if (!el) return false;
            el.value = ref;
            el.dispatchEvent(new Event("input", {{ bubbles: true }}));
            el.dispatchEvent(new KeyboardEvent("keyup", {{ bubbles: true, key: "Enter" }}));
            return true;
        }}

        function tickHighlightedCheckbox() {{
            const row = document.querySelector("tr.highlighted");
            if (!row) return false;

            const cb = row.querySelector('input[type="checkbox"]');
            if (!cb) return false;

            if (!cb.checked) {{
                cb.checked = true;
                cb.dispatchEvent(new Event("change", {{ bubbles: true }}));
                cb.dispatchEvent(new Event("input", {{ bubbles: true }}));
            }}
            return true;
        }}

        let attempts = 0;
        const timer = setInterval(() => {{
            attempts += 1;
            const ok0 = forceFB();
            const ok1 = setRefLookup();
            const ok2 = tickHighlightedCheckbox();
            if ((ok1 && ok2) || attempts > 60) {{
                clearInterval(timer);
            }}
        }}, 120);
    }})();
    </script>
    """
    return ibom_html + js

# ----------------------------
# OpenAI key + call
# ----------------------------
def get_api_key() -> str:
    """
    Streamlit Cloud: use st.secrets
    Local/dev: use OPENAI_API_KEY env var
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("OPENAI_API_KEY", "")
    return str(key).strip()

def call_openai(messages, model=DEFAULT_MODEL):
    from openai import OpenAI
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it in Streamlit Secrets or set env var OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    layout="wide",
    page_title="JOVE Troubleshooting + iBOM",
    initial_sidebar_state="collapsed"
)

st.title("JOVE Troubleshooting with visual guidance")

with st.sidebar:
    st.header("Settings")
    use_openai = st.toggle("Use OpenAI API", value=True)
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("If you get quota errors, disable API to test UI.")

    st.divider()
    st.subheader("API key status")
    has_key = bool(get_api_key())
    st.write("✅ Key detected" if has_key else "❌ No key detected")
    if not has_key:
        st.caption("Add OPENAI_API_KEY in Streamlit Secrets (recommended) or set env var OPENAI_API_KEY.")

    st.divider()
    st.subheader("Files found")
    for f in [SYSTEM_PROMPT_FILE, CHECKLIST_FILE, EXPECTED_FILE, BOM_FILE, IBOM_FILE]:
        st.write(("✅" if Path(f).exists() else "❌"), f)

# Load files
system_prompt = load_text(SYSTEM_PROMPT_FILE) or (
    "You are a troubleshooting assistant for the JOVE Eurorack module. "
    "Be short. One step per turn. Prefer measurements. "
    "Always instruct the user to locate parts using the iBOM search on the right. "
    "Use PCB refdes exactly (U#, R#, C#, VR#). Do not use 'IC#'."
)

checklist = load_json(CHECKLIST_FILE)  # optional usage
df_expected = load_csv(EXPECTED_FILE)
df_bom = load_csv(BOM_FILE)

# Load iBOM HTML
if not Path(IBOM_FILE).exists():
    st.error(f"Missing {IBOM_FILE}. Put it in the same folder as app.py.")
    st.stop()
ibom_html = load_text(IBOM_FILE)

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
if "last_refdes" not in st.session_state:
    st.session_state.last_refdes = []
if "ibom_ref" not in st.session_state:
    st.session_state.ibom_ref = ""  # what to auto-highlight in iBOM

# Layout
col_chat, col_ibom = st.columns([1, 2], gap="medium")

# ----------------------------
# Chat column
# ----------------------------
with col_chat:
    st.subheader("Troubleshooting chat")

    # Display chat history (skip system)
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe the symptom (dead / no output / no resonance / distorted / weak)...")

    if user_input:
        # Normalize user input (IC3 -> U3 etc.)
        user_input_norm = normalize_refdes_text(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input_norm})

        # Refdes seen so far (user + last bot)
        refdes = set(extract_refdes(user_input_norm))
        refdes |= set(st.session_state.last_refdes)

        checklist_hint = ""
        if checklist is not None:
            checklist_hint = (
                "Project has a symptom-driven checklist in jove_troubleshooting_checklist.json. "
                "Follow it. Do not invent steps."
            )

        # Pull relevant expected/BOM rows for refdes mentioned
        exp_rows = expected_rows_for_refdes(df_expected, sorted(refdes))
        bom_rows = bom_rows_for_refdes(df_bom, sorted(refdes))

        exp_txt = df_to_compact_text(exp_rows, max_rows=8, max_cols=10)
        bom_txt = df_to_compact_text(bom_rows, max_rows=8, max_cols=10)

        context_blocks = []
        if checklist_hint:
            context_blocks.append(f"CHECKLIST: {checklist_hint}")
        if exp_txt:
            context_blocks.append("EXPECTED_MEASUREMENTS (relevant rows):\n" + exp_txt)
        if bom_txt:
            context_blocks.append("BOM (relevant rows):\n" + bom_txt)

        if context_blocks:
            st.session_state.messages.append({
                "role": "system",
                "content": "CONTEXT (authoritative; do not override):\n" + "\n\n".join(context_blocks)
            })

        if use_openai:
            try:
                bot_reply = call_openai(st.session_state.messages, model=model)
            except Exception as e:
                bot_reply = (
                    f"API error: {e}\n\n"
                    "Fix: add OPENAI_API_KEY in Streamlit Secrets, or set env var OPENAI_API_KEY, "
                    "or disable 'Use OpenAI API' in the sidebar to test UI."
                )
        else:
            guess = next(iter(refdes), "P1")
            bot_reply = (
                f"1. Find {guess} in the iBOM on the right (it will highlight).\n"
                "2. Tell me when you can see it clearly."
            )

        # Normalize model output too (IC3 -> U3)
        bot_reply = normalize_refdes_text(bot_reply)

        bot_reply = safe_one_step_enforcer(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        # Extract refdes from bot reply and auto-highlight the first one
        st.session_state.last_refdes = extract_refdes(bot_reply)
        if st.session_state.last_refdes:
            st.session_state.ibom_ref = st.session_state.last_refdes[0]

        st.rerun()

    st.divider()
    st.subheader("Component location helper")

    if st.session_state.last_refdes:
        st.write("The assistant mentioned:", ", ".join(st.session_state.last_refdes))

        chosen = st.selectbox("Highlight in iBOM:", st.session_state.last_refdes, index=0)
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Highlight now"):
                st.session_state.ibom_ref = chosen
                st.rerun()
        with cols[1]:
            if st.button("Clear highlight"):
                st.session_state.ibom_ref = ""
                st.rerun()
    else:
        st.caption("When the assistant mentions parts (U3/R31/VR2...), you can auto-highlight them here.")

    st.divider()
    st.caption(
        "This app was created as a Proof of Concept by Rasmus Nyåker as an AI project for a class at "
        "Business Academy Copenhagen (EK). The service uses OpenAI for the communication layer. "
        "By using the service you agree that your data is shared with OpenAI."
    )

# ----------------------------
# iBOM column
# ----------------------------
with col_ibom:
    st.subheader("iBOM Viewer")
    html = ibom_with_ref_highlight(ibom_html, st.session_state.ibom_ref)
    components.html(html, height=900, scrolling=True)
