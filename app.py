import streamlit as st
import importlib
from src.classifier import CascadePromptBouncer

# --- Page Config ---
st.set_page_config(page_title="Prompt Police", page_icon="🛡️", layout="wide")

# --- Initialize Model ---
@st.cache_resource
def load_model():
    return CascadePromptBouncer(save_dir='models')

try:
    classifier = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("Please ensure you have run `python src/train.py` to generate the models.")
    st.stop()

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
    
def reset_conversation():
    st.session_state.history = []
    classifier.reset_history()

# --- Sync classifier history with session state if needed ---
classifier.history = st.session_state.history.copy()

# --- UI Layout ---
st.title("🛡️ Prompt Police: AI Safety Guardrail")
st.markdown("A real-time binary classifier that catches adversarial prompts using a dual-stage cascade architecture.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Test a Prompt")
    
    prompt = st.text_area("Type a prompt here to test the classifier...", height=150, key="input_prompt")
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("Classify Prompt", type="primary", use_container_width=True):
            if prompt.strip():
                with st.spinner("Analyzing..."):
                    result = classifier.classify(prompt)
                    st.session_state.history = classifier.history.copy()
            else:
                st.warning("Please enter a prompt.")
    with cols[1]:
        if st.button("New Conversation", type="secondary", use_container_width=True, on_click=reset_conversation):
            pass

    st.markdown("---")
    
    if 'result' in locals() and result:
        label = result['label']
        conf = result['confidence']
        bg_color = "#ffebee" if label == "ADVERSARIAL" else "#e8f5e9"
        text_color = "#c62828" if label == "ADVERSARIAL" else "#2e7d32"
        border_color = text_color
        
        st.markdown(f"""
        <div style="border:2px solid {border_color};border-radius:8px;padding:14px;background:{bg_color};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:22px;font-weight:bold;color:{text_color};">{label}</span>
            <span style="font-size:11px;color:#666;background:#fff;padding:4px 8px;border-radius:10px;border:1px solid #ddd;">
              Stage {result['stage']} · Zone: {result['zone'].upper()}
            </span>
          </div>
          <div style="margin:8px 0;font-size:14px;">
            Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp;
            Signal Score: <b>{result['signal_score']:.1f}</b> <br>
            Hist Bonus: <b>{result['hist_bonus']:.1f}</b> &nbsp;|&nbsp;
            Cont Bonus: <b>{result['cont_bonus']:.1f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Threat Signals Breakdown")
        
        sigs = result['signals']
        threats = {
            'Persona / Injection': min(sigs.get('has_persona_override',0)*50 + sigs.get('has_injection_pattern',0)*30, 100),
            'Harmful intent': min(sigs.get('harm_keyword_score',0)*40, 100),
            'Encoding / B64': min(sigs.get('has_base64',0)*70 + sigs.get('encoding_score',0)*20, 100),
            'Roleplay framing': min(sigs.get('roleplay_score',0)*40, 100),
            'History escalation': min(int(result['hist_bonus']*25), 100),
            'Continuation attack': min(int(result['cont_bonus']*60), 100),
        }
        
        for k, v in threats.items():
            st.markdown(f"**{k}** ({v}%)")
            st.progress(v / 100.0)

with col2:
    st.subheader(f"Session History ({len(st.session_state.history)} turns)")
    st.markdown("Context context accumulates here and boosts continuation signals.")
    
    if not st.session_state.history:
        st.info("No history yet. Start typing to build conversation context.")
    else:
        for idx, (hist_prompt, hist_label) in enumerate(reversed(st.session_state.history)):
            color = "red" if hist_label == "ADVERSARIAL" else "green"
            st.markdown(f"**Turn {len(st.session_state.history)-idx}** :violet[({hist_label})]")
            st.caption(hist_prompt)
            st.markdown("---")
