import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SpamShield | Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(64,91,190,.22), transparent 35%),
        radial-gradient(ellipse at 95% 20%, rgba(24,169,153,.10), transparent 30%),
        #0b1020;
    color: #edf2ff;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.brand {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 1.1rem;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: linear-gradient(135deg, #6d8cff, #4e65d8);
    box-shadow: 0 8px 30px rgba(82,108,230,.3);
    font-size: 24px;
}

.brand-name {
    color: #f3f6ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}

.hero-title {
    text-align: center;
    color: #f4f6ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.1rem);
    font-weight: 700;
    letter-spacing: -1.8px;
    line-height: 1.12;
    margin: .4rem 0 .85rem;
}

.hero-title span {
    background: linear-gradient(90deg, #8da4ff, #63d8c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #a9b5ce;
    font-size: 1.03rem;
    text-align: center;
    line-height: 1.7;
    max-width: 580px;
    margin: 0 auto 2rem;
}

.trust-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 9px;
    margin-bottom: 1.7rem;
}

.trust-chip {
    color: #c8d3ea;
    background: rgba(255,255,255,.04);
    border: 1px solid #293650;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: .78rem;
}

.panel {
    background: rgba(18,26,45,.92);
    border: 1px solid #26334d;
    border-radius: 22px;
    padding: 1.5rem;
    box-shadow: 0 22px 65px rgba(0,0,0,.22);
    margin-bottom: 1rem;
}

.panel-heading {
    color: #f1f4ff;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: .35rem;
}

.panel-description {
    color: #9ba9c4;
    font-size: .9rem;
    line-height: 1.6;
}

div[data-testid="stTextArea"] label {
    color: #d7e0f5 !important;
    font-weight: 600 !important;
}

div[data-testid="stTextArea"] textarea {
    background: #0c1426 !important;
    color: #edf2ff !important;
    border: 1px solid #30405e !important;
    border-radius: 14px !important;
    min-height: 190px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 15px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #718cff !important;
    box-shadow: 0 0 0 2px rgba(113,140,255,.15) !important;
}

div.stButton > button {
    border-radius: 12px;
    min-height: 3rem;
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #718cff, #5e72e8);
    color: white;
    border: 1px solid transparent;
    box-shadow: 0 7px 20px rgba(88,110,230,.2);
}

div.stButton > button[kind="primary"]:hover {
    border-color: #a8b6ff;
    box-shadow: 0 9px 26px rgba(88,110,230,.32);
    transform: translateY(-1px);
}

div.stButton > button[kind="secondary"] {
    background: #19243a;
    color: #dce5f8;
    border: 1px solid #33415d;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #718cff;
    color: white;
}

.section-label {
    color: #9eacc7;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 1.45rem 0 .65rem;
}

.result {
    border-radius: 16px;
    padding: 1.2rem;
    margin-top: .5rem;
    border: 1px solid;
}

.result-spam {
    background: rgba(239,68,68,.09);
    border-color: rgba(248,113,113,.38);
}

.result-safe {
    background: rgba(34,197,145,.09);
    border-color: rgba(52,211,153,.35);
}

.result-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: .35rem;
}

.result-copy {
    color: #b6c2d8;
    font-size: .9rem;
    line-height: 1.65;
}

.footer {
    color: #7786a4;
    text-align: center;
    font-size: .8rem;
    margin-top: 2.5rem;
    line-height: 1.8;
}

@media (max-width: 600px) {
    .block-container {
        padding: 1.4rem 1rem 2rem;
    }

    .panel {
        padding: 1rem;
        border-radius: 17px;
    }

    .hero-subtitle {
        font-size: .94rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- NLTK RESOURCES ----------------
@st.cache_resource
def load_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords")
    ]

    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

    return set(stopwords.words("english"))


try:
    ENGLISH_STOPWORDS = load_nltk_resources()
except Exception as e:
    st.error(f"Could not load NLTK resources: {e}")
    st.stop()


ps = PorterStemmer()


# ---------------- TEXT PREPROCESSING ----------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords and punctuation
    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOPWORDS
        and word not in string.punctuation
    ]

    # Apply stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    APP_DIR = Path(__file__).resolve().parent

    with open(APP_DIR / "vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open(APP_DIR / "model.pkl", "rb") as file:
        classifier = pickle.load(file)

    return vectorizer, classifier


try:
    tfidf, model = load_model_files()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. "
        "Keep model.pkl and vectorizer.pkl beside app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# ---------------- SESSION STATE ----------------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ---------------- CALLBACK FUNCTIONS ----------------
def set_example(message):
    st.session_state.message_input = message
    st.session_state.prediction = None


def clear_message():
    st.session_state.message_input = ""
    st.session_state.prediction = None


# ---------------- HEADER ----------------
st.markdown("""
<div class="brand">
    <div class="brand-icon">🛡️</div>
    <div class="brand-name">SpamShield</div>
</div>

<div class="hero-title">
    Your inbox,<br><span>protected.</span>
</div>

<div class="hero-subtitle">
    Check any email or SMS for suspicious content
    using machine-learning-powered spam detection.
</div>

<div class="trust-row">
    <div class="trust-chip">✦ ML-powered detection</div>
    <div class="trust-chip">⚡ Instant prediction</div>
    <div class="trust-chip">🔒 Local app processing</div>
</div>
""", unsafe_allow_html=True)


# ---------------- CLASSIFIER PANEL ----------------
st.markdown("""
<div class="panel">
    <div class="panel-heading">Analyze a message</div>
    <div class="panel-description">
        Paste your email or SMS below. SpamShield will classify
        the message as spam or not spam.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- MESSAGE INPUT ----------------
message = st.text_area(
    "Message content",
    key="message_input",
    height=190,
    placeholder=(
        "Example: Congratulations! You've won a free prize. "
        "Click here to claim it..."
    ),
    help="Paste the message you want to check."
)

st.caption(f"✍️ {len(message)} characters")


# ---------------- BUTTONS ----------------
predict_col, clear_col = st.columns([3, 1], gap="small")

with predict_col:
    predict_clicked = st.button(
        "🛡️ Analyze message",
        type="primary",
        use_container_width=True
    )

with clear_col:
    st.button(
        "Clear",
        type="secondary",
        use_container_width=True,
        on_click=clear_message
    )


# ---------------- PREDICTION ----------------
if predict_clicked:

    if not message.strip():
        st.warning("Please enter a message before running the analysis.")
        st.session_state.prediction = None

    else:
        try:
            transformed_sms = transform_text(message)

            vector_input = tfidf.transform([transformed_sms])

            result = model.predict(vector_input)[0]

            st.session_state.prediction = int(result)

        except LookupError:
            st.error(
                "An NLTK resource is missing. Check your internet "
                "connection and restart the app."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------- DISPLAY RESULT ----------------
if st.session_state.prediction is not None:

    st.markdown(
        '<div class="section-label">Analysis result</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction == 1:

        st.markdown("""
        <div class="result result-spam">
            <div class="result-title">
                🚨 Spam detected
            </div>
            <div class="result-copy">
                This message has been classified as spam by the model.
                Be cautious with suspicious links, requests for personal
                information, and unexpected offers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result result-safe">
            <div class="result-title">
                ✅ Not classified as spam
            </div>
            <div class="result-copy">
                The model classified this message as not spam.
                This does not guarantee that the message is safe.
                Always check unexpected links and requests carefully.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- EXAMPLE MESSAGES ----------------
st.markdown(
    '<div class="section-label">Try an example</div>',
    unsafe_allow_html=True
)

st.caption(
    "Choose an example to fill the message box, then analyze it."
)

example_col1, example_col2 = st.columns(2, gap="medium")

with example_col1:
    st.markdown("**🎁 Promotional message**")
    st.caption("An example of a suspicious prize offer.")

    st.button(
        "Use spam example",
        use_container_width=True,
        on_click=set_example,
        args=(
            "Congratulations! You have won a free prize. "
            "Click here now to claim your reward.",
        )
    )

with example_col2:
    st.markdown("**👋 Everyday message**")
    st.caption("An example of an ordinary personal message.")

    st.button(
        "Use normal example",
        use_container_width=True,
        on_click=set_example,
        args=(
            "Hey, are we still meeting for lunch today? "
            "Let me know when you are free.",
        )
    )


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong>SpamShield</strong> · Machine-learning message classification<br>
    Predictions are model-based estimates and may not always be correct.
</div>
""", unsafe_allow_html=True)
