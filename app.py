import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from knowledge_base import KNOWLEDGE_BASE, LEARN_MORE_LINKS

#defining out of scope keywords
OUT_OF_SCOPE_KEYWORDS = ["consciousness", "conscious", "dream", "weather", "cook", "cooking", "recipe", "joke", "funny", "stock", "price", "market", "psychology", "philosophy", "politics", "religion", "spiritual", "god", "love", "emotion", "feeling", "relationship", "dating"]

#page configuration
st.set_page_config(page_title="NeuroSpike", layout="wide", initial_sidebar_state="expanded")

#CSS
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-light: #e2e8f0;
    }
    
    * {
        color: var(--text-light);
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--dark-bg);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-right: 2px solid var(--primary);
    }
    
    [data-testid="stChatInputContainer"] {
        background-color: var(--card-bg);
        border-top: 1px solid var(--primary);
    }
    
    .stTextInput > div > div > input {
        background-color: #0f172a;
        color: var(--text-light);
        border: 2px solid var(--primary);
        border-radius: 8px;
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% {
            border-color: var(--primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
        }
        50% {
            border-color: var(--secondary);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
        }
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid var(--secondary);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.8);
        animation: none;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid var(--secondary);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.3);
    }
    
    .stChatMessage {
        background-color: var(--card-bg);
        border-left: 3px solid var(--primary);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
            [data-testid="stChatInputContainer"] button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stChatInputContainer"] button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .stMarkdown h1 {
        color: #818cf8;
        text-shadow: 0 0 20px rgba(129, 140, 248, 0.3);
    }
    
    .stMarkdown h2 {
        color: #a78bfa;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 8px;
    }
    
    .stMarkdown h3 {
        color: #c084fc;
    }
    
    a {
        color: #60a5fa !important;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        color: #93c5fd !important;
        text-decoration: underline;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-light);
    }
</style>
""", unsafe_allow_html=True)

#sidebar
with st.sidebar:
    st.markdown("# 🧠 NeuroSpike")
    st.markdown("**Brain-Inspired AI for Everyone**")
    st.markdown("---")
    
    st.markdown("### 📚 Topics Covered")
    topics = [
        "🧠 Neurons & Brain Basics",
        "⚡ Electrical Signals & Dynamics",
        "🔄 LIF Model & Computation",
        "⏱️ Refractory Period & Timing",
        "📡 Spiking Neural Networks",
        "🖥️ Neuromorphic Hardware",
        "🚀 Real-World Applications",
        "💡 Startup & Career Insights"
    ]
    for topic in topics:
        st.write(topic)
    
    st.markdown("---")
    
    st.markdown("### 🎯 About This Project")
    st.markdown("""
    **NeuroSpike** is an AI chatbot that democratizes access to neuromorphic computing knowledge. Built from scratch with 100+ Q&A pairs spanning neuroscience, computational models, and real-world applications.
    
    **Why it matters:**
    - Makes complex neuroscience accessible to students
    - Bridges gap between brain biology and AI engineering  
    - Shows how startups can leverage neuromorphic tech
    - Demonstrates AI for social impact & education
    
    **Creator:** Arush Kalra (Class 11)  
    **Technology:** Python, Streamlit, scikit-learn  
    **Knowledge Base:** 100 Q&A pairs with difficulty levels
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎚️ Difficulty Levels")
    st.markdown("""
    🟢 **Beginner** — Foundational concepts      
    🟡 **Intermediate** — Applied understanding  
    🔴 **Advanced** — Research-level depth
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    st.caption("🔧 Fine-tune matching precision")
    threshold = st.slider("Confidence Threshold", 0.0, 0.5, 0.25, 0.01)
    st.caption("Higher = stricter | Lower = more lenient")
    
    st.markdown("---")
    st.markdown("💡 **Ask anything about neurons, SNNs, or neuromorphic computing!**")

#main content
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #818cf8; margin-bottom: 0;">🧠 NeuroSpike</h1>
    <p style="color: #a78bfa; font-size: 1.2rem; margin-top: 0;">Your AI Assistant for Neuromorphic Computing & Brain-Inspired AI</p>
</div>
""", unsafe_allow_html=True)

#built matching engine
all_patterns = []
answers = []
difficulties = []
learn_more_ids = []

for entry in KNOWLEDGE_BASE:
    for pattern in entry["patterns"]:
        all_patterns.append(pattern.lower())
        answers.append(entry["answer"])
        difficulties.append(entry.get("difficulty", "beginner"))
        learn_more_ids.append(entry.get("learn_more_id", None))

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
pattern_vectors = vectorizer.fit_transform(all_patterns)

FALLBACK = "I don't have information on that topic. I specialize in: neurons, LIF models, spiking neural networks, neuromorphic hardware, and their applications. Try asking about those instead!"

def get_response(user_input, threshold=0.25):
    """Find the closest matching answer using TF-IDF + fuzzy matching for typos"""
    user_lower = user_input.lower()
    
    #check for out of scope keywords
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in user_lower:
            return {"answer": FALLBACK, "index": -1}
    
    #try exact TF-IDF match 
    user_vector = vectorizer.transform([user_lower])
    scores = cosine_similarity(user_vector, pattern_vectors)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    
    #if TF-IDF score is good then using it
    if best_score >= threshold:
        return {"answer": answers[best_idx], "index": best_idx}
    
    #if TF-IDF failed then it tries fuzzy matching for typos
    best_fuzzy_idx = -1
    best_fuzzy_score = 0.7
    
    for i, pattern in enumerate(all_patterns):
        similarity = SequenceMatcher(None, user_lower, pattern).ratio()
        if similarity > best_fuzzy_score:
            best_fuzzy_score = similarity
            best_fuzzy_idx = i
    
    #if fuzzy match found then use it
    if best_fuzzy_idx != -1:
        return {"answer": answers[best_fuzzy_idx], "index": best_fuzzy_idx}
    
    #otherwise fallback
    return {"answer": FALLBACK, "index": -1}

#example questions - mix of foundational & impact focused
st.markdown("### 🚀 Quick Start Examples")
example_questions = [
    "What is a neuron?",
    "How can startups use neuromorphic AI?",
    "What are spiking neural networks?",
    "Why does neuromorphic computing matter?",
    "What careers exist in this field?",
    "How do neurons fire?"
]

cols = st.columns(3)
for idx, question in enumerate(example_questions):
    with cols[idx % 3]:
        if st.button(question, key=f"btn_{idx}", use_container_width=True):
            response_obj = get_response(question, threshold=threshold)
            response_text = response_obj["answer"]
            response_idx = response_obj["index"]
            
            #building display response with difficulty and links
            display_response = response_text
            
            if response_idx >= 0 and response_idx < len(difficulties):
                difficulty = difficulties[response_idx]
                learn_more_id = learn_more_ids[response_idx]
                
                #add difficulty badge
                if difficulty:
                    badge = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(difficulty, "")
                    display_response += f"\n\n{badge} {difficulty.capitalize()}"
                
                #add links
                if learn_more_id and learn_more_id in LEARN_MORE_LINKS:
                    links = LEARN_MORE_LINKS[learn_more_id]
                    link_text = ""
                    if links.get("wikipedia"):
                        link_text += f"[📚 Wikipedia]({links['wikipedia']})"
                    if links.get("arxiv"):
                        if link_text:
                            link_text += " | "
                        link_text += f"[📄 arXiv]({links['arxiv']})"
                    if link_text:
                        display_response += f"\n\n{link_text}"
            
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": display_response})
            st.rerun()

st.markdown("---")

#chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#user input
user_input = st.chat_input("Ask me anything about neuromorphic computing...")

if user_input:
    response_obj = get_response(user_input, threshold=threshold)
    response_text = response_obj["answer"]
    response_idx = response_obj["index"]
    
    #built display response with difficulty and links
    display_response = response_text
    
    if response_idx >= 0 and response_idx < len(difficulties):
        difficulty = difficulties[response_idx]
        learn_more_id = learn_more_ids[response_idx]
        
        #adding difficulty badge
        if difficulty:
            badge = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(difficulty, "")
            display_response += f"\n\n{badge} {difficulty.capitalize()}"
        
        #adding links
        if learn_more_id and learn_more_id in LEARN_MORE_LINKS:
            links = LEARN_MORE_LINKS[learn_more_id]
            link_text = ""
            if links.get("wikipedia"):
                link_text += f"[📚 Wikipedia]({links['wikipedia']})"
            if links.get("arxiv"):
                if link_text:
                    link_text += " | "
                link_text += f"[📄 arXiv]({links['arxiv']})"
            if link_text:
                display_response += f"\n\n{link_text}"
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": display_response})
    
    st.rerun()