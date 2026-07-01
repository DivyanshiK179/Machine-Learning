import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="KAALVYUH | NASA NEO Predictor",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("Models/best_model.pkl")
    scaler = joblib.load("Models/scaler.pkl")
    feature_names = joblib.load("Models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ==========================================
# NASA CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ------------------------------
GENERAL
------------------------------ */

html,
body {
    margin:0;
    font-family:'Rajdhani',sans-serif;
    font-size:15px;
    color:white;

    background:linear-gradient(
        -45deg,
        #000000,
        #0f2027,
        #203a43,
        #2c5364
    );

    background-size:400% 400%;
    animation:gradient 12s ease infinite;
}

@keyframes gradient{
    0%{
        background-position:0% 50%;
    }
    50%{
        background-position:100% 50%;
    }
    100%{
        background-position:0% 50%;
    }
}

/* ------------------------------
BACKGROUND
------------------------------ */

.stApp{
background:
radial-gradient(circle at top,#152642 0%,#071320 45%,#010409 100%);
background-attachment:fixed;
}

/* ------------------------------
Hide Streamlit Branding
------------------------------ */

#MainMenu{
visibility:hidden;
}

/* Keep the header visible so the sidebar toggle remains available */
header{
background:transparent !important;
}

/* Hide only the footer */
footer{
visibility:hidden;
}

/* ------------------------------
Scrollbar
------------------------------ */

::-webkit-scrollbar{
width:8px;
}

::-webkit-scrollbar-thumb{
background:#1d4ed8;
border-radius:20px;
}

/* ------------------------------
SIDEBAR
------------------------------ */

section[data-testid="stSidebar"]{
background:#081321;
border-right:1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] *{
color:white;
}

/* ------------------------------
HEADINGS
------------------------------ */

h1,h2,h3,h4,h5,h6{
    font-family:'Orbitron',sans-serif;
    color:#55e6ff;
    font-weight:700;
    letter-spacing:2px;
    text-shadow:
        0 0 10px rgba(0,255,255,.6),
        0 0 20px rgba(0,255,255,.3);
}
            
/* ------------------------------
HERO
------------------------------ */

.hero{
    background:linear-gradient(
        135deg,
        rgba(0,150,255,.18),
        rgba(20,30,70,.85)
    );
    border-radius:25px;
    padding:55px;
    border:1px solid rgba(0,255,255,.25);
    box-shadow:
        0 0 35px rgba(0,180,255,.25);
}

.hero h1{
    font-size:70px;
    color:#63dfff;
    font-family:'Orbitron',sans-serif;
    text-transform:uppercase;
    letter-spacing:4px;
    text-shadow:
        0 0 15px #00e5ff,
        0 0 30px #00c8ff,
        0 0 60px #0077ff;
}

.hero p{
    font-family:'Rajdhani',sans-serif;
    font-size:28px;
    color:#cde8ff;

}

/* ------------------------------
METRIC CARDS
------------------------------ */

.card{
    background:rgba(18,25,44,.75);
    backdrop-filter:blur(10px);
    border-radius:18px;
    padding:25px;
    border:1px solid rgba(0,255,255,.18);
    box-shadow:
        0 0 18px rgba(0,255,255,.15);
    transition:.35s;

}

.card:hover{
    transform:translateY(-6px);
    box-shadow:
        0 0 30px rgba(0,255,255,.35);

}

/* ------------------------------
BUTTON
------------------------------ */

.stButton>button{
    background:linear-gradient(
        90deg,
        #00b7ff,
        #0059ff
    );
    color:white;
    font-family:'Orbitron',sans-serif;
    font-size:18px;
    font-weight:700;
    border-radius:12px;
    border:none;
    transition:.3s;

}

.stButton>button:hover{
    transform:scale(1.05);
    box-shadow:0 0 25px cyan;

}

/* ------------------------------
TEXT INPUT
------------------------------ */

.stNumberInput input{
    background:#12192b !important;
    color:white !important;
    border:1px solid rgba(0,255,255,.25)!important;
    border-radius:10px!important;
}

div[data-baseweb="select"] > div{
    background:#12192b !important;
    color:white !important;
    border:1px solid rgba(0,255,255,.25)!important;
    border-radius:10px!important;
}

/* ------------------------------
NUMBER INPUT
------------------------------ */

div[data-baseweb="input"]{
background:#081321;
border-radius:12px;
}

/* ------------------------------
SELECT BOX
------------------------------ */

/* ------------------------------
SELECT BOX
------------------------------ */

div[data-testid="stSelectbox"]{
    margin-top:0 !important;
}

div[data-baseweb="select"]{
    min-height:42px !important;
}

div[data-baseweb="select"] > div{
    height:42px !important;
    background:#12192b !important;
    border:1px solid rgba(0,255,255,.25)!important;
    border-radius:10px !important;
    display:flex;
    align-items:center;
}

div[data-baseweb="select"] span{
    color:white !important;
    font-size:16px;
}

/* ------------------------------
DATAFRAME
------------------------------ */

[data-testid="stDataFrame"]{
border-radius:18px;
overflow:hidden;
}

/* ------------------------------
TABLE
------------------------------ */

table{
background:rgba(255,255,255,.04);
}

/* ------------------------------
PROGRESS BAR
------------------------------ */

.stProgress>div>div{
background:
linear-gradient(
90deg,
#06b6d4,
#2563eb
);

}

/* ------------------------------
SUCCESS
------------------------------ */

div[data-testid="stAlert"]{
border-radius:16px;
}

/* ------------------------------
PLOTLY CHART
------------------------------ */

.js-plotly-plot{
background:transparent !important;
}

/* ------------------------------
HORIZONTAL LINE
------------------------------ */

hr{
border:1px solid rgba(255,255,255,.08);
}

/* ------------------------------
DOWNLOAD BUTTON
------------------------------ */

.stDownloadButton>button{
width:100%;
height:55px;
font-size:18px;
border-radius:15px;
background:
linear-gradient(
90deg,
#0891b2,
#2563eb
);

color:white;
font-weight:bold;
border:none;
}

.stDownloadButton>button:hover{
box-shadow:
0 0 25px #38bdf8;
}

/* ------------------------------
FOOTER
------------------------------ */

.footer{
text-align:center;
padding:40px;
color:#94a3b8;
font-size:16px;
}

/* ------------------------------
SMALL ANIMATION
------------------------------ */

@keyframes glow{
0%{
box-shadow:
0 0 8px #2563eb;
}

50%{
box-shadow:
0 0 22px #38bdf8;
}

100%{
box-shadow:
0 0 8px #2563eb;
}

}

.hero{
animation:glow 5s infinite;
}
            
.block-container{
    max-width:96% !important;
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

html{
    zoom:110%;
}
            
/* Sidebar Toggle Button */

button[kind="header"]{
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    color:white !important;
    z-index:999999 !important;
}

[data-testid="collapsedControl"]{
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    
    st.markdown("""
<div style="text-align:center;padding:20px 10px;">

<h1 style="color:#55e6ff;
font-size:42px;
font-family:'Orbitron',sans-serif;
text-shadow:0 0 10px cyan,0 0 20px #00bfff,0 0 35px #008cff;
margin-bottom:5px;">

🌍 KAALVYUH

</h1>

<p style="color:#9bdcff;
font-size:16px;
letter-spacing:2px;
margin-top:0;">

Orbital Intelligence

</p>

</div>
""", unsafe_allow_html=True)

    st.markdown("# 🚀 Mission Control")
    st.markdown("---")

    # Mission Status
    st.success("🟢 Mission Status : ACTIVE")
    st.info("📡 Connection : Online")
    st.markdown("---")

    # Model Information
    st.subheader("🤖 AI Model")
    st.write("**Model**")
    st.caption("Random Forest Classifier")
    st.write("**Accuracy**")
    st.caption("91.6%")
    st.write("**Target**")
    st.caption("Hazardous / Non-Hazardous")
    st.markdown("---")

    # Dataset Information
    st.subheader("🌍 Dataset")
    st.write("NASA Near Earth Objects")
    st.metric(
        "Records",
        "90,836"
    )

    st.metric(
        "Features",
        "6"
    )

    st.metric(
        "Classes",
        "2"
    )

    st.markdown("---")

    # Feature List
    st.subheader("📋 Features")
    st.caption("• Estimated Diameter Min")
    st.caption("• Estimated Diameter Max")
    st.caption("• Relative Velocity")
    st.caption("• Miss Distance")
    st.caption("• Sentry Object")
    st.caption("• Absolute Magnitude")
    st.markdown("---")

    # Technology Stack
    st.subheader("💻 Tech Stack")
    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🤖 Scikit-Learn")
    st.write("📈 Plotly")
    st.write("🌐 Streamlit")

    st.markdown("---")

    # System Health
    st.subheader("⚙ System Health")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Model",
            "Loaded"
        )
    with col2:
        st.metric(
            "Scaler",
            "Loaded"
        )

    st.success("✔ Prediction Engine Ready")
    st.success("✔ Feature Pipeline Ready")
    st.success("✔ Dashboard Ready")
    st.markdown("---")

    # Developer Card
with st.sidebar:

    st.subheader("👨‍💻 Developer")

    st.info("""
**NASA NEO Hazard Predictor**
Machine Learning Internship Project
### Developed using
• Streamlit
• Random Forest
• Plotly
• Python
• Scikit-Learn
""")

    st.markdown("---")

    st.caption("© 2026 KAALVYUH | NASA NEO Predictor")

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌍 KAALVYUH</h1>
        <h3>NASA Near Earth Object Hazard Prediction System</h3>
        <p>
        An Artificial Intelligence powered dashboard that predicts whether a
        Near-Earth Object (NEO) is Hazardous or Non-Hazardous using a trained
        Random Forest Machine Learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# PROJECT DESCRIPTION
# ==========================================================

st.markdown("---")
st.markdown("## 🚀 Mission Overview")
st.write("""
The objective of this project is to assist in identifying potentially
hazardous asteroids using Machine Learning techniques.

This dashboard analyzes orbital and physical characteristics of Near-Earth
Objects and predicts whether the asteroid poses a potential threat.
""")

st.markdown("---")

# ==========================================================
# DASHBOARD METRICS
# ==========================================================

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "🌍 Dataset",
        "90,836"
    )
with c2:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )
with c3:
    st.metric(
        "📊 Features",
        "6"
    )
with c4:
    st.metric(
        "🎯 Accuracy",
        "91.6%"
    )

st.markdown("---")

# ==========================================================
# PROJECT INFORMATION CARDS
# ==========================================================

left, right = st.columns(2)

with left:
    st.markdown("""
### 🌍 About Near Earth Objects

Near-Earth Objects (NEOs) are asteroids and comets whose orbits bring them
close to Earth's orbit.

Some of these objects are classified as Potentially Hazardous Asteroids
(PHAs) due to their size and orbital proximity.

Machine Learning enables rapid hazard assessment based on observed
astronomical parameters.
""")

with right:
    st.markdown("""
### 🤖 AI Prediction Pipeline
✔ Data Preprocessing
                
✔ Feature Scaling
                
✔ Random Forest Classification
                
✔ Hazard Prediction
                
✔ Probability Estimation
                
✔ Mission Dashboard
""")

st.markdown("---")

# ==========================================================
# MISSION STATUS
# ==========================================================

st.subheader("🛰 Mission Status")

status1, status2, status3, status4 = st.columns(4)
with status1:
    st.success("🟢 System Online")
with status2:
    st.success("📡 Model Loaded")
with status3:
    st.success("⚙ Prediction Ready")
with status4:
    st.success("🚀 Dashboard Active")

st.markdown("---")

# ==========================================================
# ASTEROID INPUT PANEL
# ==========================================================

st.header("🛰 Asteroid Parameters")

st.write(
    """
Enter the physical and orbital characteristics of the Near-Earth Object.
The trained Random Forest model will estimate whether the asteroid is
Hazardous or Non-Hazardous.
"""
)

st.markdown("---")

# ==========================================================
# INPUT COLUMNS
# ==========================================================

left_col, right_col = st.columns(2)

# ==========================================================
# LEFT COLUMN
# ==========================================================

with left_col:
    st.subheader("📏 Physical Characteristics")
    est_diameter_min = st.number_input(
        "Estimated Diameter Min (km)",
        min_value=0.0,
        value=0.12,
        step=0.01,
        format="%.4f",
        help="Minimum estimated asteroid diameter."
    )
    est_diameter_max = st.number_input(
        "Estimated Diameter Max (km)",
        min_value=0.0,
        value=0.25,
        step=0.01,
        format="%.4f",
        help="Maximum estimated asteroid diameter."
    )
    absolute_magnitude = st.number_input(
        "Absolute Magnitude",
        min_value=0.0,
        max_value=40.0,
        value=22.5,
        step=0.1,
        help="Brightness of the asteroid."
    )

# ==========================================================
# RIGHT COLUMN
# ==========================================================

with right_col:
    st.subheader("🌍 Orbital Characteristics")
    relative_velocity = st.number_input(
        "Relative Velocity (km/h)",
        min_value=0.0,
        value=45000.0,
        step=1000.0,
        help="Relative velocity of the asteroid."
    )
    miss_distance = st.number_input(
        "Miss Distance (km)",
        min_value=0.0,
        value=3500000.0,
        step=10000.0,
        help="Closest distance between Earth and the asteroid."
    )
    sentry_object = st.selectbox(
        "Sentry Object",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Indicates whether NASA's Sentry system tracks this object."
    )

st.markdown("---")

# ==========================================================
# LIVE INPUT PREVIEW
# ==========================================================

st.subheader("📋 Current Mission Parameters")
preview = pd.DataFrame(
    {
        "Parameter": [
            "Estimated Diameter Min",
            "Estimated Diameter Max",
            "Relative Velocity",
            "Miss Distance",
            "Sentry Object",
            "Absolute Magnitude"
        ],
        "Value": [
            est_diameter_min,
            est_diameter_max,
            relative_velocity,
            miss_distance,
            "Yes" if sentry_object == 1 else "No",
            absolute_magnitude
        ]
    }
)

st.dataframe(
    preview,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_dict = {
    "est_diameter_min": est_diameter_min,
    "est_diameter_max": est_diameter_max,
    "relative_velocity": relative_velocity,
    "miss_distance": miss_distance,
    "sentry_object": sentry_object,
    "absolute_magnitude": absolute_magnitude
}

input_df = pd.DataFrame([input_dict])
# Keep the same feature order used during training
input_df = input_df[feature_names]
# Scale the input
scaled_data = scaler.transform(input_df)

# ==========================================================
# PREDICT BUTTON
# ==========================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict = st.button(
        "🚀 Predict Hazard",
        use_container_width=True
    )

# ==========================================================
# PREDICTION ENGINE
# ==========================================================
# Computed unconditionally (not just inside "if predict") because
# later sections (Analytics Dashboard, Mission Summary, Report)
# reference these values even before the button is first clicked.

prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0]
hazard_probability = probability[1] * 100
safe_probability = probability[0] * 100

if predict:
    # ------------------------------------
    # Display Prediction Result
    # ------------------------------------
    st.markdown("---")
    st.header("🚀 Prediction Result")

    # ------------------------------------
    # Prediction Card
    # ------------------------------------

    if prediction == 1:
        st.error("""
# 🔴 HAZARDOUS ASTEROID DETECTED

The predicted Near-Earth Object is classified as **Hazardous**.

NASA should continue monitoring this object's trajectory.
""")

    else:
        st.success("""
# 🟢 NON-HAZARDOUS ASTEROID

The predicted Near-Earth Object is classified as **Non-Hazardous**.

No immediate threat to Earth is detected.
""")

    st.markdown("---")

    # ------------------------------------
    # Probability Metrics
    # ------------------------------------

    st.subheader("📊 Prediction Confidence")

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "🟢 Safe Probability",

            f"{safe_probability:.2f}%"
        )

    with c2:
        st.metric(
            "🔴 Hazard Probability",

            f"{hazard_probability:.2f}%"
        )

    st.progress(hazard_probability / 100)
    st.markdown("---")

    # ------------------------------------
    # Threat Level
    # ------------------------------------

    st.subheader("⚠ Threat Level")
    if hazard_probability < 30:
        st.success("🟢 LOW RISK")
    elif hazard_probability < 70:
        st.warning("🟠 MODERATE RISK")
    else:
        st.error("🔴 HIGH RISK")

    st.markdown("---")

    # ------------------------------------
    # Gauge Chart
    # ------------------------------------

    st.subheader("🎯 Hazard Gauge")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=hazard_probability,
            number={"suffix":"%"},
            title={"text":"Hazard Probability"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"red"},
                "steps":[
                    {"range":[0,30],"color":"green"},
                    {"range":[30,70],"color":"orange"},
                    {"range":[70,100],"color":"red"}
                ]
            }
        )
    )

    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=420

    )

    st.plotly_chart(
        gauge,
        use_container_width=True

    )

    st.markdown("---")

    # ------------------------------------
    # Quick Mission Statistics
    # ------------------------------------

    st.subheader("📡 Mission Statistics")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Velocity (km/h)",
            f"{relative_velocity:,.0f}"
        )

    with m2:
        st.metric(
            "Miss Distance (km)",
            f"{miss_distance:,.0f}"
        )

    with m3:
        st.metric(
            "Absolute Magnitude",
            absolute_magnitude
        )

    st.markdown("---")

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

st.header("📊 Mission Analytics Dashboard")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.subheader("📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature"),
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# PROBABILITY DISTRIBUTION
# ==========================================================

st.subheader("🥧 Prediction Probability")

probability_df = pd.DataFrame({
    "Category":[
        "Safe",
        "Hazardous"
    ],
    "Probability":[
        safe_probability,
        hazard_probability
    ]

})

pie = px.pie(
    probability_df,
    names="Category",
    values="Probability",
    hole=0.55,
    color="Category",
    color_discrete_map={
        "Safe":"#22c55e",
        "Hazardous":"#ef4444"
    }
)

pie.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    height=450
)

st.plotly_chart(
    pie,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# MISSION HEALTH
# ==========================================================

# st.subheader("🛰 Mission Health")
# h1,h2,h3,h4 = st.columns(4)
# with h1:
#     st.metric(
#         "Prediction",
#         "Completed"
#     )
# with h2:
#     st.metric(
#         "Model",
#         "Online"
#     )
# with h3:
#     st.metric(
#         "Scaler",
#         "Loaded"
#     )
# with h4:
#     st.metric(
#         "System",
#         "Healthy"
#     )

# st.markdown("---")

# ==========================================================
# MISSION TELEMETRY
# ==========================================================

st.subheader("📡 Mission Telemetry")

telemetry = pd.DataFrame({
    "Parameter":[
        "Estimated Diameter Min",
        "Estimated Diameter Max",
        "Relative Velocity",
        "Miss Distance",
        "Sentry Object",
        "Absolute Magnitude"
    ],
    "Value":[
        est_diameter_min,
        est_diameter_max,
        f"{relative_velocity:,.2f}",
        f"{miss_distance:,.2f}",
        "Yes" if sentry_object else "No",
        absolute_magnitude
    ]
})

st.dataframe(
    telemetry,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# # ==========================================================
# # AI INSIGHTS
# # ==========================================================

# st.subheader("🧠 AI Insights")
# top_feature = importance_df.iloc[0]["Feature"]
# top_score = importance_df.iloc[0]["Importance"]
# st.info(f"""

# ### Model Observation

# The trained **Random Forest** model predicts the asteroid using six
# astronomical parameters.

# The most influential feature in this prediction is:

# **{top_feature}**

# Feature Importance Score:

# **{top_score:.3f}**

# The prediction confidence is calculated using the probability estimates
# returned by the trained classifier.

# """)

# st.markdown("---")

# ==========================================================
# QUICK STATISTICS
# ==========================================================

st.subheader("📋 Prediction Statistics")
s1, s2, s3 = st.columns(3)
top_feature = importance_df.iloc[0]["Feature"]
with s1:
    st.metric(
        "Safe Probability",
        f"{safe_probability:.2f}%"
    )
with s2:
    st.metric(
        "Hazard Probability",
        f"{hazard_probability:.2f}%"
    )
with s3:
    st.metric(
        "Top Feature",
        top_feature
    )

st.markdown("---")

# ==========================================================
# MISSION SUMMARY
# ==========================================================

st.header("📖 Mission Summary")

if prediction == 0:
    st.success("""
### ✅ Prediction Result

The analyzed Near-Earth Object (NEO) has been classified as **Non-Hazardous**.

Based on the trained Random Forest model, the asteroid does not pose an
immediate threat to Earth.

The probability of hazard is relatively low and no special monitoring is
recommended beyond standard observation.
""")
else:
    st.error("""
### 🚨 Prediction Result

The analyzed Near-Earth Object (NEO) has been classified as **Hazardous**.

According to the trained Random Forest model, this asteroid shows
characteristics associated with potentially hazardous objects.

Further monitoring and orbital analysis are recommended.
""")

st.markdown("---")

# ==========================================================
# WHY THIS PREDICTION
# ==========================================================

st.header("🧠 Why This Prediction?")

top_feature = importance_df.iloc[0]["Feature"]

st.info(f"""
The prediction is based on six astronomical features.

The **Random Forest Classifier** combines all these features to estimate
whether an asteroid is hazardous.

The most influential feature for this trained model is:

### ⭐ {top_feature}

Other important features such as velocity, miss distance,
diameter and absolute magnitude also contribute to the prediction.

The confidence values displayed above are generated using
`predict_proba()` from the trained model.
""")

st.markdown("---")

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.header("🤖 Model Information")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "Model",
        "Random Forest"
    )
with c2:
    st.metric(
        "Training Accuracy",
        "91.6%"
    )
with c3:
    st.metric(
        "Input Features",
        "6"
    )

st.markdown("---")

# ==========================================================
# CREATE REPORT
# ==========================================================

report = f"""
==========================================
NASA NEO HAZARD PREDICTION REPORT
==========================================

Prediction :
{"Hazardous" if prediction else "Non-Hazardous"}
Hazard Probability :
{hazard_probability:.2f} %
Safe Probability :
{safe_probability:.2f} %

------------------------------------------

Input Parameters
Estimated Diameter Min :
{est_diameter_min}
Estimated Diameter Max :
{est_diameter_max}
Relative Velocity :
{relative_velocity}
Miss Distance :
{miss_distance}
Sentry Object :
{"Yes" if sentry_object else "No"}
Absolute Magnitude :
{absolute_magnitude}

------------------------------------------

Machine Learning Model

Random Forest Classifier

Accuracy : 91.6 %

------------------------------------------

Generated by

KAALVYUH

NASA Near Earth Object Prediction System

==========================================
"""

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.download_button(
    label="📥 Download Prediction Report",
    data=report,
    file_name="NASA_NEO_Report.txt",
    mime="text/plain"
)

st.markdown("---")

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.header("🚀 About This Project")

st.write("""
This application was developed as a Machine Learning project to predict
whether a Near-Earth Object (NEO) is hazardous.

The project includes:
         
✔ Data Cleaning
         
✔ Exploratory Data Analysis
         
✔ Feature Engineering
         
✔ Machine Learning Model Comparison
         
✔ Hyperparameter Tuning
         
✔ Random Forest Classifier
         
✔ Streamlit Dashboard

The application demonstrates how Machine Learning can assist in
classifying astronomical objects using NASA's publicly available dataset.
""")

st.markdown("---")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div style='text-align:center;padding:25px;'>

<h2 style='color:#60A5FA;'>
🚀 KAALVYUH
</h2>
<h4>
NASA Near Earth Object Hazard Prediction System
</h4>
<p>
Machine Learning Internship Project
</p>
<p>
Python • Streamlit • Scikit-Learn • Plotly
</p>
<p>
© 2026 KAALVYUH
</p>
</div>
""", unsafe_allow_html=True)