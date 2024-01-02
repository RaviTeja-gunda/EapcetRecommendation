import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Define branch mapping
# ... (your branch_map code)
branch_map = {
    "CIV": ["civil", "construction", "building", "bridge", "infrastructure", "road", "dam"],
    "CSE": ["computer", "science", "cse", "software", "IT", "programming", "coding", "algorithm", "data", "structure", "system"],
    "ECE": ["ece", "electronics", "circuit", "communication", "networking", "wireless", "rf", "antenna"],
    "EEE": ["eee", "electricity", "power", "energy", "circuit", "control"],
    "MEC": ["design", "manufacturing", "robotics", "mechanic", "fluid", "thermodynamics"],
    "CSM": ["csm", "mathematics", "algorithm", "theory", "computation", "modeling", "analysis", "problem", "solving"],
    "CSO": ["cso", "operation", "management", "business", "optimization", "decision", "making"],
    "CSD": ["csd", "datascience", "machine", "learning", "statistic", "artificial", "intelligence", "bigdata", "visualization"],
    "INF": ["inf", "information", "system", "technology", "management", "database", "network", "software"],
    "PHD": ["phd", "physic", "optics", "material", "quantum", "mechanic", "semiconductor"],
    "PHM": ["phm", "physic", "modeling", "theoretical", "computational"],
    "AGR": ["agr", "farming", "crop", "animal", "soil", "environment", "food", "production", "sustainable", "agriculture"],
    "AIM": ["aim", "ai", "machine", "learning", "deep", "learning", "computer", "vision", "natural", "language", "processing", "robotics"],
    "MIN": ["min", "mining", "mineral", "resource", "geology", "extraction", "processing", "earth", "science"],
    "PET": ["pet", "oil", "gas", "exploration", "production", "drilling", "reservoir", "petroleum"],
    "EIE": ["eie", "instrumentation", "control", "automation", "process", "control", "sensor", "actuator"],
    "CAD": ["cad", "design", "engineering", "3d", "modeling", "simulation"],
    "AID": ["aid", "automotive", "design", "engineering", "manufacturing", "performance", "safety"],
    "AUT": ["aut", "automobile", "vehicle", "engine", "transmission", "chassis", "brake", "dynamic"],
    "CSC": ["csc", "computer", "network", "communication", "security", "cybersecurity", "networking", "protocol"],
    "PEE": ["pee", "power", "electronic", "renewable", "control", "smart", "grid", "energy", "efficiency"],
    "CAI": ["cai", "industrial", "automation", "robotics", "software", "application", "manufacturing"],
    "FDE": ["fde", "food", "processing", "technology", "safety", "preservation", "quality", "control"],
    "CHE": ["che", "chemical", "process", "design", "development", "production", "optimization", "material"],
    "CIT": ["cit", "informatic", "data", "modeling", "simulation", "drug", "discovery", "material"],
    "PHE": ["phe", "pharmaceutical", "drug", "development", "manufacturing", "quality", "control", "process", "engineering", "biopharmaceutical"],
    "CS": ["cs", "software", "algorithm", "data", "structure", "web", "development", "mobile",],
    "DS": ["ds", "data", "analysis", "machine", "learning", "statistic", "visualization", "bigdata", "predictive"],
    "CSG": ["csg", "game", "development", "design", "graphic", "animation", "virtual", "reality", "augmented",],
    "CSB": ["csb", "bioinformatic", "computational", "biology", "genomic", "proteomic"],
    "AI": ["ai", "machine", "learning", "deep", "learning", "computer", "vision", "robotics", "expert"],
    "IOT": ["iot", "connected", "device", "sensor", "network", "bigdata", "cloud", "computing", "smart" "home"],
    "CIC": ["cic", "construction", "management", "infrastructure", "project"],
    "CBA": ["cba", "bioinformatic", "data", "analysis", "statistic", "modeling", "simulation", "drug", "personalized", "medicine"],
    "ASE": ["ase", "aeronautic", "astronautic", "propulsion", "aerodynamic" "flight" ,"dynamic", "control", "spacecraft"]
}


# Define stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and pre-process data
data = pd.read_csv("eapcet2023.csv")

# Function to predict colleges based on text input and rank
desired_branch = None
# Placeholder for filtered_data
filtered_data = pd.DataFrame()


def predict_colleges(text, rank, gender, data, target_rank):

    # Pre-process student text
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]

    # Identify desired branch based on keywords in text
    desired_branch = None
    for entity, keywords in branch_map.items():
        for keyword in keywords:
            if keyword in text:
                desired_branch = entity
                break

    # Filter colleges by rank range
    if gender == "Male":
        filtered_data = data[
            (data[target_rank] < rank + 30000) & (data[target_rank] > rank - 2000) & (data["branch_code"] == desired_branch)
            & (data["COED"] == "COED")
        ]
    else:
        filtered_data = data[
            (data[target_rank] < rank + 30000) & (data[target_rank] > rank - 2000) & (data["branch_code"] == desired_branch)
        ]

    # Check if branch identified
    if desired_branch is None or len(filtered_data) < 5:
        if len(filtered_data) > 0:
            st.write(f"Your desired branch is {desired_branch}")
            st.write("Very few number of colleges found as per your desired branch.....")
        else:
            st.write("Branch could not be identified from the text....")

        # Update recommendations based on rank
        if gender == "Male":
            filtered_data = data[
                (data[target_rank] < rank + 30000) & (data[target_rank] > rank - 2000) & (data["COED"] == "COED")
            ]
        else:
            filtered_data = data[
                (data[target_rank] < rank + 30000) & (data[target_rank] > rank - 2000)
            ]

        st.subheader("Here are the recommendations based on your rank:")
        st.write(f"Recommended colleges for rank {rank}, gender {gender}, and caste {student_caste}:")
        st.write(filtered_data[["inst_code", target_rank, "COED", "branch_code", "FEE"]].head(30))
        return

    # Recommend colleges
    st.write(f"Your desired branch is {desired_branch}")
    st.subheader(
        f"Recommended colleges for rank {rank}, desired branch {desired_branch}, gender {gender}, and caste {student_caste}:")
    st.write(filtered_data[["inst_code", target_rank, "COED", "branch_code", "FEE"]].head(30))

# Streamlit UI
st.title("College Recommendation App")

# User input
student_text = st.text_area("Enter your interests:", value="", placeholder="I am interested in .....")
student_rank = st.number_input("Enter your rank:",value="")
student_gender = st.selectbox("Select your gender:", ["Male", "Female"])
student_caste = st.selectbox("Select your caste:", ["OC", "EWS", "SC", "ST", "BCA", "BCB", "BCC", "BCD", "BCE"])

# Button to trigger recommendations
if st.button("Get College Recommendations"):
    if student_gender == "Male":
        if student_caste == "OC":
            target_rank = "OC_BOYS"
        elif student_caste == "EWS":
            target_rank = "OC_EWS_BOYS"
        elif student_caste == "SC":
            target_rank = "SC_BOYS"
        elif student_caste == "ST":
            target_rank = "ST_BOYS"
        elif student_caste == "BCA":
            target_rank = "BCA_BOYS"
        elif student_caste == "BCB":
            target_rank = "BCB_BOYS"
        elif student_caste == "BCC":
            target_rank = "BCC_BOYS"
        elif student_caste == "BCD":
            target_rank = "BCD_BOYS"
        elif student_caste == "BCE":
            target_rank = "BCE_BOYS"
    else:
        if student_caste == "OC":
            target_rank = "OC_GIRLS"
        elif student_caste == "EWS":
            target_rank = "OC_EWS_GIRLS"
        elif student_caste == "SC":
            target_rank = "SC_GIRLS"
        elif student_caste == "ST":
            target_rank = "ST_GIRLS"
        elif student_caste == "BCA":
            target_rank = "BCA_GIRLS"
        elif student_caste == "BCB":
            target_rank = "BCB_GIRLS"
        elif student_caste == "BCC":
            target_rank = "BCC_GIRLS"
        elif student_caste == "BCD":
            target_rank = "BCD_GIRLS"
        elif student_caste == "BCE":
            target_rank = "BCE_GIRLS"
    predict_colleges(student_text, student_rank, student_gender, data, target_rank)

# Display recommendations using Streamlit
st.subheader("College Recommendations")
st.write(f"Recommended colleges for rank {student_rank}, desired branch {desired_branch}, gender {student_gender}, and caste {student_caste}:")

table_columns = ['inst_code', 'rank_cutoff', 'inst_type', 'branch_code', 'FEE']
recommended_colleges = []
for index, row in filtered_data.iterrows():
    recommended_colleges.append([row[col] for col in table_columns])
recommended_colleges=sorted(recommended_colleges,key=lambda x:x[1])

st.table(pd.DataFrame(recommended_colleges, columns=table_columns))
