import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('Assessment_with_GPA_IQ.csv')

print(df.to_string())

print(df['Department'].unique())

df['Department']=df['Department'].str.strip().str.lower()

def standardize_department(dept):

    if any(keyword in dept for keyword in ["cs", "computer", "s/e","se", "it", "software", "cse"]):
        return "CS&IT Department"
    elif "radiology" in dept:
        return "Radiology Department"
    elif "pharmacy" in dept:
        return "Pharmacy Department"
    elif "cardiology" in dept:
        return "Cardiology Department"
    elif "english" in dept:
        return "English Department"
    elif "nursing" or "bsn" in dept:
        return"Nursing Department"
    else:
        return dept

df["Department"] = df["Department"].apply(standardize_department)

print(df['Department'].unique())
print(df["Department"].value_counts())

df.isnull().sum()

def clean_semester(sem):
    sem = str(sem).lower().strip()

    if sem in ["1", "1st", "first"]:
        return 1
    elif sem in ["2", "2nd", "2nd semester", "second", "2 nd"]:
        return 2
    elif sem in ["3", "3rd", "third"]:
        return 3
    elif sem in ["4", "4th", "4th semester"]:
        return 4
    elif sem in ["5", "5th", "fifth"]:
        return 5
    elif sem in ["6", "6th", "6th semester", "Software Engineering 6th"]:
        return 6
    elif sem in ["7", "7th", "seventh"]:
        return 7
    elif sem in ["8", "8th", "eighth"]:
        return 8
    elif sem in ["10", "10th", "tenth"]:
        return 10
    else:
        return None  # return a missing value for unknowns

df["Semester"] = df["Semester"].apply(clean_semester)
print(df["Semester"].value_counts().sort_index())


df.info()

df.head()

# FOR VISULAIZATION
print(df.columns.tolist())
def classify_user(freq):
    if freq == "Daily":
        return "Heavy"
    elif freq in ["A few times a week", "Occasionally"]:
        return "Occasional"
    else:
        return "Rare"

df["User Type"] = df["How often do you use AI tools like ChatGPT, Copilot, Gemini, etc?"].apply(classify_user)

df["Dependency"] = df['Do you feel you’ve become dependent on AI tools for academic tasks?'].map({
    "Yes": "Dependent",
    "No": "Not Dependent",
    "Maybe": "Neutral"
})

likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

df["Understand Less Score"] = df['I understand less when I use AI tools too much'].map(likert_map)
df["Struggle Score"] = df['I struggle to solve tasks without using AI now.'].map(likert_map)
df["Creativity Score"] = df['I feel less creative when I use AI tools.'].map(likert_map)

df["Impact Score"] = df[["Understand Less Score", "Struggle Score", "Creativity Score"]].mean(axis=1)

# Label learning impact based on average score
df["Learning Impact"] = df["Impact Score"].apply(lambda x: "Negative" if x >= 4 else "Positive")


user_counts = df["User Type"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(user_counts, labels=user_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("AI User Types")
plt.show()

impact_counts = df["Learning Impact"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(impact_counts.index, impact_counts.values, color=["skyblue", "orange"])
plt.title("Learning Impact of AI Usage")
plt.xlabel("Impact Type")
plt.ylabel("Number of Students")
plt.show()

dep_counts = df["Dependency"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(dep_counts, labels=dep_counts.index, autopct='%1.1f%%',startangle=90)
plt.title("AI Dependency Among Students")
plt.show()

df["Cognitive Impact"] = df["Impact Score"].apply(lambda x: "Negative" if x >= 4 else "Positive")
cog_counts = df["Cognitive Impact"].value_counts()

avg_gpa = df.groupby("User Type")["GPA"].mean()


plt.figure(figsize=(6, 4))
plt.bar(avg_gpa.index, avg_gpa.values, color=["tomato", "orange", "green"])
plt.title("Average GPA by AI User Type")
plt.xlabel("User Type")
plt.ylabel("Average GPA")
plt.ylim(2.5, 4.0)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Count responses for each question
q1_counts = df["Q1_Sequence"].value_counts().sort_index()
q2_counts = df["Q2_Logic"].value_counts().sort_index()
q3_counts = df["Q3_Math"].value_counts().sort_index()


df["IQ_Correct_Count"] = (
    (df["Q1_Sequence"] == 32).astype(int) +
    (df["Q2_Logic"].str.lower().str.strip() == "no").astype(int) +
    (df["Q3_Math"] == 8).astype(int)
)

iq_vs_ai = pd.crosstab(df["User Type"], df["IQ_Correct_Count"])

iq_vs_ai.plot(kind="bar", stacked=True, figsize=(8,5), colormap="Set2")
plt.title("IQ Performance vs AI User Type")
plt.xlabel("AI User Type")
plt.ylabel("Number of Students")
plt.legend(title="IQ Questions Correct")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



