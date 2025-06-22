import pandas as pd
import json

# Load the main ESCO skills CSV
df = pd.read_csv("data/skills_en.csv")

# Optional: Preview columns if unsure
print("Columns:", df.columns.tolist())

# Step 1: Filter only released and English-labeled skills (if available)
# If there's a 'status' or 'language' column, filter accordingly
# For now, assume 'preferredLabel' is the column with skill names

# Step 2: Drop missing values and clean
skills = df['preferredLabel'].dropna().unique()
skills_list = sorted(skill.strip().lower() for skill in skills)

# Step 3: Save to JSON
with open("data/skills_list.json", "w") as f:
    json.dump(skills_list, f, indent=2)

print(f"✅ Saved {len(skills_list)} skills to data/skills_list.json")
