import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

data = {
    'Skill': ['Python', 'Java', 'Machine Learning', 'Python', 'Java'],
    'Work_Environment': ['Online', 'Offline', 'Hybrid', 'Hybrid', 'Online'],
    'Career': ['Data Scientist', 'Backend Developer', 'AI Engineer', 'Full Stack Developer', 'Mobile Developer']
}

df = pd.DataFrame(data)

skill_encoder = LabelEncoder()
env_encoder = LabelEncoder()
career_encoder = LabelEncoder()

df['Skill_enc'] = skill_encoder.fit_transform(df['Skill'])
df['Env_enc'] = env_encoder.fit_transform(df['Work_Environment'])
df['Career_enc'] = career_encoder.fit_transform(df['Career'])

joblib.dump(skill_encoder, 'skill_encoder.pkl')
joblib.dump(env_encoder, 'env_encoder.pkl')
joblib.dump(career_encoder, 'career_encoder.pkl')

X = df[['Skill_enc', 'Env_enc']]
y = df['Career_enc']

model = DecisionTreeClassifier()
model.fit(X, y)
joblib.dump(model, 'career_model.pkl')
