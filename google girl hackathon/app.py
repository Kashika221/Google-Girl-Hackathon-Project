from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('disease_prediction.pkl', "rb"))
app = Flask(__name__)

disease = ['Acne', 'Allergic Rhinitis', "Alzheimer's Disease", 'Anemia',
       'Anxiety Disorders', 'Appendicitis', 'Asthma', 'Atherosclerosis',
       'Autism Spectrum Disorder (ASD)', 'Bipolar Disorder',
       'Bladder Cancer', 'Brain Tumor', 'Breast Cancer', 'Bronchitis',
       'Cataracts', 'Cerebral Palsy', 'Chickenpox', 'Cholecystitis',
       'Cholera', 'Chronic Kidney Disease',
       'Chronic Obstructive Pulmonary Disease (COPD)',
       'Chronic Obstructive Pulmonary...', 'Cirrhosis',
       'Colorectal Cancer', 'Common Cold', 'Conjunctivitis (Pink Eye)',
       'Coronary Artery Disease', "Crohn's Disease", 'Cystic Fibrosis',
       'Dementia', 'Dengue Fever', 'Depression', 'Diabetes',
       'Diverticulitis', 'Down Syndrome',
       'Eating Disorders (Anorexia,...', 'Ebola Virus', 'Eczema',
       'Endometriosis', 'Epilepsy', 'Esophageal Cancer', 'Fibromyalgia',
       'Gastroenteritis', 'Glaucoma', 'Gout', 'HIV/AIDS', 'Hemophilia',
       'Hemorrhoids', 'Hepatitis', 'Hepatitis B', 'Hyperglycemia',
       'Hypertension', 'Hypertensive Heart Disease', 'Hyperthyroidism',
       'Hypoglycemia', 'Hypothyroidism', 'Influenza', 'Kidney Cancer',
       'Kidney Disease', 'Klinefelter Syndrome', 'Liver Cancer',
       'Liver Disease', 'Lung Cancer', 'Lyme Disease', 'Lymphoma',
       'Malaria', 'Marfan Syndrome', 'Measles', 'Melanoma', 'Migraine',
       'Multiple Sclerosis', 'Mumps', 'Muscular Dystrophy',
       'Myocardial Infarction (Heart...',
       'Obsessive-Compulsive Disorde...', 'Osteoarthritis',
       'Osteomyelitis', 'Osteoporosis', 'Otitis Media (Ear Infection)',
       'Ovarian Cancer', 'Pancreatic Cancer', 'Pancreatitis',
       "Parkinson's Disease", 'Pneumocystis Pneumonia (PCP)', 'Pneumonia',
       'Pneumothorax', 'Polio', 'Polycystic Ovary Syndrome (PCOS)',
       'Prader-Willi Syndrome', 'Prostate Cancer', 'Psoriasis', 'Rabies',
       'Rheumatoid Arthritis', 'Rubella', 'Schizophrenia', 'Scoliosis',
       'Sepsis', 'Sickle Cell Anemia', 'Sinusitis', 'Sleep Apnea',
       'Spina Bifida', 'Stroke', 'Systemic Lupus Erythematosus...',
       'Testicular Cancer', 'Tetanus', 'Thyroid Cancer', 'Tonsillitis',
       'Tourette Syndrome', 'Tuberculosis', 'Turner Syndrome',
       'Typhoid Fever', 'Ulcerative Colitis', 'Urinary Tract Infection',
       'Urinary Tract Infection (UTI)', 'Williams Syndrome', 'Zika Virus']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict_house_price():
    Disease = disease.index(request.form.get('Disease'))
    Fever = int(request.form.get('Fever'))
    Cough = int(request.form.get('Cough'))
    Fatigue = int(request.form.get('Fatigue'))
    Difficulty_Breathing = int(request.form.get('Difficulty_Breathing'))
    Age = int(request.form.get('Age'))
    Gender = int(request.form.get('Gender'))
    Blood_Pressure = int(request.form.get('Blood_Pressure'))
    Cholesterol_Level = int(request.form.get('Cholesterol_Level'))
    

    # prediction
    result = model.predict(np.array([Disease, Fever, Cough, Fatigue, Difficulty_Breathing, Age, Gender, Blood_Pressure, Cholesterol_Level]).reshape(1, 9))
    if(result[0] == 0):
        result = 'Negative'
    else:
        result = 'Positive'
    print(result)

    return render_template('index.html', result = result)

if __name__ == '__main__':
    app.run(debug = True)