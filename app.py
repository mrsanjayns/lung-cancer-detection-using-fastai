import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import os
from flask import Flask, request, render_template, redirect, url_for
from fastai.vision.all import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a dummy DataLoaders object
path = Path('/Users/sanjayns/Downloads/lung-cancer-detection/lung_dataset')
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=10.0)
)

# Create the learner and load the weights
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])
learn.load('/Users/sanjayns/Downloads/lung-cancer-detection/lung_cancer_model')

# Define class explanations
class_explanations = {
    "lung_aca": {
        "name": "Adenocarcinoma",
        "description": "A type of non-small cell lung cancer that forms in the outer parts of the lung.",
        "symptoms": "Persistent cough, coughing up blood, chest pain, shortness of breath, wheezing, fatigue, and unexplained weight loss.",
        "risk_factors": "Smoking, exposure to secondhand smoke, radon gas, asbestos, air pollution, and family history of lung cancer.",
        "prevention": "Quit smoking, avoid secondhand smoke, test home for radon, avoid carcinogens at work, eat a diet rich in fruits and vegetables, exercise regularly.",
        "treatment": "Surgery, radiation therapy, chemotherapy, targeted drug therapy, and immunotherapy, depending on the stage and individual case.",
        "prognosis": "Varies depending on the stage at diagnosis. Early detection significantly improves outcomes.",
        "screening": "Annual low-dose CT scans recommended for high-risk individuals.",
        "research": "Ongoing studies on targeted therapies and immunotherapies show promising results."
    },
    "lung_scc": {
        "name": "Squamous Cell Carcinoma",
        "description": "A type of non-small cell lung cancer that forms in the central part of the lungs, near the main airway (bronchus).",
        "symptoms": "Persistent cough, coughing up blood, hoarseness, wheezing, difficulty swallowing, and recurrent respiratory infections.",
        "risk_factors": "Heavy smoking, exposure to industrial chemicals, radiation therapy to the chest, and chronic lung disease.",
        "prevention": "Quit smoking, avoid occupational exposure to harmful substances, use protective equipment in high-risk work environments, get regular check-ups.",
        "treatment": "Surgery, radiation therapy, chemotherapy, and immunotherapy, often used in combination depending on the stage of cancer.",
        "prognosis": "Generally has a poorer prognosis than adenocarcinoma, but outcomes have improved with advances in treatment.",
        "screening": "Similar to adenocarcinoma, annual low-dose CT scans for high-risk individuals.",
        "research": "Studies focus on improving targeted therapies and understanding resistance mechanisms."
    },
    "lung_n": {
        "name": "Benign",
        "description": "Non-cancerous growths that do not spread to other parts of the body. Often harmless but may require monitoring.",
        "importance": "Regular check-ups are still important even with benign results, as early detection of any changes is crucial.",
        "health_tips": "Maintain a healthy lifestyle, avoid smoking, exercise regularly, eat a balanced diet, and manage stress.",
        "follow_up": "Continue regular screenings as recommended by your doctor, especially if you have risk factors for lung cancer.",
        "prevention": "Focus on lung health by avoiding pollutants, getting vaccinated against pneumonia and flu, and practicing good respiratory hygiene.",
        "risk_reduction": "Even with normal results, it's important to reduce risk factors like smoking and exposure to environmental toxins.",
        "general_lung_health": "Practice deep breathing exercises, stay hydrated, and avoid excessive alcohol consumption to maintain lung health."
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        img = PILImage.create(filename)
        pred, pred_idx, probs = learn.predict(img)
        explanation = class_explanations.get(str(pred), {'name': 'Unknown', 'description': 'No explanation available.'})
        return render_template('result.html', 
                               prediction=pred, 
                               probability=probs[pred_idx].item(), 
                               filename=file.filename,
                               explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')