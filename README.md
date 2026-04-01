# Symptora AI - Intelligent Health Assessment Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![License](https://img.shields.io/badge/License-Proprietary-red.svg)
![Author](https://img.shields.io/badge/Author-Krish-purple.svg)

A machine learning-powered healthcare chatbot that predicts potential diseases based on user-reported symptoms. Built with Flask and trained using Random Forest classification on a comprehensive health dataset.

## Features

- **AI-Powered Disease Prediction**: Uses a trained Random Forest model to analyze symptoms and predict potential health conditions
- **Multi-Disease Classification**: Supports prediction across 100+ diseases with confidence scoring
- **Top 3 Predictions**: Returns the three most likely conditions with confidence percentages
- **Grok Guidance Layer**: Generates AI-based precautions, next steps, and urgent red-flag signs from the predicted condition
- **Interactive Web Interface**: Modern, responsive chat-style interface with DNA helix animation
- **Symptom Severity Tracking**: Allows users to specify symptom intensity (mild, moderate, severe)
- **REST API Support**: JSON-based API endpoint for integration with other applications
- **Print-Friendly Reports**: Generate assessment summaries for healthcare consultations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| Machine Learning | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Frontend | HTML5, CSS3, JavaScript |
| Styling | Custom CSS with CSS Variables |

## Project Structure

```
Symptora AI/
├── app.py                          # Main Flask application
├── disease_model.pkl               # Trained Random Forest model
├── label_encoder.pkl               # Label encoder for disease classes
├── create_presentation.py          # Presentation generator script
├── Health Disease Classification   # Jupyter notebook for model training
│   Using Machine Learning.ipynb
├── Datasets/
│   ├── Dataset For Model Training.csv
│   ├── generate_health_dataset.py  # Dataset generation script
│   ├── health_dataset.csv
│   └── validation_report.csv
├── static/
│   └── style.css                   # Stylesheet with modern design
└── templates/
    └── index.html                  # Main web interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 8GB+ RAM recommended for model loading

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishyadav90/Symptora-AI.git
   cd Symptora-AI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn joblib
   ```

4. **Configure Grok API (optional but recommended)**

   Set your xAI Grok key as an environment variable:

   ```bash
   # Windows PowerShell
   $env:GROK_API_KEY="your_grok_api_key"

   # macOS/Linux
   export GROK_API_KEY="your_grok_api_key"
   ```

4. **Download/Generate model files**
   
   The model files are not pushed to GitHub due to size limits. You have two options:
   
   **Option A: Train the model yourself**
   - Open `Health Disease Classification Using Machine Learning.ipynb`
   - Run all cells to train and save the model
   - This will generate `disease_model.pkl` and `label_encoder.pkl`
   
   **Option B: Download pre-trained files**
   - Download the model files from Hugging Face: https://huggingface.co/krish78787/disease_model
   
   Required files in root directory:
   - `disease_model.pkl` (~722 MB)
   - `label_encoder.pkl` (~1 KB)

## Usage

### Running the Application

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000`

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Enter your demographic information (age, gender, BMI)
3. Select your symptoms and specify severity levels
4. Click "Analyze Symptoms" to get predictions
5. View the top 3 most likely conditions with confidence scores

### API Endpoints

#### Health Check
```http
GET /health
```
Returns the server status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoder_loaded": true
}
```

#### Predict Disease (Form)
```http
POST /predict
Content-Type: application/x-www-form-urlencoded
```
Submit symptom data via form and receive HTML response.

#### Predict Disease (API)
```http
POST /api/predict
Content-Type: application/json
```
Submit symptom data as JSON and receive JSON response with predictions.

**Response:**
```json
{
  "success": true,
  "predictions": [
    {"disease": "Common Cold", "confidence": 45.2, "level": "High Match"},
    {"disease": "Influenza", "confidence": 28.1, "level": "Moderate Match"},
    {"disease": "Allergic Rhinitis", "confidence": 12.5, "level": "Possible Match"}
   ],
   "guidance": {
      "summary": "Likely viral upper respiratory pattern.",
      "precautions": ["Hydrate well", "Rest", "Monitor fever"],
      "next_steps": ["Consult a clinician if persistent", "Track symptoms for 48h"],
      "urgent_red_flags": ["Breathing difficulty", "Chest pain", "Confusion"],
      "disclaimer": "Informational only"
   },
  "disclaimer": "This is not a medical diagnosis. Please consult a healthcare professional."
}
```

## Model Information

### Training Data
- **Samples**: 200-300 samples per disease
- **Features**: 832 features after one-hot encoding
- **Symptoms**: Severity levels (0-3) for 200+ symptoms
- **Demographics**: Age, Gender, BMI, Smoking Status

### Algorithm
- **Model**: Random Forest Classifier
- **Preprocessing**: One-hot encoding for categorical variables
- **Output**: Probability distribution across all disease classes

### Diseases Covered
The model covers various categories including:
- Respiratory diseases (Cold, Flu, COVID-19, Pneumonia, Bronchitis)
- Cardiovascular conditions
- Gastrointestinal disorders
- Neurological conditions
- Dermatological issues
- Mental health conditions
- And many more...

## Screenshots

### Main Interface
The application features a modern chat-style interface with:
- DNA helix animated background
- Step-by-step symptom input
- Real-time form validation
- Mobile-responsive design

### Prediction Results
Results display:
- Primary prediction with confidence score
- Top 3 alternative diagnoses
- Patient summary for printing
- Medical disclaimer

## Development

### Training a New Model

1. Open `Health Disease Classification Using Machine Learning.ipynb`
2. Run all cells to:
   - Load and preprocess the dataset
   - Train the Random Forest model
   - Save `disease_model.pkl` and `label_encoder.pkl`

### Generating Dataset

```bash
cd Datasets
python generate_health_dataset.py
```

This creates a synthetic healthcare dataset with:
- Probabilistic symptom generation
- Realistic demographic distributions
- Symptom severity and duration features

## Important Disclaimer

**This application is for educational and informational purposes only.**

- It is NOT a substitute for professional medical advice, diagnosis, or treatment
- Always seek the advice of a qualified healthcare provider
- Never disregard professional medical advice because of AI predictions
- In case of emergency, contact emergency services immediately

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

**Copyright (c) 2026 Krish. All Rights Reserved.**

This project is proprietary software. No permission is granted to use, copy, modify, or distribute this software without explicit written permission from the author. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Healthcare data patterns based on medical literature
- UI/UX inspired by modern health applications
- Built as part of PBL-4 Healthcare Chatbot project

## Contact

For questions or feedback, please contact Krish.

---

**Made with care by Krish for better health awareness**