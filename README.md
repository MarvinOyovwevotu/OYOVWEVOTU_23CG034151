# Emotion Detection (HOG + SVM)

## Setup
1. Create virtualenv:
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

2. Install:
   pip install -r requirements.txt

3. Put training images in data/<label>/*.jpg (e.g. data/happy/1.jpg)

4. Train model:
   python model.py

5. Run app:
   streamlit run app.py

Model saved to models/emotion_svm.pkl
