# 🛡️ SMS & Email Spam Classifier

A machine learning web application that classifies SMS and email messages as **Spam** or **Not Spam** using NLP and a trained classification model.

## 🚀 Live Demo
https://sms-spam-classifier-system.streamlit.app/

## ✨ Features
- Spam detection using machine learning.
- Text preprocessing with NLTK.
- TF-IDF feature extraction.
- Interactive Streamlit interface with example messages.
- Instant predictions.

## 🛠️ Tech Stack
Python, NLTK, Scikit-learn, Streamlit, Pandas, NumPy, and Pickle.

## ⚙️ Workflow
1. Preprocess the input message using tokenization, stopword removal, and stemming.
2. Convert text into numerical features using TF-IDF.
3. Predict whether the message is spam or not spam.
4. Display the result in the web interface.

## 📂 Project Structure
```text
SMS-Email-spam-classifier/
├── app/
│   ├── app.py
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── requirements.txt
├── dataset/
├── sms-spam detection.ipynb
└── README.md
```

## 🖥️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Shivam760/sms-email-spam-classifier.git
cd https://github.com/Shivam760/sms-email-spam-classifier
```

**2. Install dependencies**
```bash
pip install -r app/requirements.txt
```

**3. Run the application**
```bash
streamlit run app/app.py
```

## ☁️ Deployment
The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/) by connecting the GitHub repository and selecting `app/app.py` as the main file.

## ⚠️ Limitations
- Predictions may not always be accurate.
- Messages classified as Not Spam are not guaranteed to be safe.

## 👨‍💻 Author
**Shivam**


*Built with Python, NLP, Machine Learning, and Streamlit.*
