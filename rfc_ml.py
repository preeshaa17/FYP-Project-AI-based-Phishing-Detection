import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, hstack

# --- File paths ---
good_folder = r"C:\Users\USER\OneDrive\Desktop\FYP1 Preeshaa\Good Email"
bad_folder  = r"C:\Users\USER\OneDrive\Desktop\FYP1 Preeshaa\Bad Email"

# --- Utility Functions ---
def extract_url_if_exists(line):
    url_regex = r'(https?://[^\s]+|www\.[^\s]+)'
    match = re.search(url_regex, line)
    return match.group(0) if match else ""

def detect_urgent_phrases(text):
    urgent_phrases = [
        "verify your account", "act now", "urgent", "immediate action", "click here",
        "update your information", "suspend", "limited time", "reset your password", "login now"
    ]
    text = text.lower()
    return int(any(phrase in text for phrase in urgent_phrases))

def detect_suspicious_anchor_text(text):
    anchor_regex = r'<a\s+href="([^"]+)">([^<]+)</a>'
    matches = re.findall(anchor_regex, text)
    for href, anchor in matches:
        if href and anchor and anchor not in href:
            return 1
    return 0

def extract_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_mixed_folder(folder_path, label, excel_column='Content'):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            if file_path.endswith('.xlsx') and os.path.isfile(file_path):
                try:
                    df_excel = pd.read_excel(file_path)
                    if excel_column not in df_excel.columns:
                        print(f"⚠️ Skipping {filename}: column '{excel_column}' not found.")
                        continue
                    for line in df_excel[excel_column].dropna():
                        line = str(line).strip()
                        if line:
                            url = extract_url_if_exists(line)
                            data.append({'text': line, 'url': url, 'label': label})
                except Exception as e:
                    print(f"❌ Error reading Excel file {filename}: {e}")

            elif os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                url = extract_url_if_exists(line)
                                data.append({'text': line, 'url': url, 'label': label})
                except Exception as e:
                    print(f"❌ Error reading text file {filename}: {e}")
    return pd.DataFrame(data)

# --- Load full dataset ---
df_good = process_mixed_folder(good_folder, label=0, excel_column='Content')
df_bad  = process_mixed_folder(bad_folder, label=1, excel_column='Content')
df = pd.concat([df_good, df_bad], ignore_index=True)

# --- Feature Engineering ---
def extract_features_from_url(url):
    return {
        'url_length': len(str(url)),
        'num_dots': str(url).count('.'),
        'has_https': int('https' in str(url).lower()),
        'has_at_symbol': int('@' in str(url)),
        'has_hyphen': int('-' in str(url)),
        'has_ip_address': int(bool(re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', str(url)))),
        'subdomain_count': len(str(url).split('//')[-1].split('/')[0].split('.')) - 2 if '.' in str(url) else 0
    }

url_features = df['url'].apply(extract_features_from_url)
X_url = pd.DataFrame(url_features.tolist())
X_url_sparse = csr_matrix(X_url.values)

df['urgent_phrase'] = df['text'].apply(detect_urgent_phrases)
df['suspicious_anchor'] = df['text'].apply(detect_suspicious_anchor_text)
df['sentiment_score'] = df['text'].apply(extract_sentiment_score)
X_custom_text = df[['urgent_phrase', 'suspicious_anchor', 'sentiment_score']]
X_custom_sparse = csr_matrix(X_custom_text.values)

tfidf = TfidfVectorizer(max_features=300)
X_tfidf = tfidf.fit_transform(df['text'].astype(str))

X_combined_sparse = hstack([X_url_sparse, X_custom_sparse, X_tfidf])
y = df['label'].values

X_train_sparse, X_test_sparse, y_train, y_test = train_test_split(
    X_combined_sparse, y, test_size=0.3, random_state=42, stratify=y
)

# --- Train RandomForestClassifier ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sparse, y_train)
y_pred = model.predict(X_test_sparse)

# --- Evaluation ---
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- PIE CHART ---
labels = ['Legitimate (0)', 'Phishing (1)']
pred_counts = pd.Series(y_pred).value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
plt.title("Prediction Distribution (Pie Chart)")
plt.tight_layout()
plt.show()

# --- BAR CHART ---
actual_counts = pd.Series(y_test).value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
plt.bar(x + width/2, predicted_counts, width, label='Predicted', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Emails')
plt.title('Actual vs Predicted Email Classification')
plt.xticks(ticks=x, labels=labels)
plt.legend()
plt.tight_layout()
plt.show()

print("✅ Script started")
print(df.head())
print(len(df))