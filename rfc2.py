import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, hstack

# === File paths ===
good_folder = r"C:\Users\USER\OneDrive\Desktop\FYP1 Preeshaa\Good Email"
bad_folder  = r"C:\Users\USER\OneDrive\Desktop\FYP1 Preeshaa\Bad Email"

# === Feature helper functions ===
def extract_url_if_exists(line):
    url_regex = r'(https?://[^\s]+|www\.[^\s]+)'
    match = re.search(url_regex, line)
    return match.group(0) if match else ""

def detect_urgent_phrases(text):
    phrases = [
        "verify your account", "act now", "urgent", "immediate action", "click here",
        "update your information", "suspend", "limited time", "reset your password", "login now"
    ]
    text = text.lower()
    return int(any(p in text for p in phrases))

def detect_suspicious_anchor_text(text):
    anchor_regex = r'<a\s+href="([^"]+)">([^<]+)</a>'
    matches = re.findall(anchor_regex, text)
    return int(any(href and anchor and anchor not in href for href, anchor in matches))

def extract_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

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

# === Folder processing ===
def process_mixed_folder(folder_path, label, excel_column='Content'):
    """
    Scans a folder (and subfolders) for .xlsx, .csv, .txt and unknown files.
    Extracts text content, URLs, and assigns a label.
    """
    data_frames = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip empty files
            if os.path.getsize(file_path) == 0:
                print(f"⚠️ Skipping empty file: {filename}")
                continue

            # --- Handle Excel files (.xlsx) ---
            if filename.endswith('.xlsx'):
                try:
                    df_excel = pd.read_excel(file_path)  # no usecols here

                    # detect the text column automatically
                    if excel_column in df_excel.columns:
                        text_col = excel_column
                    else:
                        text_col = df_excel.columns[0]

                    df_excel = df_excel.dropna(subset=[text_col])
                    if df_excel.empty:
                        print(f"⚠️ Skipping {filename}: no usable data.")
                        continue

                    df_excel = df_excel.rename(columns={text_col: 'text'})
                    df_excel['url'] = df_excel['text'].apply(extract_url_if_exists)
                    df_excel['label'] = label
                    data_frames.append(df_excel)

                except Exception as e:
                    print(f"❌ Error reading Excel file {filename}: {e}")

            # --- Handle CSV files (.csv) ---
            elif filename.endswith('.csv'):
                try:
                    df_csv = pd.read_csv(file_path, encoding='utf-8')  # no usecols here

                    # detect the text column automatically
                    if excel_column in df_csv.columns:
                        text_col = excel_column
                    else:
                        text_col = df_csv.columns[0]

                    df_csv = df_csv.dropna(subset=[text_col])
                    if df_csv.empty:
                        print(f"⚠️ Skipping {filename}: no usable data.")
                        continue

                    df_csv = df_csv.rename(columns={text_col: 'text'})
                    df_csv['url'] = df_csv['text'].apply(extract_url_if_exists)
                    df_csv['label'] = label
                    data_frames.append(df_csv)

                except Exception as e:
                    print(f"❌ Error reading CSV file {filename}: {e}")

            # --- Handle .txt or no extension ---
            elif filename.endswith('.txt') or '.' not in filename:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines:
                            print(f"⚠️ Skipping empty text file: {filename}")
                            continue
                        df_text = pd.DataFrame({
                            'text': lines,
                            'url': [extract_url_if_exists(line) for line in lines],
                            'label': label
                        })
                        data_frames.append(df_text)
                except Exception as e:
                    print(f"❌ Error reading text file {filename}: {e}")

            # --- Handle unknown file types (fallback) ---
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines:
                            print(f"⚠️ Skipping empty unknown file: {filename}")
                            continue
                        df_unknown = pd.DataFrame({
                            'text': lines,
                            'url': [extract_url_if_exists(line) for line in lines],
                            'label': label
                        })
                        data_frames.append(df_unknown)
                        print(f"ℹ️ Read unknown file type as text: {filename}")
                except Exception as e:
                    print(f"❌ Error reading unknown file {filename}: {e}")

    # Combine all collected data into one DataFrame
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        print(f"⚠️ No data found in folder: {folder_path}")
        return pd.DataFrame(columns=['text', 'url', 'label'])

# === Load Data ===
df_good = process_mixed_folder(good_folder, label=0)
df_bad  = process_mixed_folder(bad_folder, label=1)
df = pd.concat([df_good, df_bad], ignore_index=True)

# Optional: Limit for faster training
# df = df.sample(n=10000, random_state=42)

# === Feature Engineering ===
url_features = df['url'].apply(extract_features_from_url)
X_url = pd.DataFrame(url_features.tolist())
X_url_sparse = csr_matrix(X_url.values)

df['urgent_phrase'] = df['text'].apply(detect_urgent_phrases)
df['suspicious_anchor'] = df['text'].apply(detect_suspicious_anchor_text)
df['sentiment_score'] = df['text'].apply(extract_sentiment_score)
X_text_custom = df[['urgent_phrase', 'suspicious_anchor', 'sentiment_score']]
X_text_sparse = csr_matrix(X_text_custom.values)

tfidf = TfidfVectorizer(max_features=300)
X_tfidf = tfidf.fit_transform(df['text'].astype(str))

# Combine all features
X_combined = hstack([X_url_sparse, X_text_sparse, X_tfidf])
y = df['label'].values

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42, stratify=y
)

# === Train RandomForestClassifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Evaluation ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Pie Chart ===
labels = ['Legitimate (0)', 'Phishing (1)']
pred_counts = pd.Series(y_pred).value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
plt.title("Prediction Distribution (Pie Chart)")
plt.tight_layout()
plt.show()

# === Bar Chart ===
actual_counts = pd.Series(y_test).value_counts().sort_index()
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
plt.bar(x + width/2, pred_counts, width, label='Predicted', color='orange')
plt.xlabel('Class')
plt.ylabel('Number of Emails')
plt.title('Actual vs Predicted Email Classification')
plt.xticks(ticks=x, labels=labels)
plt.legend()
plt.tight_layout()
plt.show()

# === SAVE TRAINED MODEL AND VECTORIZER ===
import joblib
import os

# Create a folder named 'model' if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained RandomForest model
model_path = "model/phishing_model.pkl"
joblib.dump(model, model_path)

# Save the trained TF-IDF vectorizer
vectorizer_path = "model/vectorizer.pkl"
joblib.dump(tfidf, vectorizer_path)

print(f"\n✅ Model and vectorizer saved successfully!")
print(f"   Model path: {model_path}")
print(f"   Vectorizer path: {vectorizer_path}")
