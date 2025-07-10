# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import nltk
import spacy
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from rapidfuzz import process
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import webbrowser
import joblib

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
tqdm.pandas()

nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df_train = pd.read_csv(r"C:\Users\L15\PycharmProjects\NLP_Project\data\bbc-text.csv")
print("Data loaded successfully")

def nameEntityRec(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df_train['Entities'] = df_train['text'].progress_apply(nameEntityRec)

def textCleaning(text, preserved_entities):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens += [ent[0] for ent in preserved_entities]
    return ' '.join(tokens)

df_train['Clean_Text'] = df_train.progress_apply(
    lambda row: textCleaning(row['text'], row['Entities']), axis=1)

def textNormalizing(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized)

df_train['Lemmatized_Text'] = df_train['Clean_Text'].apply(textNormalizing)

allWords = ' '.join(df_train['text']).split()
wordFrequency = Counter(allWords)
vocabulary = set([word.lower() for word, frequency in wordFrequency.items() if frequency > 2])

def correctTypo(word, vocab):
    best_match = process.extractOne(word, vocab, score_cutoff=80)
    return best_match[0] if best_match else word

def typoCorrection(text, vocab, preserved_entities):
    tokens = text.split()
    corrected = [
        token if token in preserved_entities or token in vocab or len(token) <= 2 else correctTypo(token, vocab)
        for token in tokens
    ]
    return ' '.join(corrected)

df_train['Corrected_Text'] = df_train.progress_apply(
    lambda row: typoCorrection(row['Lemmatized_Text'], vocabulary, [ent[0] for ent in row['Entities']]), axis=1)

example_index = 51
original_text = df_train.loc[example_index, 'text']
cleaned_text = df_train.loc[example_index, 'Corrected_Text']
print("\nBefore Preprocessing")
print("Original Text:")
print(original_text)
print("\nAfter Preprocessing:")
print(cleaned_text)

X = df_train['Corrected_Text']
y = df_train['category']

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=3, max_features=8000, sublinear_tf=True, stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)


models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {'clf__C': [0.1, 1, 10]}),
    "DecisionTree": (DecisionTreeClassifier(random_state=42), {'clf__max_depth': [10, 20]}),
    "SVM": (LinearSVC(max_iter=10000), {'clf__C': [0.1, 1]}),
    "MLP": (MLPClassifier(max_iter=500), {'clf__hidden_layer_sizes': [(100,)]}),
    "NaiveBayes": (MultinomialNB(), {}),
    "RandomForest": (RandomForestClassifier(), {'clf__n_estimators': [100]}),
    "VotingEnsemble": (VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', LinearSVC(max_iter=10000)),
        ('rf', RandomForestClassifier())
    ], voting='hard'), {})
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, (model, param_grid) in models.items():
    print(f"Training {name}")
    pipe = Pipeline([('clf', model)])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, Y_train)

    best_model = grid.best_estimator_
    Y_pred = best_model.predict(X_test)

    acc_train = accuracy_score(Y_train, best_model.predict(X_train))
    acc_test = accuracy_score(Y_test, Y_pred)
    cv_scores = cross_val_score(best_model, X_train, Y_train, cv=cv, scoring='f1_macro')

    results[name] = {
        'model': best_model,
        'train_acc': acc_train,
        'test_acc': acc_test,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'params': grid.best_params_
    }

    print(classification_report(Y_test, Y_pred))


best_model_name = max(results, key=lambda k: results[k]['test_acc'])

summary_metrics = []

for name, result in results.items():
    y_pred = result['model'].predict(X_test)
    report = classification_report(Y_test, y_pred, output_dict=True)
    acc = result['test_acc']
    macro_f1 = report['macro avg']['f1-score']

    summary_metrics.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Macro F1-Score": round(macro_f1, 4)
    })

df_summary = pd.DataFrame(summary_metrics).sort_values(by="Accuracy", ascending=False)
print("\n===== Model Comparison Summary =====\n")
print(df_summary.to_string(index=False))
print(f"\nBest model: {best_model_name} with test accuracy: {results[best_model_name]['test_acc']:.4f}")

best_model = results[best_model_name]['model']

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(Y_test, y_pred_best)
precision_best = precision_score(Y_test, y_pred_best, average='macro')
recall_best = recall_score(Y_test, y_pred_best, average='macro')
f1_best = f1_score(Y_test, y_pred_best, average='macro')

print(f"\n{'='*5}{best_model_name}=====")
print(f"Accuracy: {accuracy_best}")
print(f"Precision: {precision_best}")
print(f"Recall: {recall_best}")
print(f"F1-score: {f1_best}")

print("\nClassification Report:")
print(classification_report(Y_test, y_pred_best))

joblib.dump(results[best_model_name]['model'], "news_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test)
labels = sorted(y.unique())

cm = confusion_matrix(Y_test, y_pred_best, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Best Model ({best_model_name})")
plt.tight_layout()
plt.show()


model_names = list(results.keys())
accuracies = [results[name]['test_acc'] for name in model_names]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color='blue')
plt.ylabel("Test Accuracy")
plt.title("Model Comparison")
plt.ylim(0.8, 1.0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

#ovfit<--------
train_scores = [results[m]['train_acc'] for m in model_names]
test_scores = [results[m]['test_acc'] for m in model_names]

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(model_names))

plt.bar(x, train_scores, width=bar_width, label='Train Accuracy')
plt.bar(x + bar_width, test_scores, width=bar_width, label='Test Accuracy')
plt.xticks(x + bar_width / 2, model_names)
plt.ylim(0.6, 1.0)
plt.title('Overfitting Check')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

#lcrv------------------
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, Y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1_macro'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
plt.fill_between(train_sizes,
                 np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
plt.fill_between(train_sizes,
                 np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                 np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title(f'Learning Curves - {best_model_name}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# LIME Analysis for Best Model (if supports predict_proba)
try:
    print(f"\n[INFO] Running LIME analysis for best model: {best_model_name}")
    if hasattr(best_model.named_steps['clf'], 'predict_proba'):
        clf = best_model.named_steps['clf']
        class_names = clf.classes_
        lime_pipeline = make_pipeline(vectorizer, clf)
        text_instance = df_train['Corrected_Text'].iloc[51]
        predicted_index = np.argmax(lime_pipeline.predict_proba([text_instance]))
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            text_instance,
            lime_pipeline.predict_proba,
            num_features=10,
            labels=[predicted_index]
        )
        html = exp.as_html(labels=[predicted_index])
        html_file = "lime_explanation.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[INFO] LIME explanation saved to {html_file}")
        webbrowser.open(html_file)
    else:
        print(f"[WARNING] LIME is not supported because {best_model_name} has no predict_proba.")
except ImportError:
    print("[ERROR] LIME not installed. Run `pip install lime` to use it.")



