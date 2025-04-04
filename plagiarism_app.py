
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Read uploaded files
def read_file(file):
    if file is not None:
        return file.read().decode("utf-8")
    return ""

# Streamlit UI
st.set_page_config(page_title="Plagiarism Detector")
st.title("📄 Plagiarism Detector using NLP")
st.write("Upload two .txt files to compare them for plagiarism.")

file1 = st.file_uploader("Upload First File (.txt)", type=["txt"])
file2 = st.file_uploader("Upload Second File (.txt)", type=["txt"])

if st.button("Check Plagiarism"):
    if not file1 or not file2:
        st.warning("Please upload both text files.")
    else:
        text1 = read_file(file1)
        text2 = read_file(file2)

        if text1.strip() == "" or text2.strip() == "":
            st.warning("One or both files are empty.")
        else:
            texts = [preprocess(text1), preprocess(text2)]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = similarity[0][0] * 100

            st.success(f"🟢 Plagiarism Score: **{score:.2f}%**")

            if score > 80:
                st.error("⚠️ High similarity detected. Integrity: LOW ❌")
            elif score > 50:
                st.warning("🟡 Moderate similarity. Integrity: MEDIUM ⚠️")
            else:
                st.info("🟢 Low similarity. Integrity: HIGH ✅")
