import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Plagiarism Checker", page_icon="🧠")
st.title("📄 Plagiarism Detector using NLP")

uploaded_files = st.file_uploader("📤 Upload two .txt files to compare:", type="txt", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    text1 = uploaded_files[0].read().decode("utf-8")
    text2 = uploaded_files[1].read().decode("utf-8")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    plagiarism_percent = round(similarity * 100, 2)

    st.subheader("📊 Result")
    st.metric(label="Similarity Score", value=f"{plagiarism_percent}%")

    if plagiarism_percent > 80:
        st.error("⚠️ High chance of plagiarism!")
    elif plagiarism_percent > 50:
        st.warning("⚠️ Moderate similarity, needs review.")
    else:
        st.success("✅ Low similarity, looks original!")

    with st.expander("📄 View File 1"):
        st.text(text1)

    with st.expander("📄 View File 2"):
        st.text(text2)
else:
    st.info("Please upload **exactly two .txt files** to check for plagiarism.")
