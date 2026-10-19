import os
import pandas as pd
import numpy as np

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PDF DOCUMENT FOLDER
# =========================================================

DOCUMENT_FOLDER = "documents"


# =========================================================
# 2. READ PDF DOCUMENTS
# =========================================================

documents = []
document_names = []

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):

    if filename.lower().endswith(".pdf"):

        file_path = os.path.join(DOCUMENT_FOLDER, filename)

        print(f"Reading: {filename}")

        reader = PdfReader(file_path)

        text = ""

        # Extract text from every page
        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

        documents.append(text)
        document_names.append(filename)


# =========================================================
# 3. DISPLAY DOCUMENT INFORMATION
# =========================================================

print("\n========================================")
print("DOCUMENT INFORMATION")
print("========================================")

print("Number of documents:", len(documents))

for i, name in enumerate(document_names):
    print(f"{i + 1}. {name}")
    print(f"   Characters: {len(documents[i])}")


# =========================================================
# 4. CHECK WHETHER DOCUMENTS WERE FOUND
# =========================================================

if len(documents) < 2:

    print("\nError: At least 2 PDF documents are required.")

    exit()


# =========================================================
# 5. CREATE TF-IDF VECTORS
# =========================================================

print("\n========================================")
print("CREATING TF-IDF VECTORS")
print("========================================")

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_df=0.95,
    min_df=1
)

tfidf_matrix = vectorizer.fit_transform(documents)


print("TF-IDF matrix shape:", tfidf_matrix.shape)

print(
    "Number of unique terms:",
    len(vectorizer.get_feature_names_out())
)


# =========================================================
# 6. CALCULATE COSINE SIMILARITY
# =========================================================

print("\n========================================")
print("CALCULATING COSINE SIMILARITY")
print("========================================")

similarity_matrix = cosine_similarity(tfidf_matrix)


# =========================================================
# 7. CREATE SIMILARITY DATAFRAME
# =========================================================

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=document_names,
    columns=document_names
)


# Round scores for easier reading
similarity_df = similarity_df.round(4)


# =========================================================
# 8. DISPLAY SIMILARITY MATRIX
# =========================================================

print("\n========================================")
print("COSINE SIMILARITY MATRIX")
print("========================================")

print(similarity_df)


# =========================================================
# 9. SAVE SIMILARITY MATRIX
# =========================================================

similarity_df.to_csv("similarity_matrix.csv")

print("\nSimilarity matrix saved to:")
print("similarity_matrix.csv")


# =========================================================
# 10. FIND SIMILARITY BETWEEN DOCUMENT PAIRS
# =========================================================

pairs = []

for i in range(len(document_names)):

    for j in range(i + 1, len(document_names)):

        score = similarity_matrix[i][j]

        pairs.append({
            "Document 1": document_names[i],
            "Document 2": document_names[j],
            "Similarity Score": round(score, 4)
        })


# =========================================================
# 11. SORT DOCUMENT PAIRS
# =========================================================

pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    by="Similarity Score",
    ascending=False
)


# =========================================================
# 12. DISPLAY SIMILARITY RANKING
# =========================================================

print("\n========================================")
print("DOCUMENT SIMILARITY RANKING")
print("========================================")

print(pairs_df.to_string(index=False))


# =========================================================
# 13. SAVE PAIR RESULTS
# =========================================================

pairs_df.to_csv(
    "document_similarity_results.csv",
    index=False
)

print("\nDetailed results saved to:")
print("document_similarity_results.csv")


# =========================================================
# 14. MOST SIMILAR DOCUMENTS
# =========================================================

highest = pairs_df.iloc[0]

print("\n========================================")
print("MOST SIMILAR DOCUMENTS")
print("========================================")

print("Document 1:", highest["Document 1"])
print("Document 2:", highest["Document 2"])
print("Similarity Score:", highest["Similarity Score"])


# =========================================================
# 15. LEAST SIMILAR DOCUMENTS
# =========================================================

lowest = pairs_df.iloc[-1]

print("\n========================================")
print("LEAST SIMILAR DOCUMENTS")
print("========================================")

print("Document 1:", lowest["Document 1"])
print("Document 2:", lowest["Document 2"])
print("Similarity Score:", lowest["Similarity Score"])


# =========================================================
# 16. TOP 5 TERMS FOR EACH DOCUMENT
# =========================================================

print("\n========================================")
print("TOP TERMS IN EACH DOCUMENT")
print("========================================")

feature_names = vectorizer.get_feature_names_out()

for i, document in enumerate(document_names):

    row = tfidf_matrix[i].toarray().flatten()

    top_indices = row.argsort()[-5:][::-1]

    top_terms = [
        feature_names[index]
        for index in top_indices
        if row[index] > 0
    ]

    print(f"\n{document}")
    print("Top terms:", ", ".join(top_terms))


# =========================================================
# 17. PROGRAM COMPLETED
# =========================================================

print("\n========================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("========================================")