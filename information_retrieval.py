# ==========================================================
# BASIC INFORMATION RETRIEVAL SYSTEM
# ==========================================================

import re


# ==========================================================
# 1. DOCUMENT COLLECTION
# ==========================================================

documents = {
    1: "Artificial intelligence is changing modern technology.",
    2: "Machine learning is a branch of artificial intelligence.",
    3: "Deep learning uses neural networks to solve complex problems.",
    4: "Information retrieval helps users find relevant information.",
    5: "Search engines use information retrieval techniques.",
    6: "Natural language processing helps computers understand human language.",
    7: "Machine learning and deep learning are important areas of AI.",
    8: "Artificial intelligence is widely used in healthcare and education."
}


# ==========================================================
# 2. ADD DOCUMENT
# ==========================================================

def add_document(doc_id, text):
    documents[doc_id] = text
    print(f"Document {doc_id} added successfully!")


# ==========================================================
# 3. TOKENIZATION
# ==========================================================

def tokenize(text):
    # Convert text to lowercase
    text = text.lower()

    # Extract words only
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    return words


# ==========================================================
# 4. CREATE DICTIONARY
# ==========================================================

def create_dictionary():
    dictionary = set()

    for text in documents.values():
        words = tokenize(text)

        for word in words:
            dictionary.add(word)

    return sorted(dictionary)


# ==========================================================
# 5. CREATE INVERTED INDEX
# ==========================================================

def create_inverted_index():
    inverted_index = {}

    for doc_id, text in documents.items():

        words = tokenize(text)

        for word in words:

            if word not in inverted_index:
                inverted_index[word] = set()

            inverted_index[word].add(doc_id)

    return inverted_index


# ==========================================================
# 6. BOOLEAN AND
# ==========================================================

def boolean_and(word1, word2, index):

    result1 = index.get(word1.lower(), set())
    result2 = index.get(word2.lower(), set())

    return result1 & result2


# ==========================================================
# 7. BOOLEAN OR
# ==========================================================

def boolean_or(word1, word2, index):

    result1 = index.get(word1.lower(), set())
    result2 = index.get(word2.lower(), set())

    return result1 | result2


# ==========================================================
# 8. BOOLEAN NOT
# ==========================================================

def boolean_not(word, index):

    word_documents = index.get(word.lower(), set())

    all_documents = set(documents.keys())

    return all_documents - word_documents


# ==========================================================
# 9. DISPLAY DOCUMENTS
# ==========================================================

print("\n")
print("=" * 60)
print("DOCUMENT COLLECTION")
print("=" * 60)

for doc_id, text in documents.items():
    print(f"Document {doc_id}: {text}")


# ==========================================================
# 10. CREATE AND DISPLAY DICTIONARY
# ==========================================================

dictionary = create_dictionary()

print("\n")
print("=" * 60)
print("DICTIONARY")
print("=" * 60)

print("Number of unique terms:", len(dictionary))

print("\nUnique Terms:")

for word in dictionary:
    print(word)


# ==========================================================
# 11. CREATE AND DISPLAY INVERTED INDEX
# ==========================================================

inverted_index = create_inverted_index()

print("\n")
print("=" * 60)
print("INVERTED INDEX")
print("=" * 60)

for word in sorted(inverted_index):
    print(f"{word:25} -> {sorted(inverted_index[word])}")


# ==========================================================
# 12. BOOLEAN RETRIEVAL
# ==========================================================

print("\n")
print("=" * 60)
print("BOOLEAN RETRIEVAL")
print("=" * 60)


# AND QUERY
result = boolean_and("machine", "learning", inverted_index)

print("\nQuery: machine AND learning")
print("Retrieved Documents:", sorted(result))


# OR QUERY
result = boolean_or("artificial", "search", inverted_index)

print("\nQuery: artificial OR search")
print("Retrieved Documents:", sorted(result))


# NOT QUERY
result = boolean_not("learning", inverted_index)

print("\nQuery: NOT learning")
print("Retrieved Documents:", sorted(result))


# SECOND AND QUERY
result = boolean_and("information", "retrieval", inverted_index)

print("\nQuery: information AND retrieval")
print("Retrieved Documents:", sorted(result))


# SECOND OR QUERY
result = boolean_or("deep", "neural", inverted_index)

print("\nQuery: deep OR neural")
print("Retrieved Documents:", sorted(result))


# ==========================================================
# 13. ADD A NEW DOCUMENT
# ==========================================================

print("\n")
print("=" * 60)
print("ADDING NEW DOCUMENT")
print("=" * 60)

add_document(
    9,
    "Artificial intelligence and machine learning support modern applications."
)


# ==========================================================
# 14. UPDATE INVERTED INDEX
# ==========================================================

inverted_index = create_inverted_index()


print("\nUpdated Inverted Index:")
print("-" * 60)

for word in sorted(inverted_index):
    print(f"{word:25} -> {sorted(inverted_index[word])}")


# ==========================================================
# 15. SEARCH AFTER ADDING NEW DOCUMENT
# ==========================================================

result = boolean_and("artificial", "machine", inverted_index)

print("\nQuery: artificial AND machine")
print("Retrieved Documents:", sorted(result))


# ==========================================================
# END
# ==========================================================

print("\n")
print("=" * 60)
print("INFORMATION RETRIEVAL SYSTEM COMPLETED")
print("=" * 60)