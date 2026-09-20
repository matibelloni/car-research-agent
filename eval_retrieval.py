from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# bigger model
# model = SentenceTransformer("intfloat/multilingual-e5-large")

documents = [
    "The 1.6 THP engine has known problems in the timing chain.",
    "The 2015 Volkswagen Golf uses 7.3 liters per 100 km in the city.",
    "To change the oil, you need to loosen the oil pan plug.",
    "Disc brakes need to be checked every 30,000 kilometers.",
    "The 2016 Ford Focus has a 375 liter trunk capacity.",
    # competition
    "The 2018 Volkswagen Golf uses 6.8 liters per 100 km in the city.",
    "The 2016 Ford Focus uses 8.1 liters per 100 km in the city.",
    "The 2015 Volkswagen Polo uses 6.2 liters per 100 km in the city.",
    "The 2.0 TDI engine has problems in the injection system.",
    "The 2015 Golf has a 380 liter trunk capacity.",
]

# The dataset: question → index of the correct document
cases = [
    ("how much does the 2015 Golf use?", 1),
    ("how much does the 2018 Golf use?", 5),
    ("how much does the Focus use?", 6),
    ("what trunk does the Golf have?", 9),
    ("what trunk does the Focus have?", 4),
    ("diesel engine problems", 8),
    ("is the Polo economical?", 7),
]

doc_vectors = model.encode(documents)

successes = 0
for question, expected_index in cases:
    question_vector = model.encode(question)
    relations = cos_sim(question_vector, doc_vectors)[0]
    winner_index = relations.argmax().item()

    ok = winner_index == expected_index
    successes += ok

    print(f"{'✅' if ok else '❌'}")
    if not ok:
        print(f"Question: {question}")
        print(f"Document expected: {documents[expected_index]}")
        print(f"Document received: {documents[winner_index]}")

print(f"Precision: successes/len(cases): {successes/len(cases):.2f}")
