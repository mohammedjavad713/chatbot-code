
import os
import time
import numpy as np
import pymupdf
import faiss

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file")

client = genai.Client(api_key=api_key)

CHAT_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# 1. Extract text from PDF

def extract_pdf(pdf_path):

    doc = pymupdf.open(pdf_path)

    text = ""

    try:
        for page in doc:
            text += page.get_text() + "\n"
    finally:
        doc.close()

    return text


# 2. Split text into chunks

def split_text(text, chunk_size=1000, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 3. Create embeddings

def create_embeddings(texts, task_type):

    all_embeddings = []

    for start in range(0, len(texts), 50):

        batch = texts[start:start + 50]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )

        for embedding in response.embeddings:
            all_embeddings.append(embedding.values)

    return np.array(all_embeddings, dtype="float32")


# 4. Create vector database

def create_database(embeddings):

    if embeddings.size == 0:
        raise ValueError("No embeddings were generated.")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index


# 5. Search PDF

def search_pdf(question, index, chunks):

    question_embedding = create_embeddings(
        [question],
        "RETRIEVAL_QUERY"
    )

    faiss.normalize_L2(question_embedding)

    distances, indices = index.search(
        question_embedding, 3
    )

    results = []

    for i in indices[0]:

        if i != -1:
            results.append(chunks[i])

    return results


# 6. Generate answer with retry handling

def generate_answer(question, context):

    prompt = f"""
    You are a PDF question-answering chatbot.

    Answer ONLY using the information in the PDF context.

    If the answer is not available in the PDF,
    say that it is not available.

    PDF CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    # Retry only temporary API errors
    retryable_codes = [408, 429, 500, 502, 503, 504]

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt
            )

            if response.text:
                return response.text

            return "Sorry, Gemini returned an empty response."

        except errors.APIError as e:

            if e.code not in retryable_codes:
                print("\nGemini API error:", e)
                return "Sorry, an API error occurred. Please check your API configuration."

            print(f"\nGemini temporarily unavailable (HTTP {e.code}).")

            if attempt < 2:

                wait_time = 5 * (2 ** attempt)

                print(f"Retrying in {wait_time} seconds...")

                time.sleep(wait_time)

        except Exception as e:

            print("\nUnexpected error:", e)

            return "Something went wrong while generating the answer."

    return "Gemini is currently busy. Please try again after some time."


# 7. Main program

def main():

    print("\n==============================")
    print("       PDF AI CHATBOT")
    print("==============================")

    pdf_path = input("\nEnter PDF path: ").strip().strip('"')

    if not os.path.isfile(pdf_path):
        print("PDF not found!")
        return

    print("\nReading PDF...")

    try:
        text = extract_pdf(pdf_path)
    except Exception as e:
        print("Error reading PDF:", e)
        return

    if not text.strip():
        print("No readable text found in PDF.")
        return

    print("PDF loaded successfully!")

    print("\nSplitting text...")

    chunks = split_text(text)

    if not chunks:
        print("No text chunks created.")
        return

    print("Total chunks:", len(chunks))

    print("\nCreating embeddings...")

    try:
        embeddings = create_embeddings(
            chunks,
            "RETRIEVAL_DOCUMENT"
        )
    except Exception as e:
        print("Error creating embeddings:", e)
        return

    if len(embeddings) != len(chunks):
        print("Embedding count does not match chunk count.")
        return

    print("\nCreating vector database...")

    try:
        index = create_database(embeddings)
    except Exception as e:
        print("Error creating vector database:", e)
        return

    print("\nChatbot is ready!")

    print("Type 'exit' to close the chatbot.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Chatbot closed!")
            break

        if not question:
            continue

        try:

            relevant_chunks = search_pdf(
                question,
                index,
                chunks
            )

            if not relevant_chunks:
                print("\nBot: No relevant information found in the PDF.")
                continue

            context = "\n\n".join(relevant_chunks)

            answer = generate_answer(
                question,
                context
            )

            print("\nBot:", answer)

        except Exception as e:

            print("\nError processing question:", e)


if __name__ == "__main__":
    main()
