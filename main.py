import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise Exception("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")


def generate_queries_chatgpt(original_query):
    """Generate multiple search queries from a single input query using ChatGPT."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    )
    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries


def create_collection():
    """Create a ChromaDB collection and populate it with climate change documents."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="all_documents")

    documents = [
        "Climate change and economic impact.",
        "Public health concerns due to climate change.",
        "Climate change: A social perspective.",
        "Technological solutions to climate change.",
        "Policy changes needed to combat climate change.",
        "Climate change and its impact on biodiversity.",
        "Climate change: The science and models.",
        "Global warming: A subset of climate change.",
        "How climate change affects daily weather.",
        "The history of climate change activism."
    ]
    doc_ids = [f"doc{i+1}" for i in range(len(documents))]

    collection.add(documents=documents, ids=doc_ids)
    return collection


def vector_search(query, collection):
    """Search the ChromaDB collection and return results with real similarity scores."""
    results = collection.query(query_texts=[query], n_results=5)

    # ChromaDB returns distances (lower = more similar), convert to similarity scores
    scores = {}
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        scores[doc_id] = 1 / (1 + distance)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def reciprocal_rank_fusion(search_results_dict, k=60):
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores = {}

    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")

    for query, doc_scores in search_results_dict.items():
        for rank, (doc, _) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
    print("Final reranked results:", reranked_results)
    return reranked_results


def generate_output(reranked_results, queries):
    """Produce a final output from the reranked documents."""
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"


if __name__ == "__main__":
    original_query = "impact of climate change"
    generated_queries = generate_queries_chatgpt(original_query)

    collection = create_collection()

    all_results = {}
    for query in generated_queries:
        search_results = vector_search(query, collection)
        all_results[query] = search_results

    reranked_results = reciprocal_rank_fusion(all_results)

    final_output = generate_output(reranked_results, generated_queries)
    print(final_output)
