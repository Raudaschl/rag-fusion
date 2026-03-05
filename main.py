import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

_client = None


def get_client():
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        _client = OpenAI(api_key=api_key)
    return _client


def generate_queries_chatgpt(original_query, diverse=False):
    """Generate multiple search queries from a single input query using ChatGPT."""
    if diverse:
        messages = [
            {"role": "system", "content": "You are a search expert. Generate diverse search queries that explore different aspects of the user's question. Each query should target a different angle: use synonyms, vary specificity (broader/narrower), and consider related sub-topics. Avoid generating queries that are just minor rewordings of each other."},
            {"role": "user", "content": f"Generate 4 diverse search queries for: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    response = get_client().chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=messages
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


def vector_search(query, collection, n_results=5):
    """Search the ChromaDB collection and return results with real similarity scores."""
    results = collection.query(query_texts=[query], n_results=n_results)

    # ChromaDB returns distances (lower = more similar), convert to similarity scores
    scores = {}
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        scores[doc_id] = 1 / (1 + distance)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def reciprocal_rank_fusion(search_results_dict, k=60, verbose=True, query_weights=None):
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores = {}

    if verbose:
        print("Initial individual search result ranks:")
        for query, doc_scores in search_results_dict.items():
            print(f"For query '{query}': {doc_scores}")

    for query, doc_scores in search_results_dict.items():
        weight = query_weights.get(query, 1.0) if query_weights else 1.0
        for rank, (doc, _) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += weight * (1 / (rank + k))
            if verbose:
                print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
    if verbose:
        print("Final reranked results:", reranked_results)
    return reranked_results


def generate_output(reranked_results, queries, collection=None, original_query=None, use_llm=False):
    """Produce a final output from the reranked documents."""
    if not use_llm:
        return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"

    # Retrieve document text from collection
    doc_ids = list(reranked_results.keys())
    docs = collection.get(ids=doc_ids)
    doc_texts = "\n".join(f"- {text}" for text in docs["documents"])

    all_queries = "\n".join(f"- {q}" for q in queries)

    prompt = (
        f"Based on the following reranked documents and queries, generate a comprehensive answer.\n"
        f"Give more weight to the original query: \"{original_query}\"\n\n"
        f"Queries:\n{all_queries}\n\n"
        f"Reranked documents:\n{doc_texts}\n\n"
        f"Provide a well-structured response that synthesizes the information from these documents."
    )

    response = get_client().chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that synthesizes search results into a comprehensive answer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    original_query = "impact of climate change"
    generated_queries = generate_queries_chatgpt(original_query)

    collection = create_collection()

    all_results = {}
    # Search original query alongside generated ones (per article)
    all_queries = [original_query] + generated_queries
    for query in all_queries:
        search_results = vector_search(query, collection)
        all_results[query] = search_results

    reranked_results = reciprocal_rank_fusion(all_results)

    final_output = generate_output(
        reranked_results, all_queries,
        collection=collection, original_query=original_query, use_llm=True
    )
    print(final_output)
