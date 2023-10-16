import os
import openai
import random
import chromadb


# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")  # Alternative: Use environment variable
if openai.api_key is None:
    raise Exception("No OpenAI API key found. Please set it as an environment variable or in main.py")

# Initialize ChromeDB
chroma_client = chromadb.Client()

# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    )

    generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return generated_queries

# Function to perform a vector search, returning random scores
def vector_search(query, collection):
    # Perform the Chroma vector search for the given query
    chroma_results = collection.query(
        query_texts=[query],
        n_results=4  # Adjust the number of results as needed
    )

    # Extract the relevant document texts from Chroma results
    chroma_doc_texts = [result for result in chroma_results]

    # Assign random scores to each document from Chroma results
    scores = {doc_text: round(random.uniform(0.7, 0.9), 2) for doc_text in chroma_doc_texts}

    return scores

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results

# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"

# Chroma: Create Collection in this case "all_documents"
collection = chroma_client.create_collection(name="all_documents")

#################################
# Predefined set of documents (usually these would be from your search database)
## Replaced the "all_documents" section with the below chroma collection.
#################################

#Add some text documents to the collection
#Chroma will store your text, and handle tokenization, embedding, and indexing automatically.
collection.add(
    documents=[
        "Climate change and economic impact. Addressing climate change can be challenging, but by investing in sustainable technologies and policies, we can mitigate its economic impact and build a greener, more resilient future.",
        "Public health concerns due to climate change. Addressing public health concerns linked to climate change is feasible through proactive measures like improved healthcare infrastructure, climate adaptation, and public awareness campaigns.",
        "Climate change: A social perspective. Understanding climate change from a social perspective is possible. It requires fostering a sense of collective responsibility, promoting eco-friendly behaviors, and equitable climate policies for a sustainable future.",
        "Technological solutions to climate change. Implementing technological solutions to combat climate change is both possible and promising. Advancements in renewable energy, carbon capture, and sustainable agriculture offer hope for a greener future.",
        "Policy changes needed to combat climate change. Making policy changes to combat climate change is not impossible but rather necessary. Governments worldwide can enact laws to reduce emissions, promote renewable energy, and incentivize sustainable practices for a greener planet.",
        "Climate change and its impact on biodiversity. Addressing the impact of climate change on biodiversity is vital. Conservation efforts, habitat restoration, and global cooperation can help protect threatened species and ecosystems from further harm.",
        "Climate change: The science and models. Understanding climate change through science and models is entirely possible. Researchers use data, simulations, and predictive models to study and anticipate climate trends, providing valuable insights for mitigation and adaptation strategies.",
        "Global warming: A subset of climate change. Yes, global warming is a subset of climate change. It specifically refers to the long-term increase in Earth's average surface temperature, which is a key aspect of the broader phenomenon of climate change.",
        "How climate change affects daily weather. Climate change can influence daily weather patterns, making certain extreme events more frequent. While not impossible to explain, it's a complex process involving shifts in atmospheric circulation and temperature, which scientists study to better understand and predict weather changes.",
        "The history of climate change activism. The history of climate change activism is rich and inspiring. It began with grassroots movements and evolved into a global force for environmental awareness and policy change, demonstrating the power of collective action."
    ],
    metadatas=[
        {"source": "documents1"},
        {"source": "documents2"},
        {"source": "documents3"},
        {"source": "documents4"},
        {"source": "documents5"},
        {"source": "documents6"},
        {"source": "documents7"},
        {"source": "documents8"},
        {"source": "documents9"},
        {"source": "documents10"}
    ],
    ids=[
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5",
        "doc6",
        "doc7",
        "doc8",
        "doc9",
        "doc10"
    ]
)


# Main function
if __name__ == "__main__":
    # Define your Chroma collection and other necessary configurations here

    original_query = "impact of climate change"
    generated_queries = generate_queries_chatgpt(original_query)
    
    all_results = {}
    for query in generated_queries:
        search_results = vector_search(query, collection)  # Use the modified vector_search() function
        all_results[query] = search_results
    
    reranked_results = reciprocal_rank_fusion(all_results)
    
    final_output = generate_output(reranked_results, generated_queries)
    
    print(final_output)
