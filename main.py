import os
import openai
import random
import chromadb
import os

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now you can safely import tokenizers and the rest of your code
#from transformers import AutoTokenizer

# OpenAI API Initialization
# Retrieve the OpenAI API key from the environment variables.
# If the API key is not found, an exception is raised to alert the user.
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("No OpenAI API key found. Please set it as an environment variable or in main.py")

# Initialize the ChromaDB client
# This client will be used to interact with the ChromaDB database.
chroma_client = chromadb.Client()

# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):
    """
    Uses OpenAI's ChatGPT model to generate multiple related search queries from an original query.
    
    Parameters:
    original_query (str): The initial query from which related queries will be generated.
    
    Returns:
    list: A list of generated queries based on the original query.
    """
    # Call to OpenAI's API, specifying the ChatGPT model and the conversation flow to generate queries.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    )

    # Parse the response to extract the generated queries and return them as a list.
    generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return generated_queries

# Function to perform a vector search within the ChromaDB collection
def vector_search(query, collection):
    """
    Performs a vector search for a given query in the specified ChromaDB collection.
    Assigns random relevance scores to the search results.
    
    Parameters:
    query (str): The search query.
    collection (chromadb.Collection): The ChromaDB collection to be searched.
    
    Returns:
    tuple: A tuple containing the search results, score values, document IDs, metadata, and document texts.
    """
    # Query the collection using the provided search term.
    chroma_results = collection.query(
        query_texts=[query],  
        n_results=4
    )
    
    # Extract the document IDs from the search results.
    document_ids = chroma_results['ids'][0]

    # Initialize lists to store document texts and metadata.
    chroma_doc_texts = []
    chroma_doc_metadata = []
    metadatas = []
    documents = []  

    # Retrieve and store the documents and metadata for each document ID.
    for doc_id in document_ids:
        doc_info = collection.get(ids=[doc_id])

        if doc_info:
            # Append the document text and metadata to their respective lists if found.
            chroma_doc_texts.append(doc_info['documents'][0])
            chroma_doc_metadata.append(doc_info['metadatas'][0])
            metadata = doc_info['metadatas'][0] 
            metadatas.append(metadata)
            document = doc_info['documents'][0]
            documents.append(document)
        else:
            # If the document is not found, append placeholders.
            chroma_doc_texts.append("Document not found")
            chroma_doc_metadata.append({})

    # Extract the document titles from the metadata.
    document_titles = [metadata.get("title", "Unknown Title") for metadata in chroma_doc_metadata]

    # Create a dictionary to hold the random scores assigned to each document.
    scores_dict = {
        doc_id: round(random.uniform(0.7, 0.9), 2) for doc_id in document_ids
    }
    
    # Combine the document texts, titles, and scores into a single dictionary.
    scores = {
        "text": chroma_doc_texts,
        "titles": document_titles, 
        "scores": scores_dict
    }

    # Convert the scores dictionary into a list of values.
    score_values = list(scores_dict.values())

    return scores, score_values, document_ids, metadatas, documents

# Implementation of the Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(all_results, document_ids, k=60):
    """
    Applies the Reciprocal Rank Fusion (RRF) algorithm to combine multiple search result rankings.
    
    Parameters:
    all_results (dict): A dictionary of search results keyed by query, each containing score values.
    document_ids (list): A list of document IDs.
    k (int, optional): The RRF parameter, used in the score calculation. Default is 60.
    
    Returns:
    dict: A dictionary of reranked results with document IDs as keys and fused scores as values.
    """
    fused_scores = {}

    # Iterate through each set of results and apply the Reciprocal Rank Fusion (RRF) formula to calculate fused scores.
    for query, result in all_results.items():
        score_values = result["score_values"]

        for rank, score in enumerate(sorted(score_values)):
            doc_id = document_ids[rank]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            previous_score = fused_scores[doc_id]
            fused_scores[doc_id] += 1 / (rank + k)

    # Sort the fused scores in descending order to rerank the results.
    reranked_results = {doc_id: score for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    
    print("Final reranked results:", reranked_results)

    return reranked_results

# Function to generate a final output that includes the most relevant document
def generate_output(reranked_results, queries, metadatas, documents):
    """
    Generates a detailed output from the reranked results, including the top document's title, score, and summary.
    
    Parameters:
    reranked_results (dict): The reranked results from the RRF algorithm.
    queries (list): The list of original queries.
    metadatas (list): The list of metadata for documents.
    documents (list): The list of document texts.
    
    Returns:
    str: A string containing a formatted response with the search results.
    """
    # Find the top-ranked document and its score.
    top_document_id, top_score = next(iter(reranked_results.items()))

    # Find the index of the top document in the list of document IDs.
    top_doc_index = document_ids.index(top_document_id)
    # Retrieve the metadata for the top-ranked document.
    top_doc_metadata = metadatas[top_doc_index]
    # Get the title from the metadata, defaulting to "Unknown Title" if not present.
    top_document_title = top_doc_metadata.get("title", "Unknown Title")

    # Call the function to generate a summary for the top document.
    document_summary = generate_summary(top_document_id, metadatas, documents)

    # Construct the final response string with the document title, score, summary, and original queries.
    response = f"Here is the most relevant information regarding '{queries[0]}':\n\n"
    response += f"Document Title: {top_document_title}\n"
    response += f"Relevance Score: {top_score}\n\n"
    response += f"Summary of the document:\n{document_summary}\n\n"

    # Append the original queries to the response.
    response += "Original Queries:\n"
    for query in queries:
        response += f"- {query}\n"

    # Add a concluding statement about the technologies used in this script.
    response += "\nBy layering these technologies and techniques, RAG Fusion offers a powerful, nuanced approach to text generation. It leverages the best of search technology and generative AI to produce high-quality, reliable outputs."

    return response

# Create a new collection within ChromaDB named "all_documents".
collection = chroma_client.create_collection(name="all_documents")

# Add a predefined set of documents related to climate change to the collection, along with their metadata.
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
        {"source": "doc1", "title": "Climate change and economic impact"},
        {"source": "doc2", "title": "Public health concerns due to climate change"},
        {"source": "doc3", "title": "Climate change: A social perspective"},
        {"source": "doc4", "title": "Technological solutions to climate change"},
        {"source": "doc5", "title": "Policy changes needed to combat climate change"},
        {"source": "doc6", "title": "Climate change and its impact on biodiversity"},
        {"source": "doc7", "title": "Climate change: The science and models"},
        {"source": "doc8", "title": "Global warming: A subset of climate change"},
        {"source": "doc9", "title": "How climate change affects daily weather"},
        {"source": "doc10", "title": "The history of climate change activism"}
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

# Function to generate a summary for a given document
def generate_summary(document_id, metadatas, documents):
    """
    Generates a summary for a specified document using OpenAI's GPT-3 model.
    
    Parameters:
    document_id (str): The ID of the document to summarize.
    metadatas (list): The list of metadata for documents.
    documents (list): The list of document texts.
    
    Returns:
    str: The generated summary for the document.
    """
    # Find the index of the document using its ID.
    doc_index = document_ids.index(document_id)
    # Construct the prompt for the summary generation.
    prompt = f"Summarize the following document:\n{documents[doc_index]}\n\nSummary:"

    # Generate the summary using OpenAI's model.
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,  # Adjust the length of the summary as needed
        stop=None,      # Allow the model to generate the summary freely
        temperature=0.7  # Adjust the temperature for creativity
    )

    # Extract and return the summary from the response.
    summary = response.choices[0].text.strip()
    return summary

# Main function to be executed when the script is run
if __name__ == "__main__":
    # Example query to kickstart the process.
    original_query = "impact of climate change"
    generated_queries = generate_queries_chatgpt(original_query)
    
    # Perform vector searches and store results.
    all_results = {}
    for query in generated_queries:
        scores, score_values, document_ids, metadatas, documents = vector_search(query, collection)
        all_results[query] = {
            "scores": scores,
            "score_values": score_values
        }
    
    # Apply reciprocal rank fusion to the collected results.
    reranked_results = reciprocal_rank_fusion(all_results, document_ids)
    print("Final reranked results <bottom>:", reranked_results)
    
    # Generate the final output and print it.
    final_output = generate_output(reranked_results, generated_queries, metadatas, documents)
    print(final_output)
