import os
import openai
import random
import chromadb
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',filename='/tmp/ragfusion.log', filemode='w')


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

  # Perform vector search
  chroma_results = collection.query(
    query_texts=[query],  
    n_results=4
  )
  logging.info(f"chroma_results: {chroma_results}")
  
  # Extract document IDs (flattened)
  document_ids = chroma_results['ids'][0]  
  logging.info(f"document_ids: {document_ids}")

  # Retrieve documents
  chroma_doc_texts = []
  chroma_doc_metadata = []
  # Initialize metadatas as a list
  metadatas = []
  documents = []  

  for doc_id in document_ids:
    doc_info = collection.get(ids=[doc_id])
    logging.info(f"doc_info 56: {doc_info}")

    if doc_info:

      chroma_doc_texts.append(doc_info['documents'][0])
      chroma_doc_metadata.append(doc_info['metadatas'][0])
      metadata = doc_info['metadatas'][0] 
      metadatas.append(metadata)
      document = doc_info['documents'][0]
      documents.append(document)

    else:
      chroma_doc_texts.append("Document not found")
      chroma_doc_metadata.append({})

  document_ids = chroma_results['ids'][0]
  logging.info(f"document_ids: {document_ids}")
  # Extract titles 
  document_titles = [metadata.get("title", "Unknown Title") for metadata in chroma_doc_metadata]
  logging.info(f"document_titles: {document_titles}")

  # Generate scores dict
  scores_dict = {
    doc_id: round(random.uniform(0.7, 0.9), 2) for doc_id in document_ids
  }
  # Create scores
  scores = {
    "text": chroma_doc_texts,
    "titles": document_titles, 
    "scores": scores_dict
  }
  logging.info(f"################## doc_info: {doc_info}")
  # Extract just the score values into a list 
  score_values = list(scores_dict.values())
  # Also build documents list
  #documents = [doc_info['documents'][0][0] for doc_info in collection.get(document_ids)]
  logging.info(f"################## documents: {documents}")

  logging.info(f"scores: {scores}")

  # Log results
  logging.info(f"scores: {scores}")
  logging.info(f"document_ids: {document_ids}")

  return scores, score_values, document_ids, metadatas, documents

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(all_results, document_ids, k=60):

  fused_scores = {}

  #for query, doc_scores in search_results_dict.items():
  for query, result in all_results.items():
    logging.info(f"For query '{query}': {result}")
    score_values = result["score_values"]

    for rank, score in enumerate(sorted(score_values)):    
      doc_id = document_ids[rank]      
      if doc_id not in fused_scores:
        fused_scores[doc_id] = 0        
      previous_score = fused_scores[doc_id]      
      fused_scores[doc_id] += 1 / (rank + k)      
      logging.info(f"Updating score for {doc_id} from {previous_score} to {fused_scores[doc_id]} based on rank {rank} in query '{query}'")

  reranked_results = {doc_id: score for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
  
  print("Final reranked results:", reranked_results)

  return reranked_results

def generate_output(reranked_results, queries, metadatas, documents):
    # Extract the top-ranked document ID and its score
    top_document_id, top_score = next(iter(reranked_results.items()))

    # Fetch the actual document title associated with the document ID 
    logging.info(f"metadatas: {metadatas}")
    top_doc_index = document_ids.index(top_document_id)
    logging.info(f"top_doc_index: {top_doc_index}")
    top_doc_metadata = metadatas[top_doc_index]
    logging.info(f"top_doc_metadata: {top_doc_metadata}")
    top_document_title = top_doc_metadata.get("title", "Unknown Title")

    # Generate_summary() generates a summary for the top document
    document_summary = generate_summary(top_document_id, metadatas, documents)  # Implement this function

    # Generate a response with the correct document title, relevance score, summary, and original queries
    response = f"Here is the most relevant information regarding '{queries[0]}':\n\n"
    response += f"Document Title: {top_document_title}\n"
    response += f"Relevance Score: {top_score}\n\n"
    response += f"Summary of the document:\n{document_summary}\n\n"

    response += "Original Queries:\n"
    for query in queries:
        response += f"- {query}\n"

    response += "\nBy layering these technologies and techniques, RAG Fusion offers a powerful, nuanced approach to text generation. It leverages the best of search technology and generative AI to produce high-quality, reliable outputs."

    return response

# Chroma: Create Collection in this case "all_documents"
collection = chroma_client.create_collection(name="all_documents")

#################################
# Predefined set of documents (usually these would be from your search database)
## Replaced the "all_documents" section with the below chroma collection.
#################################

#Added some text documents to the collection
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

def generate_summary(document_id, metadatas, documents):
    # Lookup document text
    doc_index = document_ids.index(document_id)
    logging.info(f"meta_data: {metadatas}")
    logging.info(f"documents: {documents}")

    # Define the prompt for generating the summary
    prompt = f"Summarize the following document:\n{documents}\n\nSummary:"

    # Use the GPT-3 model to generate the summary
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,  # Adjust the length of the summary as needed
        stop=None,      # Allow the model to generate the summary freely
        temperature=0.7  # Adjust the temperature for creativity
    )

    # Extract the generated summary from the response
    summary = response.choices[0].text.strip()    
    return summary

# Main function
if __name__ == "__main__":
    # Define your Chroma collection and other necessary configurations here
    original_query = "impact of climate change"
    generated_queries = generate_queries_chatgpt(original_query)
    
    all_results = {}
    for query in generated_queries:

      scores, score_values, document_ids, metadatas, documents= vector_search(query, collection)
      
      all_results[query] = {
        "scores": scores,
        "score_values": score_values
      }
    
    logging.info(f"all_results: {all_results}")
    reranked_results = reciprocal_rank_fusion(all_results, document_ids)
    print("Final reranked results <bottom>:", reranked_results)  
    
    final_output = generate_output(reranked_results, generated_queries, metadatas, documents)
    
    print(final_output)