import chromadb
from chromadb.config import Settings
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

class RetentionSearchIndex:
    def __init__(self, host=None, port=None):
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = port or os.getenv("CHROMA_PORT", "8000")
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        self.collection_name = "retention_policies"

    def initialize_index(self, data_path="data/retention_knowledge.json"):
        """Creates the collection and populates it with data."""
        with open(data_path, "r") as f:
            documents = json.load(f)
        
        # Reset/Create collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        collection = self.client.create_collection(name=self.collection_name)
        
        # Prepare data for Chroma
        ids = [str(doc['id']) for doc in documents]
        texts = [f"{doc['category']}: {doc['description']} Benefit: {doc['benefit']}" for doc in documents]
        metadatas = [
            {
                "category": doc["category"], 
                "benefit": doc["benefit"], 
                "condition": doc["condition"],
                "rule_id": str(doc["id"])
            } 
            for doc in documents
        ]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Collection '{self.collection_name}' initialized with {len(documents)} documents.")

    def search_policy(self, query, n_results=3):
        """Performs a vector search for relevant policies."""
        collection = self.client.get_collection(name=self.collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results to match the expected tool output
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "category": results['metadatas'][0][i]['category'],
                "benefit": results['metadatas'][0][i]['benefit'],
                "condition": results['metadatas'][0][i]['condition'],
            })
        return formatted

if __name__ == "__main__":
    # Example usage for initialization
    index = RetentionSearchIndex()
    index.initialize_index()
    
    # Quick test search
    test_results = index.search_policy("loyalty discount")
    print(f"Test search results: {test_results}")
