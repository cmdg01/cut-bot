import os
from typing import List, Dict, Any
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, data_dir: str = "data/documents/"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings()
        
    def load_text_documents(self) -> List:
        """Load all text documents from the data directory"""
        logger.info("Loading text documents...")
        try:
            text_loader = DirectoryLoader(
                os.path.join(self.data_dir, "text"),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = text_loader.load()
            logger.info(f"Loaded {len(documents)} text documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading text documents: {str(e)}")
            return []

    def process_json_data(self, json_file: str) -> List:
        """Process university JSON data"""
        logger.info(f"Processing JSON data from {json_file}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            texts = []
            
            def process_dict(d: Dict[str, Any], prefix: str = '') -> None:
                for key, value in d.items():
                    if isinstance(value, dict):
                        process_dict(value, f"{prefix}{key} - ")
                    elif isinstance(value, list):
                        texts.append(f"{prefix}{key}: {', '.join(str(x) for x in value)}")
                    else:
                        texts.append(f"{prefix}{key}: {value}")
            
            process_dict(data)
            documents = self.text_splitter.create_documents(texts)
            logger.info(f"Processed {len(documents)} sections from JSON data")
            return documents
        except Exception as e:
            logger.error(f"Error processing JSON data: {str(e)}")
            return []

    def create_vector_store(self, documents: List, store_name: str = "university_vectorstore") -> FAISS:
        """Create FAISS vector store from documents"""
        logger.info("Creating vector store...")
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            save_path = f"data/embeddings/{store_name}"
            vectorstore.save_local(save_path)
            logger.info(f"Vector store saved to {save_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

def main():
    """Main function to process documents and create vector store"""
    # Create necessary directories
    os.makedirs("data/documents/text", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)

    logger.info("Starting document processing...")
    
    processor = DocumentProcessor()
    
    # Load and process documents with progress bar
    with tqdm(total=3, desc="Processing Steps") as pbar:
        # Load text documents
        documents = processor.load_text_documents()
        pbar.update(1)
        
        # Process JSON data
        json_docs = processor.process_json_data("university_data.json")
        pbar.update(1)
        
        # Combine all documents
        all_documents = documents + json_docs
        
        # Create and save vector store
        vectorstore = processor.create_vector_store(all_documents)
        pbar.update(1)
    
    logger.info("Document processing completed successfully!")
    logger.info(f"Total documents processed: {len(all_documents)}")

if __name__ == "__main__":
    main()