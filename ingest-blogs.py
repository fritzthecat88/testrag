import os
import argparse
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.schema import Document
import pinecone
import sys

# Add the main application directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the custom splitter from the main app
from app import EnhancedMarkdownSplitter, extract_metadata_from_markdown

# Load environment variables
load_dotenv()

# Configure constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"


def process_blog_directory(blog_dir, index_name):
    """Process all markdown files in the given directory and index them."""
    # Find all markdown files
    markdown_files = glob.glob(os.path.join(blog_dir, "*.md"))
    
    if not markdown_files:
        print(f"No markdown files found in {blog_dir}")
        return
    
    print(f"Found {len(markdown_files)} markdown files")
    
    # Process each file
    all_docs = []
    for file_path in markdown_files:
        try:
            print(f"Processing {file_path}...")
            
            # Extract basic file metadata
            file_name = os.path.basename(file_path)
            file_metadata = {
                "source": file_path,
                "file_name": file_name,
                "title": file_name.replace(".md", "").replace("-", " ").title()
            }
            
            # Load the markdown document
            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            docs = loader.load()
            
            # Add file metadata to each document
            for doc in docs:
                doc.metadata.update(file_metadata)
            
            all_docs.extend(docs)
            print(f"  Added {len(docs)} documents from {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Split the documents
    splitter = EnhancedMarkdownSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks")
    
    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    # Create or check index
    if index_name not in pinecone.list_indexes():
        print(f"Creating new index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Dimensionality for OpenAI embeddings
            metric="cosine"
        )
    else:
        print(f"Using existing index: {index_name}")
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    print("Indexing documents (this may take a while)...")
    vector_store = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"Successfully indexed {len(split_docs)} document chunks to {index_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index markdown blog files into Pinecone")
    parser.add_argument("--blog_dir", default="./blogs", help="Directory containing markdown blog files")
    parser.add_argument("--index_name", default="blog-knowledge-base", help="Name of the Pinecone index")
    
    args = parser.parse_args()
    
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT"):
        print("Error: PINECONE_API_KEY or PINECONE_ENVIRONMENT environment variables not set")
        sys.exit(1)
    
    # Process the blog directory
    process_blog_directory(args.blog_dir, args.index_name)
