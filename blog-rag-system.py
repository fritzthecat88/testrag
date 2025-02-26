import os
import glob
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pinecone
import re
from typing import List, Dict, Any, Tuple
import markdown

# Configure environment variables (in a real implementation, use .env or secrets)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

# Constants
INDEX_NAME = "blog-knowledge-base"
BLOG_PATH = "./blogs/"  # Path to your markdown blog files
CHUNK_SIZE = 800  # Target chunk size in tokens
CHUNK_OVERLAP = 150  # Overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"


def extract_metadata_from_markdown(text: str) -> Dict[str, Any]:
    """
    Extract useful metadata from markdown content.
    """
    metadata = {}
    
    # Extract headings
    headings = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
    if headings:
        metadata["headings"] = headings
    
    # Extract links
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
    if links:
        metadata["links"] = [{"text": text, "url": url} for text, url in links]
    
    # Check if contains code blocks
    metadata["contains_code"] = bool(re.search(r'```[\s\S]*?```', text))
    
    # Check if contains lists
    metadata["contains_lists"] = bool(re.search(r'^\s*[-*+]\s+', text, re.MULTILINE))
    
    # Estimate reading time (rough approximation)
    word_count = len(text.split())
    metadata["word_count"] = word_count
    metadata["reading_time_minutes"] = round(word_count / 200)  # Assuming 200 words per minute
    
    return metadata


class EnhancedMarkdownSplitter:
    """
    A markdown-aware text splitter that preserves structure and metadata.
    """
    
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.base_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split markdown documents while preserving structure and adding metadata.
        """
        result_docs = []
        
        for doc in documents:
            # First, get document-level metadata
            doc_metadata = doc.metadata.copy()
            doc_content = doc.page_content
            
            # Split into chunks with the base markdown splitter
            chunks = self.base_splitter.split_text(doc_content)
            
            for i, chunk in enumerate(chunks):
                # Extract metadata specific to this chunk
                chunk_metadata = extract_metadata_from_markdown(chunk)
                
                # Determine the context - what headings is this under?
                # This looks at the chunk and previous chunks to find the most recent headings
                current_headings = chunk_metadata.get("headings", [])
                
                # Combine document and chunk metadata
                combined_metadata = {
                    **doc_metadata,
                    **chunk_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                # Create a new document with the chunk and enhanced metadata
                result_docs.append(
                    Document(page_content=chunk, metadata=combined_metadata)
                )
        
        return result_docs


def get_context_from_headings(docs: List[Document], max_headings=2) -> Dict[str, List[str]]:
    """
    Extract hierarchical context from document headings to add to RAG context.
    """
    context = {}
    
    for doc in docs:
        headings = doc.metadata.get("headings", [])
        if headings:
            # Create hierarchical context
            for i, heading in enumerate(headings[:max_headings]):
                level = i + 1
                if f"h{level}" not in context:
                    context[f"h{level}"] = []
                if heading not in context[f"h{level}"]:
                    context[f"h{level}"].append(heading)
    
    return context


def load_and_process_blogs() -> List[Document]:
    """
    Load markdown blog files and process them into chunks with metadata.
    """
    # Find all markdown files in the blog directory
    markdown_files = glob.glob(os.path.join(BLOG_PATH, "*.md"))
    
    all_docs = []
    for file_path in markdown_files:
        try:
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
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Split the documents using our enhanced markdown splitter
    splitter = EnhancedMarkdownSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(all_docs)
    
    return split_docs


def initialize_vector_store(documents: List[Document]) -> PineconeVectorStore:
    """
    Initialize or update Pinecone vector store with document embeddings.
    """
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )
    
    # Create index if it doesn't exist
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Dimensionality for OpenAI embeddings
            metric="cosine"
        )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Create vector store from documents
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vector_store


def format_docs(docs):
    """Format retrieved documents into a string with metadata context."""
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        content = doc.page_content
        metadata = doc.metadata
        
        # Format headings for context if available
        heading_context = ""
        if "headings" in metadata and metadata["headings"]:
            heading_context = f"Section: {' > '.join(metadata['headings'])}\n"
        
        # Add source information
        source_info = f"Source: {metadata.get('title', 'Unknown')}"
        
        # Format the document with its context
        formatted_doc = f"Document {i+1}:\n{heading_context}{source_info}\n\n{content}\n\n"
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def create_rag_chain(vector_store):
    """Create the RAG retrieval and generation chain."""
    # Initialize retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    
    # Create prompt template
    template = """You are an AI assistant that answers questions based on the personal blog content of the author.
    Use only the following retrieved blog content to answer the question. If you don't know the answer based on the 
    provided content, say you don't know.

    Retrieved content:
    {context}

    Question: {question}

    Use a conversational tone that matches the writing style from the blog content. Include relevant details from 
    the blog to support your answer when possible.

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def index_blogs():
    """Index all blog content into Pinecone."""
    with st.spinner("Indexing blog content... This may take a few minutes."):
        # Load and process blog content
        docs = load_and_process_blogs()
        st.success(f"Processed {len(docs)} document chunks")
        
        # Initialize vector store
        vector_store = initialize_vector_store(docs)
        st.success("Blog content indexed successfully!")
        return vector_store


def main():
    st.title("Personal Blog Knowledge Base")
    
    with st.sidebar:
        st.title("Blog RAG System")
        st.write("This app uses RAG to answer questions about personal blog content.")
        
        # Add a button to reindex the blog content
        if st.button("Reindex Blog Content"):
            vector_store = index_blogs()
            st.session_state.vector_store = vector_store
    
    # Initialize or retrieve vector store
    if "vector_store" not in st.session_state:
        # Check if index exists in Pinecone
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"]
        )
        
        if INDEX_NAME in pinecone.list_indexes():
            # Connect to existing index
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            st.session_state.vector_store = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=embeddings
            )
            st.sidebar.success("Connected to existing index")
        else:
            # Create new index
            st.session_state.vector_store = index_blogs()
    
    # Create RAG chain
    rag_chain = create_rag_chain(st.session_state.vector_store)
    
    # User query input
    query = st.text_input("Ask a question about the blog content:")
    
    if query:
        with st.spinner("Searching knowledge base..."):
            response = rag_chain.invoke(query)
            st.write("### Answer")
            st.write(response)


if __name__ == "__main__":
    main()
