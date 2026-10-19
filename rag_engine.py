import os
from typing import Dict, Any, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI


class RAGAssistant:
    def __init__(self, pdf_path: str, persist_dir: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.rag_chain = None

    def build_pipeline(self, api_key: str, model_name: str = "gpt-3.5-turbo", chunk_size: int = 1000, chunk_overlap: int = 200):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File not found: {self.pdf_path}")

        print(f" PDF loaded: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print("Text Chunking initializing...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        print(f"Total {len(splits)} chunks created.")

        print("ChromaDB vector store initializing...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know.\n"
            "Keep the answer concise and strictly based on the context.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Initialize OpenAI LLM with user-provided API key and model
        llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG pipeline initialized.")

    def ask(self, query: str) -> Dict[str, Any]:
        if not self.rag_chain:
            raise ValueError("Pipeline not initialized. Please run build_pipeline() first.")

        response = self.rag_chain.invoke({"input": query})
        
        citations = []
        seen = set()
        for doc in response.get("context", []):
            page_num = doc.metadata.get("page", 0) + 1  
            source_file = os.path.basename(doc.metadata.get("source", "PDF"))
            key = (source_file, page_num)
            
            if key not in seen:
                seen.add(key)
                citations.append({
                    "file": source_file,
                    "page": page_num,
                    "content_snippet": doc.page_content[:150] + "..."
                })

        return {
            "answer": response["answer"],
            "citations": citations
        }