import os
import asyncio
from typing import List, Optional, Tuple
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class KnowledgeBase:
    def __init__(
        self,
        urls: List[str],
        open_ai_key: str,
        index_dir: str = "kb_index",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.urls = urls
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=open_ai_key)

        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _index_files_exist(self) -> bool:
        return (
            os.path.exists(os.path.join(self.index_dir, "index.faiss"))
            and os.path.exists(os.path.join(self.index_dir, "index.pkl"))
        )

    def save_index(self) -> None:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized; nothing to save.")
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store.save_local(self.index_dir)

    def load_index(self) -> None:
        self.vector_store = FAISS.load_local(
            self.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    async def scrape_data(self) -> List[Document]:
        sem = asyncio.Semaphore(5)

        async def load_one(link: str) -> List[Document]:
            async with sem:
                loader = OnlinePDFLoader(link)
                if hasattr(loader, "aload"):
                    docs = await loader.aload()
                else:
                    docs = await asyncio.to_thread(loader.load)

            docs_list = docs if isinstance(docs, list) else [docs]
            for d in docs_list:
                d.metadata = d.metadata or {}
                d.metadata["source_url"] = link
            return docs_list

        tasks = [load_one(url) for url in self.urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        docs: List[Document] = []
        for batch in results:
            docs.extend(batch)
        return docs

    async def process_and_store_docs(self, docs: List[Document]) -> None:
        split_docs = self.text_splitter.split_documents(docs)
        for i, d in enumerate(split_docs):
            d.metadata = d.metadata or {}
            d.metadata["chunk_id"] = i
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

    async def build_or_load(self, force_rebuild: bool = False) -> str:
        if (not force_rebuild) and self._index_files_exist():
            self.load_index()
            return "Knowledge Base Loaded"

        docs = await self.scrape_data()
        await self.process_and_store_docs(docs)
        self.save_index()
        return "Knowledge Base Built"

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Call build_or_load() first.")
        return self.vector_store.similarity_search(query, k=k)

    def retrieve(self, query: str, k: int = 10) -> Tuple[str, List[dict]]:
        docs = self.similarity_search(query, k=k)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        sources = []
        for d in docs:
            meta = d.metadata or {}
            sources.append(
                {
                    "source_url": meta.get("source_url"),
                    "chunk_id": meta.get("chunk_id"),
                }
            )
        return context, sources
