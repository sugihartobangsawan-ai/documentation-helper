import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyCrawl, TavilyMap, TavilyExtract
from sqlalchemy.testing.suite.test_reflection import metadata
from tenacity import retry

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning )

load_dotenv()

#configure SSL context to use certifi certificates
#must be run without vpn
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001'
    )

vector_store = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20,max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 5):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # # Process batches concurrently
    # tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)
    #
    # # Count successful batches
    # successful = sum(1 for result in results if result is True)

    successful = 0

    for i, batch in enumerate(batches):
        result = await add_batch(batch, i + 1)

        if result:
            successful += 1

        # small pause to avoid rate limits
        await asyncio.sleep(0.5)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )



async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://docs.langchain.com/oss/python/langchain/overview",
        Colors.PURPLE,
    )

    #crawl the documentation site
    res= tavily_crawl.invoke({
        'url':'https://docs.langchain.com/oss/python/langchain/overview',
        'max_depth':1,
        'extract_depth':'advanced',
        'instruction':'content on ai agents'
    })

    #raw content + content
    # all_docs = [
    #     Document(
    #         page_content=result.get("raw_content") or result.get("content"),
    #         metadata={"source": result["url"]}
    #     )
    #     for result in res["results"]
    #     if result.get("raw_content") or result.get("content")
    # ]

    # only raw content
    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}
        )
        for result in res["results"]
        if result.get("raw_content")
    ]

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} urls from documentation site",
    )

    #split documents into chunks
    log_header("DOCUMENTATION CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 1500 chunk size and 150 overlap",
        Colors.YELLOW
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=5)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

###IT DOESNT WORK SO DEBUG THIS SHIT


if __name__ == '__main__':
    asyncio.run(main())

