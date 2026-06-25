import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()


#initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001'
    )
vector_store = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings,
)

#initialize chat model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
     """Retrieve relevant documentation to help answer user queries about Langchain"""
      #retrieve top 4 most similar document
     retriever = vector_store.as_retriever(
         search_kwargs={"k": 4}
     )

     retrieved_docs = retriever.invoke(query)

     #serialized documents for the model
     serialized = "\n\n".join(
         (f"Source: {doc.metadata.get('source','unknown')} \n\nContent: {doc.page_content}")
         for doc in retrieved_docs
     )

     #return both serialized content and raw document
     return serialized, retrieved_docs

def run_llm(query:str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation

    Args:
         query: The user's question

     Returns:
         Dictionary containing:
            - answer: the generated answer
            - context: List of retrieved documents
    """

    #create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "

        "You have access to a tool called retrieve_context that retrieves relevant "
        "documentation from a vector database. "

        "For EVERY user question, you MUST call retrieve_context before answering. "
        "Do NOT answer from your own knowledge. "
        "Only answer using the retrieved documentation. "

        "If the retrieved documentation does not contain enough information, "
        "say that the documentation does not provide the answer. "

        "Always cite the source URLs included in the retrieved documentation."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    #build messages list
    messages = [{'role':'user','content':query}]

    #invoke the agent
    response=agent.invoke({'messages':messages})

    #extract the answer from the last ai message
    answer = response['messages'][-1].content

    #extract context documents from ToolMessage artifact
    context_docs = []
    for message in response['messages']:
        #check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message,'artifact'):
            #the artifact
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return{
        'answer': answer,
        'context': context_docs
    }

if __name__ == '__main__':
    result=run_llm(query="what are deep agents?")
    print(result)


