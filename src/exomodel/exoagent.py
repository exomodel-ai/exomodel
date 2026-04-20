# Copyright 2026 Leandro Pessoa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import datetime

# Set User-Agent for web scraping consistency
os.environ["USER_AGENT"] = "ExoAgentApp/1.0"

import requests
import bs4
import html2text
from typing import List, Optional, Any, Type
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.embeddings import init_embeddings 
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class ExoAgent:
    """
    ExoAgent manages LLM interactions, RAG (Retrieval-Augmented Generation) context,
    and tool orchestration.
    """

    def __init__(self):
        # Configuration
        self.model_id = os.getenv("MY_LLM_MODEL", "google_genai:gemini-2.5-flash-lite")

        # Universal embedding map by provider
        embedding_map = {
            "google_genai": "google_genai:gemini-embedding-001",
            "openai": "openai:text-embedding-3-small",
            "anthropic": "openai:text-embedding-3-small", # Anthropic usually uses OpenAI or Cohere
            "cohere": "cohere:embed-english-v3.0",
            "azure_openai": "azure_openai:text-embedding-3-small"
        }
            
        # Dynamic Embedding Model selection based on provider
        provider = self.model_id.split(":")[0]

        # Busca no mapa, ou usa o valor da ENV se o usuário quiser sobrescrever totalmente
        self.emb_model = os.getenv("MY_EMB_MODEL") or embedding_map.get(provider, "google_genai:gemini-embedding-001")

        # State Management
        self.sources_queue: List[str] = []
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.rag_tools: List[Any] = []
        self.external_tools: List[Any] = []
        
        self._agent = None
        self._last_schema = None
        self._current_mode = "generalist"

    def add_rag_sources(self, sources: List[str]):
        """Schedule sources for indexing. Cost: 0 tokens until processed."""
        self.sources_queue.extend(sources)
        self._agent = None  # Force agent rebuild to include new context

    def _process_pending_rag(self):
        """Loads and embeds scheduled sources only when necessary."""
        if not self.sources_queue and self.vector_store is not None:
            return

        documents = []
        for source in self.sources_queue:
            loader = self._get_loader(source)
            docs = loader.load()
            
            # Enrich metadata per chunk
            for doc in docs:
                doc.metadata["source"] = source
                doc.metadata["indexed_at"] = datetime.now().isoformat()
                doc.metadata["source_type"] = (
                    "pdf" if source.endswith(".pdf")
                    else "web" if source.startswith("http")
                    else "text"
                )
            documents.extend(docs)
        
        self.sources_queue = []

        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = InMemoryVectorStore(init_embeddings(self.emb_model))
            
            self.vector_store.add_documents(splits)

        # Update RAG tool if vector store is initialized
        if self.vector_store is not None:
            @tool
            def retrieve_context(query: str) -> str:
                """Query the private knowledge base to retrieve factual context."""
                
                # similarity_search_with_score retorna (doc, score)
                results = self.vector_store.similarity_search_with_score(query, k=5)
                
                SCORE_THRESHOLD = 0.75  # ajuste conforme o modelo de embedding
                relevant = [
                    (doc, score) for doc, score in results 
                    if score >= SCORE_THRESHOLD
                ]
                
                if not relevant:
                    return "No sufficiently relevant content found in the knowledge base."
                
                chunks = []
                for doc, score in relevant:
                    source = doc.metadata.get("source", "unknown")
                    chunks.append(f"[Source: {source} | Relevance: {score:.2f}]\n{doc.page_content}")
                
                return "\n\n---\n\n".join(chunks)
            
            self.rag_tools = [retrieve_context]
        else:
            self.rag_tools = []

    @property
    def all_tools(self) -> List[Any]:
        """Combines RAG tools and external action tools."""
        return self.rag_tools + self.external_tools

    def set_external_tools(self, tools: List[Any]):
        """Register external tools and reset agent state."""
        self.external_tools = tools
        self._agent = None

    def _get_loader(self, source: str):
        """Factory method to return the appropriate loader based on source type."""
        if source.lower().endswith(".pdf"):
            return PyPDFLoader(source)
        
        if source.startswith("http"):
            return WebBaseLoader(
                web_paths=(source,),
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/119.0.0.0 Safari/537.36'
                }
            )
        
        return TextLoader(source)

    def _get_system_prompt(self, mode: str) -> str:
        """Centralized prompt repository for different agent personas."""
        prompts = {
            "generalist": (
                "You are a helpful and direct virtual assistant. "
                "Answer based on your trained knowledge, focusing on clarity and usefulness."
            ),
            "specialist": (
                "You are a Senior Domain Specialist. "
                "Your knowledge comes exclusively from the 'retrieve_context' tool. "
                "Never answer from general knowledge.\n\n"
                "WORKFLOW:\n"
                "1. Call 'retrieve_context' immediately before any response.\n"
                "2. Base your answer solely on the retrieved content.\n"
                "3. If 'retrieve_context' returns no relevant content, "
                "respond: 'No information found in the knowledge base for this query.'\n\n"
                "STYLE: Concise, direct, and objective. "
                "No conversational filler, no code snippets, no requests for more information."
            ),
            "hybrid": (
                "You are a Senior Domain Specialist. "
                "Your primary knowledge source is the 'retrieve_context' tool.\n\n"
                "WORKFLOW:\n"
                "1. Call 'retrieve_context' first.\n"
                "2. Build your answer from the retrieved content.\n"
                "3. If the retrieved content is incomplete, supplement with your general knowledge "
                "— but never contradict what was retrieved.\n"
                "4. When using general knowledge beyond the retrieved content, "
                "signal it explicitly: '[General knowledge]'.\n\n"
                "STYLE: Concise, direct, and objective. "
                "No conversational filler, no code snippets, no requests for more information."
            ),
            "orchestrator": (
                "You are an Orchestrator Agent. Your sole function is to evaluate the user "
                "request and route it to the correct action.\n\n"
                "WORKFLOW:\n"
                "1. TOOL CALL: If the intent matches an available tool, invoke it immediately. "
                "When a tool requires a 'prompt' argument, pass the user's original request "
                "verbatim — never paraphrase or summarize it.\n"
                "2. DIRECT ANSWER: If the request is a simple read-only question answerable "
                "from the current entity state, respond concisely with that data only.\n"
                "3. REJECT: If the request is out of scope or no tool matches, reply exactly: "
                "'I cannot fulfill this request based on the available tools and data.'\n\n"
                "RULES:\n"
                "- Never perform complex logic yourself. Always delegate to tools.\n"
                "- Never hallucinate tool capabilities or arguments.\n"
                "- Output the tool result as-is. No summaries, no filler, no commentary.\n\n"
                "STYLE: Professional, direct, and objective."
            )
        }
        return prompts.get(mode, prompts["generalist"])

    def _init_agent(self, response_schema: Optional[Type] = None, mode: str = "generalist"):
        """
        Reconstructs the agent with updated tools and context using the unified 
        LangChain 0.3+ agent creation pattern.
        """
        # 1. Process any pending RAG documents to update the vector store
        self._process_pending_rag()

        # 2. Fallback to generalist if specialist/hybrid is requested without RAG data
        if mode != "generalist" and not self.rag_tools:
            print(f"Warning: Mode '{mode}' requested without RAG context. Using 'generalist'.")
            mode = "generalist"
        
        # 3. Retrieve the localized system prompt instructions for the chosen mode
        system_prompt = self._get_system_prompt(mode)
        print(f"\n[INIT] Mode: {mode} | Instructions: {system_prompt[:80]}...")

        # 4. Initialize the unified agent. 
        self._agent = create_agent(
            self.model_id, 
            tools=self.all_tools, 
            system_prompt=system_prompt,
            response_format=response_schema
        )      

    def run(self, prompt: str, response_schema: Optional[Type] = None, mode: str = "generalist") -> Any:
        """Execution entry point for the agent."""
        print(f"\n[RUN] Mode: {mode} | Schema: {response_schema.__name__ if response_schema else 'None'}")
        
        # Re-initialize if parameters changed
        if not self._agent or mode != self._current_mode or response_schema != self._last_schema:
            self._current_mode = mode
            self._last_schema = response_schema
            self._init_agent(response_schema=response_schema, mode=mode)

        config = {
            "configurable": {"max_tokens": 4096, "temperature": 0},
            "recursion_limit": 10
        }
            
        print("Waiting for LLM response...")
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config
        )
        
        # Handle Structured Output
        if response_schema and "structured_response" in result:
            raw_data = result["structured_response"]
            return response_schema(**raw_data) if isinstance(raw_data, dict) else raw_data
            
        # Extract content from the last AI Message
        final_message = result["messages"][-1]
        final_content = getattr(final_message, 'content', '')
        
        # Silent Output Fallback (Anti-Mute Trigger)
        if not final_content:
            print("[CRITICAL] Empty response detected. Triggering recovery fallback.")
            rag_text = "No context retrieved."
            for msg in reversed(result['messages']):
                if getattr(msg, 'type', '').lower() in ['tool', 'toolmessage']:
                    rag_text = getattr(msg, 'content', rag_text)
                    break
            
            final_content = (
                "⚠️ **System Warning:** The AI failed to generate a response. "
                "Displaying raw context found:\n\n"
                f"_{rag_text}_"
            )
            
        return str(final_content).strip()

    def get_web_markdown(self, url: str) -> str:
        """Fetches a URL and converts its body to clean Markdown."""
        response = requests.get(url, headers={'User-Agent': 'ExoAgentApp/1.0'})
        html_content = response.text
        
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.bypass_tables = False
        converter.body_width = 0
        converter.ignore_images = True
        
        markdown_content = converter.handle(html_content)
        return markdown_content[:16000]
        