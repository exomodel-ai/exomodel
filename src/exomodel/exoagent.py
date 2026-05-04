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

import importlib
import logging
import os
from datetime import datetime
from typing import Any, Optional

# Must be set before WebBaseLoader is imported — LangChain checks at import time.
os.environ.setdefault("USER_AGENT", "ExoAgentApp/1.0")

import html2text
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

class ExoAgent:
    """LLM engine used by `ExoModel` — manages conversation context, RAG, and tool routing.

    Can also be used directly for lower-level control: building custom agent loops,
    attaching external tools, or fetching and pre-processing web content before
    passing it to a model.

    The LLM provider is selected from the ``MY_LLM_MODEL`` environment variable using
    a ``"provider:model"`` string (e.g. ``"google_genai:gemini-2.5-flash-lite"``).
    The provider package must be installed separately — see
    ``pip install exomodel[google|anthropic|openai|cohere]``.

    Four agent personas are supported, selected per ``run()`` call via the ``mode`` parameter:

    - ``"generalist"`` — answers from the LLM's trained knowledge only.
    - ``"specialist"`` — answers exclusively from the RAG knowledge base; refuses to guess.
    - ``"hybrid"`` — prefers RAG context but supplements with general knowledge when needed.
    - ``"orchestrator"`` — routes the request to the best matching tool; used by `master_prompt`.
    """

    # Maps provider prefix → (pip extra, importable module)
    _PROVIDER_PACKAGES = {
        "google_genai": ("google", "langchain_google_genai"),
        "anthropic":    ("anthropic", "langchain_anthropic"),
        "claude":       ("anthropic", "langchain_anthropic"),
        "openai":       ("openai", "langchain_openai"),
        "azure_openai": ("openai", "langchain_openai"),
        "cohere":       ("cohere", "langchain_cohere"),
    }

    _NO_PROVIDER_MSG = (
        "No LLM provider configured. Set MY_LLM_MODEL in your .env file and install the provider:\n\n"
        "  MY_LLM_MODEL=google_genai:gemini-2.5-flash-lite  →  pip install exomodel[google]\n"
        "  MY_LLM_MODEL=anthropic:claude-sonnet-4-6          →  pip install exomodel[anthropic]\n"
        "  MY_LLM_MODEL=openai:gpt-4o                        →  pip install exomodel[openai]\n"
        "  MY_LLM_MODEL=cohere:command-r-plus                →  pip install exomodel[cohere]"
    )

    @staticmethod
    def _check_provider(provider: str, configured: bool) -> None:
        """Raises a clear ImportError if the provider's package is not installed."""
        entry = ExoAgent._PROVIDER_PACKAGES.get(provider)
        if entry is None:
            return  # unknown provider — let LangChain surface the error
        extra, module = entry
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            if not configured:
                raise ImportError(ExoAgent._NO_PROVIDER_MSG) from None
            raise ImportError(
                f"Provider '{provider}' requires '{module.replace('_', '-')}'.\n"
                f"Install it with: pip install exomodel[{extra}]"
            ) from None

    def __init__(self, temperature: float = 0, max_tokens: int = 4096):
        """Initialises the agent, validates the configured provider, and wires the RAG tool.

        The ``retrieve_context`` tool is built once here as a closure that reads
        ``self.vector_store`` at call time, so it reflects any newly indexed sources
        without being recreated on each RAG call.

        Args:
            temperature: Sampling temperature applied to every call unless overridden per call.
            max_tokens: Maximum response length applied to every call unless overridden per call.

        Raises:
            ImportError: If the required provider package is not installed.
        """
        # Detect whether the user explicitly set a provider so the error message can differ.
        raw_model_env = os.getenv("MY_LLM_MODEL")
        self.model_id = raw_model_env or "google_genai:gemini-2.5-flash-lite"
        self.temperature = temperature
        self.max_tokens = max_tokens

        embedding_map = {
            "google_genai": "google_genai:gemini-embedding-001",
            "openai": "openai:text-embedding-3-small",
            "anthropic": "openai:text-embedding-3-small",  # Anthropic has no native embedding model
            "claude": "openai:text-embedding-3-small",     # Anthropic has no native embedding model
            "cohere": "cohere:embed-english-v3.0",
            "azure_openai": "azure_openai:text-embedding-3-small"
        }

        provider = self.model_id.split(":")[0]

        # Fail fast — a missing provider package produces a cryptic LangChain error
        # deep inside invoke(); a clear ImportError here is much easier to diagnose.
        self._check_provider(provider, configured=raw_model_env is not None)

        self.emb_model = (
            os.getenv("MY_EMB_MODEL")
            or embedding_map.get(provider, "google_genai:gemini-embedding-001")
        )

        self.sources_queue: list[str] = []
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.rag_tools: list[Any] = []
        self.external_tools: list[Any] = []

        self._agent = None
        self._last_schema = None
        self._current_mode = "generalist"
        self._last_temperature = temperature
        self._last_max_tokens = max_tokens

        self._usage: dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}

        # Closure captures self so it always reads the current vector_store,
        # even after new sources are indexed between calls.
        @tool
        def retrieve_context(query: str) -> str:
            """Query the private knowledge base to retrieve factual context."""
            if self.vector_store is None:
                return "No knowledge base initialized."
            results = self.vector_store.similarity_search_with_score(query, k=5)
            SCORE_THRESHOLD = 0.75
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

        self._retrieve_context_tool = retrieve_context

    def add_rag_sources(self, sources: list[str]):
        """Enqueues sources (URLs, PDF paths, or text file paths) for lazy RAG indexing.

        No embedding or I/O happens here — sources are processed on the next ``run()``
        call. Resets the cached agent so the new context is picked up immediately.
        """
        self.sources_queue.extend(sources)
        self._agent = None

    def _process_pending_rag(self):
        """Loads, chunks, and embeds all queued sources into the in-memory vector store.

        Skips entirely when the queue is empty and a store already exists.
        Failed sources are logged and discarded — the queue is always cleared so a
        broken source does not block every subsequent call.
        """
        if not self.sources_queue and self.vector_store is not None:
            return

        documents = []
        failed = []
        for source in self.sources_queue:
            try:
                loader = self._get_loader(source)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = source
                    doc.metadata["indexed_at"] = datetime.now().isoformat()
                    doc.metadata["source_type"] = (
                        "pdf" if source.endswith(".pdf")
                        else "web" if source.startswith("http")
                        else "text"
                    )
                documents.extend(docs)
            except Exception as e:
                logger.warning("Failed to load RAG source '%s': %s", source, e)
                failed.append(source)

        # Always clear — retrying failed sources on every call would spam errors.
        self.sources_queue = []
        if failed:
            logger.warning("%d RAG source(s) skipped: %s", len(failed), failed)

        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(documents)

            if self.vector_store is None:
                self.vector_store = InMemoryVectorStore(init_embeddings(self.emb_model))

            self.vector_store.add_documents(splits)

        self.rag_tools = [self._retrieve_context_tool] if self.vector_store is not None else []

    @property
    def all_tools(self) -> list[Any]:
        """Combines RAG tools and external action tools."""
        return self.rag_tools + self.external_tools

    def set_external_tools(self, tools: list[Any]):
        """Replaces the external tool list and invalidates the cached agent.

        External tools are LangChain ``StructuredTool`` objects — typically the
        ``@llm_function``-decorated methods discovered by ``ExoModel.llm_tools``.
        Passing an empty list removes all external tools without affecting RAG tools.
        """
        self.external_tools = tools
        self._agent = None

    def get_usage(self) -> dict:
        """Returns accumulated token usage across all calls since last reset."""
        return dict(self._usage)

    def reset_usage(self) -> None:
        """Resets the token usage counters to zero."""
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}

    def _record_usage(self, messages: list) -> None:
        """Reads token counts from the last AIMessage and accumulates them into ``_usage``."""
        for msg in reversed(messages):
            # Gemini/Anthropic expose usage_metadata as a top-level AIMessage attribute.
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                # OpenAI and some others nest it inside response_metadata instead.
                meta = getattr(msg, "response_metadata", {}) or {}
                usage = (
                    meta.get("usage_metadata")
                    or meta.get("token_usage")
                    or meta.get("usage")
                )
            if not usage:
                continue
            self._usage["input_tokens"] += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            self._usage["output_tokens"] += usage.get("output_tokens") or usage.get("completion_tokens") or 0
            self._usage["total_tokens"] += usage.get("total_tokens") or (
                self._usage["input_tokens"] + self._usage["output_tokens"]
            )
            self._usage["calls"] += 1
            return  # only the last AI message carries usage

    def _get_loader(self, source: str):
        """Returns the appropriate LangChain document loader for a given source.

        Dispatches on file extension or URL scheme: ``.pdf`` → ``PyPDFLoader``,
        ``http(s)://`` → ``WebBaseLoader`` with a standard browser User-Agent,
        everything else → ``TextLoader``.
        """
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
        """Returns the system prompt for the given agent persona.

        Falls back to ``"generalist"`` for unrecognised mode strings so callers
        never receive an empty system prompt.
        """
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
                "1. Call 'retrieve_context' immediately.\n"
                "2. After the tool returns, you MUST write your answer as plain text.\n"
                "3. Base your answer solely on the retrieved content.\n"
                "4. If 'retrieve_context' returns no relevant content, "
                "respond: 'No information found in the knowledge base for this query.'\n\n"
                "STYLE: Concise, direct, and objective. "
                "No conversational filler, no code snippets, no requests for more information."
            ),
            "hybrid": (
                "You are a Senior Domain Specialist. "
                "Your primary knowledge source is the 'retrieve_context' tool.\n\n"
                "WORKFLOW:\n"
                "1. Call 'retrieve_context' first.\n"
                "2. After the tool returns, you MUST write your answer as plain text.\n"
                "3. Build your answer from the retrieved content.\n"
                "4. If the retrieved content is incomplete, supplement with your general knowledge "
                "— but never contradict what was retrieved.\n"
                "5. When using general knowledge beyond the retrieved content, "
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

    def _init_agent(
        self,
        response_schema: Optional[type] = None,
        mode: str = "generalist",
        temperature: float = 0,
        max_tokens: int = 4096,
    ):
        """Rebuilds the LangChain agent with current tools, mode, and LLM parameters.

        RAG sources are processed here (not at enqueue time) so embedding only
        happens when a call is actually made. Specialist and hybrid modes are
        silently downgraded to generalist when no RAG store exists.
        """
        self._process_pending_rag()

        if mode != "generalist" and not self.rag_tools:
            logger.warning(
                "Mode '%s' requested without RAG context. Falling back to 'generalist'.", mode
            )
            mode = "generalist"

        system_prompt = self._get_system_prompt(mode)
        logger.debug("Init — mode: %s | instructions: %.80s...", mode, system_prompt)

        # init_chat_model is provider-agnostic: reads the "provider:model" prefix and
        # instantiates the right LangChain class (ChatGoogleGenerativeAI, ChatOpenAI, …).
        llm = init_chat_model(
            self.model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._agent = create_agent(
            llm,
            tools=self.all_tools,
            system_prompt=system_prompt,
            response_format=response_schema
        )

    def run(
        self,
        prompt: str,
        response_schema: Optional[type] = None,
        mode: str = "generalist",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Sends a prompt to the agent and returns its response.

        The agent is rebuilt only when a parameter that shapes its behaviour changes
        (mode, schema, temperature, or max_tokens), so repeated calls with the same
        configuration reuse the cached agent.

        Args:
            prompt: The user message to send.
            response_schema: Optional Pydantic model class for structured output extraction.
            mode: Agent persona — ``"generalist"``, ``"specialist"``, ``"hybrid"``,
                  or ``"orchestrator"``.
            temperature: Per-call override; falls back to the instance default when ``None``.
            max_tokens: Per-call override; falls back to the instance default when ``None``.

        Returns:
            A Pydantic instance when ``response_schema`` is provided; a plain string otherwise.
        """
        eff_temp = temperature if temperature is not None else self.temperature
        eff_tokens = max_tokens if max_tokens is not None else self.max_tokens

        logger.debug("Run — mode: %s | schema: %s", mode, response_schema.__name__ if response_schema else "None")

        # Rebuild only when something that affects agent behaviour has changed.
        if (not self._agent
                or mode != self._current_mode
                or response_schema != self._last_schema
                or eff_temp != self._last_temperature
                or eff_tokens != self._last_max_tokens):
            self._current_mode = mode
            self._last_schema = response_schema
            self._last_temperature = eff_temp
            self._last_max_tokens = eff_tokens
            self._init_agent(response_schema=response_schema, mode=mode,
                             temperature=eff_temp, max_tokens=eff_tokens)

        config = {"recursion_limit": 10}

        logger.debug("Waiting for LLM response...")
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config
        )

        self._record_usage(result.get("messages", []))

        if response_schema and "structured_response" in result:
            raw_data = result["structured_response"]
            return response_schema(**raw_data) if isinstance(raw_data, dict) else raw_data

        final_message = result["messages"][-1]
        final_content = getattr(final_message, 'content', '')

        # Gemini sometimes skips the final text turn after a tool call, leaving content empty.
        # Re-ask the LLM directly (no agent, no tools) with the tool result injected as context.
        if not final_content:
            logger.warning("Empty response from agent. Attempting direct LLM recovery...")
            rag_text = None
            for msg in reversed(result['messages']):
                if getattr(msg, 'type', '').lower() in ['tool', 'toolmessage']:
                    rag_text = getattr(msg, 'content', None)
                    if rag_text:
                        break

            if rag_text:
                recovery_llm = init_chat_model(
                    self.model_id,
                    temperature=eff_temp,
                    max_tokens=eff_tokens
                )
                recovery_response = recovery_llm.invoke([
                    ("system", "Answer the user's question based solely on the "
                               "provided context. Be concise and direct."),
                    ("human", f"Context:\n{rag_text}\n\nQuestion: {prompt}")
                ])
                final_content = getattr(recovery_response, 'content', '')

            if not final_content:
                final_content = "No response generated from the knowledge base."

        return str(final_content).strip()

    def get_web_markdown(self, url: str) -> str:
        """Fetches a URL and returns its body as clean Markdown, truncated to 16 000 characters.

        Useful for pre-processing web content before passing it to the LLM — for
        example, scraping a product page and calling ``update_object`` with the result.
        The 16 000-character cap keeps the content within typical context-window limits.
        """
        response = requests.get(url, headers={'User-Agent': 'ExoAgentApp/1.0'})
        html_content = response.text

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.bypass_tables = False
        converter.body_width = 0
        converter.ignore_images = True

        markdown_content = converter.handle(html_content)
        return markdown_content[:16000]
