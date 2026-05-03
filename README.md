# 🌌 ExoModel AI
### The Object-Oriented Framework for Agentic AI

[![PyPI version](https://img.shields.io/pypi/v/exomodel?color=blue)](https://pypi.org/project/exomodel/)
[![License](https://img.shields.io/github/license/exomodel-ai/exomodel)](https://github.com/exomodel-ai/exomodel/blob/main/LICENSE)

📖 **Official Documentation:** [https://exomodel.ai](https://exomodel.ai)  
📦 **GitHub Repository:** [https://github.com/exomodel-ai/exomodel](https://github.com/exomodel-ai/exomodel)

---

**ExoModel** brings AI capabilities directly into your data models. Instead of building prompt pipelines around your objects, your objects become the agents — they populate themselves from natural language, consult their own documents via RAG, and validate their state against business rules. Built on top of LangChain and Pydantic.

---

## ⚡ Why ExoModel?

While traditional agent frameworks focus on chat, ExoModel focuses on the **Business Entity**. It brings type safety and structure to the chaotic world of LLMs.

| Traditional AI Apps | With ExoModel |
| :--- | :--- |
| Passive data models | Models that populate and update themselves from natural language |
| Fragile manual prompting | Type-safe field mapping — the schema is the prompt |
| Disconnected RAG pipelines | Documents attached directly to the class, consulted at runtime |
| Complex JSON parsing | Guaranteed schema-validated outputs via Pydantic |

---

## 🔥 Core Features

- **🧠 Smart CRUD** — Create and update class instances using natural language. ExoModel understands intent and maps it to specific schema fields.
- **📚 Native RAG Grounding** — Attach PDFs, URLs, or text files directly to your class. The object uses this "brain" to validate its own state against business rules.
- **🤖 ExoAgent Orchestration** — A centralized engine that manages tool routing and persona switching (`generalist`, `specialist`, `hybrid`, `orchestrator`) to optimize accuracy and cost.
- **🔌 API-First Design** — Transform messy human input into the strict JSON schemas required by your existing APIs and services.
- **⚙️ Agentic Tools with `@llm_function`** — Decorate any method with `@llm_function` to expose it as an agentic tool. The agent discovers and calls it autonomously — no manual tool registration required.
- **📊 Fluent List Management** — Handle collections of entities with `ExoModelList`, enabling bulk LLM generation and CSV/UI exports in a single call.

---

## 📦 Installation

ExoModel 1.0.0 is LLM-agnostic. Install only the provider package you need:

```bash
pip install exomodel[google]      # Gemini (default)
pip install exomodel[anthropic]   # Claude
pip install exomodel[openai]      # OpenAI / Azure OpenAI
pip install exomodel[cohere]      # Cohere
pip install exomodel[all]         # all providers
```

Then create a `.env` file at the root of your project:

```env
MY_LLM_MODEL=google_genai:gemini-2.5-flash-lite
MY_EMB_MODEL=google_genai:gemini-embedding-001   # optional — auto-detected from provider
GOOGLE_API_KEY=your-key-here
```

---

## 🚀 Quick Start

### 1. Create a Knowledge Base

Create a file named `proposal_rules.md`. This grounds your AI objects in real-world business logic.

```markdown
# Proposal Rules
- We only accept projects above $10,000.
- Every proposal must include a 10% safety margin in the pricing.
- We do not work with companies in the tobacco industry.
```

### 2. Define and Run your Entity

Inherit from `ExoModel` to give your data structures autonomous reasoning powers.

```python
from exomodel import ExoModel

class Proposal(ExoModel):
    client: str = ""
    budget: float = 0.0

    @classmethod
    def get_rag_sources(cls):
        # The object now 'knows' your specific business rules
        return ["proposal_rules.md"]

# Populate the object from raw natural language
p = Proposal.create("Draft a 50k proposal for Tesla")

# Display a formatted summary
print(p.to_ui())

# The object analyzes itself against proposal_rules.md
print(p.run_analysis())
```

---

## 🛠 Architecture

ExoModel is built to be modular, scalable, and provider-agnostic.

| Class | Role |
| :--- | :--- |
| **`ExoModel`** | The intelligent data foundation — schema-driven, AI-powered, RAG-aware. Subclass this to define your entities. |
| **`ExoAgent`** | The reasoning engine that routes tool calls, manages LLM context, and processes RAG sources. Used internally by `ExoModel`; also available for direct use. |
| **`ExoModelList[T]`** | Typed collection for bulk generation, updating, and export of `ExoModel` instances in a single LLM call. |
| **`@llm_function`** | Decorator that turns any method into an agentic tool, discoverable and callable by `ExoAgent` at runtime via `master_prompt`. |

---

## 🎯 Use Cases

- **🤝 Consultative Apps** — Build AI advisors that guide users through complex processes (insurance claims, financial planning) by populating structured models in real time.
- **🔌 Agentic Middleware** — Bridge human language and rigid backends. Ensure every LLM output fits your API's exact specifications before it hits the wire.
- **📊 Sales & CRM Automation** — Draft professional proposals, calculate pricing based on business rules, and update lead status autonomously.
- **🕵️ Smart Auditing & Compliance** — Create objects that read their own source contracts to populate audit fields and flag inconsistencies without manual oversight.
- **📈 Intelligent Dashboarding** — Transform raw logs or transcripts into lists of structured objects (`ExoModelList`), ready for data visualization.

---

## 🔧 Logging

ExoModel uses Python's standard `logging` module. No output is shown by default.

```python
import logging

# Show warnings and errors only (recommended for production):
logging.getLogger("exomodel").setLevel(logging.WARNING)

# Show all internal traces (useful for debugging):
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("exomodel").setLevel(logging.DEBUG)
```

---

## 🤝 Contributing

We welcome contributions! ExoModel is built by developers for developers.

1. **Fork** the project.
2. Create your **Feature Branch** (`git checkout -b feature/AmazingFeature`).
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the Apache License 2.0. See `LICENSE` for more information.
