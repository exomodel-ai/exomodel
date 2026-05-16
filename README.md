# 🌌 ExoModel AI
### Your Python objects, autonomous.
### Describe what you want — the object thinks, acts, and responds.

[![PyPI version](https://img.shields.io/pypi/v/exomodel?color=blue)](https://pypi.org/project/exomodel/)
[![License](https://img.shields.io/github/license/exomodel-ai/exomodel)](https://github.com/exomodel-ai/exomodel/blob/main/LICENSE)

📖 **Official Documentation:** [https://exomodel.ai](https://exomodel.ai)  
📦 **GitHub Repository:** [https://github.com/exomodel-ai/exomodel](https://github.com/exomodel-ai/exomodel)  
📬 **Contact:** [contact@exomodel.ai](mailto:contact@exomodel.ai)

---

Your domain objects already know how to do things.  
Making them understand a natural language instruction shouldn't require  
adapters, intent routers, and prompt templates written by hand.

ExoModel bridges natural language intent and object behavior — so your domain objects can understand instructions, call their own methods, and respond with guaranteed structure. No glue code. No rewrites.

---

**Install now:** `pip install exomodel`

---

## ⚡ Why ExoModel?

| Traditional approach | With ExoModel |
| :--- | :--- |
| Write adapters and parsers before every feature | Objects understand instructions directly — no translation layer |
| Map intent to methods by hand (`if intent == X`) | Objects discover and call their own methods based on intent |
| Glue code buries your domain logic | Domain code stays domain code |
| Tied to one LLM provider | Swap providers without touching business logic |

---

## 🤔 Why not just use LangChain directly?

LangChain is great — ExoModel is actually built on top of it. But there's a gap between "I have an LLM" and "I have a working business application", and that's exactly what ExoModel fills.

**1. LangChain thinks in pipelines. Your business thinks in objects.**

When you model a `Proposal`, a `LeadContact`, or an `AuditReport`, you're thinking in entities — not chains. LangChain makes you wrap your objects around the AI. ExoModel puts the AI inside the objects, where it belongs.

**2. You still have to write all the glue code.**

With raw LangChain, you manage prompt templates, output parsers, schema validation, RAG retrieval, and tool registration — separately, by hand, for every entity. ExoModel handles all of that by convention. Define the schema, attach the documents, and the object knows what to do.

**3. The output is still just text.**

LangChain returns strings. Your application needs structured, validated, typed objects that can act — objects you can save to a database, send to an API, render in a UI, or use to trigger further methods. ExoModel's output is always a live Pydantic object — guaranteed schema, no parsing, ready to do work.

---

## 🔥 Core Features

- **🧠 Natural Language Updates** — Give objects instructions in plain English — they update their own fields with guaranteed structure. No parsers, no manual mapping.
- **📚 Native RAG Grounding** — Ground your objects in real documents — attach PDFs, URLs, or text files and the object reasons within your actual business context.
- **🤖 ExoAgent Orchestration** — Let objects coordinate without writing routing logic — ExoAgent reads intent and decides which method or tool to invoke, not `if/else`.
- **🔌 Schema-Validated Output** — Feed raw human input, get back schema-validated output — ready for your APIs, databases, or UI. No parsing step.
- **⚙️ Agentic Tools with `@llm_function`** — Expose any method as an agent tool with one decorator — ExoAgent discovers and calls it autonomously. No registration, no wiring.
- **📊 Collection Operations** — Operate on entire collections in one LLM call — generate, update, and export lists of objects without looping manually.

---

## 📦 Installation

ExoModel is LLM-agnostic. Install only the provider package you need:

```bash
pip install "exomodel[google]"      # Gemini (default)
pip install "exomodel[anthropic]"   # Claude
pip install "exomodel[openai]"      # OpenAI / Azure OpenAI
pip install "exomodel[cohere]"      # Cohere
pip install "exomodel[all]"         # all providers
```

Then create a `.env` file at the root of your project:

```env
MY_LLM_MODEL=google_genai:gemini-2.5-flash-lite
MY_EMB_MODEL=google_genai:gemini-embedding-001   # optional — auto-detected from provider
GOOGLE_API_KEY=your-key-here
```

---

## 🚀 Quick Start

Three steps. No boilerplate.

### Step 1 — Install

```bash
pip install "exomodel[google]"
```

### Step 2 — Inherit

```python
from exomodel import ExoModel

class Proposal(ExoModel):
    client: str = ""
    budget: float = 0.0
    flagged_for_legal: bool = False

    def flag_for_review(self):
        self.flagged_for_legal = True

    @classmethod
    def get_rag_sources(cls):
        return ["proposal_rules.md"]
```

### Step 3 — Instruct

```python
# The object updates fields and calls the right method — from one instruction
p = Proposal.create("Draft a 50k proposal for Tesla")
p.update("apply the 10% safety margin and flag it for legal review")
# → budget updated, flag_for_review() called — no if/else, no manual routing

print(p.to_ui())

# The object checks itself against proposal_rules.md
print(p.run_analysis())
```

The `@llm_function` decorator makes any method callable by the agent. Point it at a rules document and the object validates itself — no extra code.

```python
from exomodel import ExoModel, llm_function

class LeadContact(ExoModel):
    name: str = ""
    status: str = "new"
    score: int = 0

    @llm_function
    def qualify(self):
        """Qualify this lead based on company size and budget signals."""
        self.status = "qualified"

    @llm_function
    def disqualify(self, reason: str):
        """Mark this lead as disqualified and record why."""
        self.status = f"disqualified: {reason}"

lead = LeadContact.create("Sarah from Acme Corp, mentioned $200k budget")
lead.master_prompt("Evaluate this lead and take the right action")
# → qualify() or disqualify() called autonomously based on context
```

---

## 🛠 Architecture

| Class | Role |
| :--- | :--- |
| **`ExoModel`** | The intelligent object foundation — schema-driven, AI-powered, RAG-aware. Subclass this to define your entities. |
| **`ExoAgent`** | The reasoning engine that routes tool calls, manages LLM context, and processes RAG sources. Used internally by `ExoModel`; also available for direct use. |
| **`ExoModelList[T]`** | Typed collection for bulk generation, updating, and export of `ExoModel` instances in a single LLM call. |
| **`@llm_function`** | Decorator that turns any method into an agentic tool, discoverable and callable by `ExoAgent` at runtime via `master_prompt`. |

---

## 🎯 Use Cases

- **🤝 Consultative Apps** — Build AI advisors that guide users through complex processes (insurance claims, financial planning) by populating structured objects in real time.
- **🔌 Agentic Middleware** — Bridge human language and rigid backends. Every LLM output fits your API's exact specification before it hits the wire.
- **📊 Sales & CRM Automation** — Draft proposals, calculate pricing against business rules, and update lead status autonomously.
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

With ExoModel, your domain code stays domain code.  
Your objects do what they were always meant to do — just smarter.

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
