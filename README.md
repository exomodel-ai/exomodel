# 🌌 ExoModel AI
### Object-Oriented Agentic AI

**ExoModel AI** is a high-level Python framework designed to bridge the gap between **structured objects** and **Large Language Models (LLMs)**. It allows you to transform static data models into "living" entities capable of autonomous updates, self-analysis, and context-aware interactions using RAG (Retrieval-Augmented Generation).

Built on top of LangChain and Pydantic, ExoModel provides a seamless "Object-to-Prompt" interface for building AI-native applications.

---

## 🚀 Key Features

* **Smart CRUD:** Create, Read, and Update your data models directly through natural language prompts.
* **Built-in RAG Support:** Attach URLs, PDFs, or Text files to your models to ground AI responses in specific business rules.
* **ExoAgent Orchestration:** A specialized agent (ExoAgent) that manages tool routing and persona switching (Generalist, Specialist, Hybrid).
* **Structured Output:** Guarantees that LLM responses strictly follow your Pydantic schemas.
* **Fluent List Management:** Handle collections of entities with `ExoModelList`, enabling bulk generation and CSV exports.
* **UI-Ready:** Built-in `to_ui()` methods optimized for Telegram, WhatsApp bots, and CLI displays.

---

## 📦 Installation

ExoModel is currently in **Beta (0.1.0)**. You can install it using pip:

```bash
pip install exomodel
```

*Ensure you have your environment variables configured in the .env file: 
 - LLM Key (e.g., `GOOGLE_API_KEY` or `OPENAI_API_KEY`).*
 - MY_LLM_MODEL (e.g., `gemini-2.5-flash` or `gpt-4o`).*
 - MY_EMBEDDING_MODEL (e.g., `gemini-embedding-001` or `text-embedding-3-small`).*

---

## 🚀 Quick Start

### 1. Setup your Environment
Install the package. 

```bash
pip install exomodel
```

Configure your API keys in a `.env` file at the root of your project.

```text
# .env
GOOGLE_API_KEY=your_key_here
MY_LLM_MODEL=gemini-1.5-flash
MY_EMBEDDING_MODEL=text-embedding-004
```

### 2\. Create a Knowledge Base

Create a file named `proposal_rules.md`. This "grounds" your AI objects in real-world logic.

```markdown
# Proposal Rules
- We only accept projects above $10,000.
- Every proposal must include a 10% safety margin in the pricing.
- We do not work with companies in the tobacco industry.
```

### 3\. Define and Run your Entity

Inherit from `ExoModel` to give your data structures autonomous reasoning powers.

```python hl_lines="3 8 11"
from exomodel import ExoModel

class Proposal(ExoModel):
    client: str = ""
    budget: float = 0.0
    
    @classmethod
    def get_rag_sources(cls):
        # The object now 'knows' your specific business rules
        return ["proposal_rules.md"]

# Initialize and populate the object from raw text
p = Proposal(prompt="Draft a 50k proposal for Tesla")

# print the object
print(p.to_ui())

# The object analyzes itself based on the 'proposal_rules.md'
print(p.run_analysis()) 

```

---

## 📂 Project Structure

```text
src/exomodel/
├── exoagent.py          # The core LLM & RAG handler
├── exomodel.py          # Base ExoModel class (Pydantic + AI)
├── exomodel_list.py      # Generic list management for ExoModels
├── prompt/               # ExoModel prompt templates
└── docs/                 # Documentation
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request or open an Issue.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the Apache License 2.0. See `LICENSE` for more information.
