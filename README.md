# 🌌 ExoModel

**ExoModel** is a professional, high-level Python framework designed to bridge the gap between **structured data** and **Large Language Models (LLMs)**. It allows you to transform static data models into "living" entities capable of autonomous updates, self-analysis, and context-aware interactions using RAG (Retrieval-Augmented Generation).

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

ExoModel is currently in **Beta (1.0)**. Install it via pip:

```bash
pip install exomodel
```

*Ensure you have your environment variables configured: 
 - LLM Key (e.g., `GOOGLE_API_KEY` or `OPENAI_API_KEY`).*
 - MY_LLM_MODEL (e.g., `gemini-2.5-flash` or `gpt-4o`).*
 - MY_EMBEDDING_MODEL (e.g., `gemini-embedding-001` or `text-embedding-3-small`).*

---

## 🛠 Basic Usage

### 1. Define your Model
Inherit from `ExoModel` to give your data structures "AI powers."

```python
from exo_model import ExoModel
from pydantic import Field

class Proposal(ExoModel):
    client: str = Field(description="Legal name of the company")
    business_challenge: str = Field(description="The main problem to solve")
    solution: str = Field(description="AI Strategy, AI Program Execution, or AI Labs")
    pricing: int = Field(default=0)

    @classmethod
    def get_rag_sources(cls):
        return ["docs/proposal_rules.md"]
```

### 2. Interact via Prompts
You can initialize an object directly with a prompt or update it later.

```python
# Create and populate
proposal = Proposal(prompt="Create a proposal for Tesla regarding a new AI Predictive Maintenance roadmap.")

# Analyze based on RAG rules
analysis = proposal.run_analysis()
print(analysis)

# Update a specific field
proposal.update_field("pricing", "Based on the roadmap complexity, set a fair price.")
```

### 3. Manage Lists
Use `ExoModelList` to handle multiple entities.

```python
from exomodel_list import ExoModelList

tasks = ExoModelList(item_class=Proposal)
tasks.create_list("Generate 3 different AI proposals for a retail company.")

print(tasks.to_ui())
print(tasks.to_csv())
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
