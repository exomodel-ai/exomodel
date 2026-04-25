# 🌌 ExoModel AI
### Object-Oriented Agentic AI

**ExoModel AI** is a high-level Python framework designed to bridge the gap between **structured objects** and **Large Language Models (LLMs)**. It allows you to transform static data models into "living" entities capable of autonomous updates, self-analysis, and context-aware interactions using RAG (Retrieval-Augmented Generation).

Built on top of LangChain and Pydantic, ExoModel provides a seamless "Object-to-Prompt" interface for building AI-native applications.

📖 **Official Documentation:** [https://exomodel.ai](https://exomodel.ai)  
📦 **GitHub Repository:** [https://github.com/exomodel-ai/exomodel](https://github.com/exomodel-ai/exomodel)

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

ExoModel is currently in **Beta**. You can install it using pip:

```bash
pip install exomodel