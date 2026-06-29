# 🚀 Valura AI – Intelligent Financial Co Investor

An AI powered financial assistant microservice that intelligently understands user investment queries, routes them to specialized agents, and streams responses in real time using **Server Sent Events (SSE)**.

The project is designed with a modular agent based architecture that combines **LLM powered intent classification**, **rule based safety guardrails**, and **portfolio analytics** to deliver secure, scalable, and production ready AI experiences.

---

# ✨ Features

* 🛡️ Regex based Safety Guard (sub-10ms)
* 🤖 LLM Intent Classification
* 🧠 Conversation Memory
* 📡 Real time SSE Streaming
* 📈 Portfolio Health Analysis
* 🔀 Modular Agent Router
* ⚡ FastAPI Async Backend
* ✅ Structured JSON Responses
* 🧪 Comprehensive Test Suite
* 🔧 Easily Extensible Architecture

---

# 🏗️ Architecture

```
                 POST /query
                      │
                      ▼
            Safety Guard (Regex)
                      │
          Blocked? ───┤──► Return Error
                      │
                      ▼
            Session Memory Store
                      │
                      ▼
          LLM Intent Classifier
                      │
                      ▼
              Agent Router
      ┌──────────┼──────────┐
      ▼          ▼          ▼
 Portfolio   Market      Strategy
  Health    Research     Planning
      │
      ▼
  SSE Streaming Response
```

---

# 📂 Project Structure

```
valura-ai/
│
├── src/
│   ├── agents/
│   ├── core/
│   ├── main.py
│
├── fixtures/
│
├── tests/
│
├── README.md
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

# ⚙️ Tech Stack

### Backend

* FastAPI
* Python 3.11+
* OpenAI SDK
* Pydantic v2
* AsyncIO

### AI

* OpenAI GPT Models
* Structured Outputs
* Intent Classification
* Prompt Engineering

### Finance

* yfinance
* Portfolio Analytics

### Testing

* Pytest
* Pytest-AsyncIO

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/yourusername/valura-ai.git
cd valura-ai
```

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment

```bash
cp .env.example .env
```

Update:

```
OPENAI_API_KEY=your_api_key
```

---

# ▶️ Run the Application

```bash
uvicorn src.main:app --reload --port 8000
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# 🧪 Run Tests

```bash
pytest tests/ -v
```

---

# 📡 API Endpoints

## POST `/query`

Processes financial queries and streams AI-generated responses.

Example request:

```json
{
  "query": "How is my portfolio performing?",
  "user_context": {
    "user_id": "user001"
  }
}
```

Response:

* Safety Validation
* Metadata
* Portfolio Analysis
* AI Summary
* Completion Event

---

## GET `/health`

Returns service health status including:

* Running Status
* Active Sessions
* Current Model

---

# 🔐 Safety Layer

The first stage of the pipeline is a lightweight regex-based safety guard.

It blocks:

* Prompt Injection
* Harmful Requests
* Jailbreak Attempts
* Unsafe Inputs

Since it does not require an LLM call, the average execution time remains below **10 milliseconds**.

---

# 🧠 Intent Classification

A single OpenAI structured-output call identifies:

* Intent
* Target Agent
* Extracted Entities
* Safety Verdict

If classification fails, the request safely falls back to the Support agent.

---

# 📈 Portfolio Health Agent

The fully implemented Portfolio Health agent performs:

* Portfolio Performance Analysis
* Concentration Risk Detection
* Benchmark Comparison
* Investment Observations
* Personalized Financial Summary

---

# 🛣️ Future Improvements

* Redis Session Storage
* Market Research Agent
* Investment Strategy Agent
* Predictive Analytics
* RAG based Financial Knowledge
* Authentication & Authorization
* Docker Deployment
* CI/CD Pipeline
* Monitoring & Logging

---

# 📊 Performance Goals

| Metric       | Target        |
| ------------ | ------------- |
| Safety Guard | <10 ms        |
| First Token  | <2 sec        |
| End-to-End   | <6 sec        |
| Streaming    | Real-time SSE |




