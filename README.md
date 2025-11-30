# 🤖 GenAI SQL Chatbot with Visualization

A powerful Streamlit application that allows users to interact with their **Microsoft SQL Server** database using natural language. Powered by **LangChain** and **OpenAI (GPT-4o)**, this chatbot can answer data questions, generate SQL queries on the fly, and automatically create data visualizations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)

## 🌟 Features

*   **Natural Language to SQL:** Ask questions in plain English (e.g., "Show me the top 5 sales by region") and get accurate results.
*   **Automatic Visualization:** The agent detects when a visual representation is needed and generates Bar charts or Line charts using Matplotlib.
*   **Schema Awareness:** The bot automatically fetches table names, columns, and foreign keys to understand relationships within your database.
*   **Interactive Chat Interface:** Maintains chat history and context.
*   **Plot Downloads:** Generated charts can be downloaded as PNG files directly from the chat window.
*   **Secure Connection:** Uses Windows Authentication (Trusted Connection) for SQL Server.

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Microsoft SQL Server** (Local or Remote)
3.  **ODBC Driver 17 for SQL Server**
    *   [Download for Windows](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
    *   *Note: If you use a different driver version, update the `ODBC_DRIVER` variable in the code.*

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sql-chatbot-vis.git
    cd sql-chatbot-vis
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file (or use the one below) and install:
    ```bash
    pip install -r requirements.txt
    ```

    **Suggested `requirements.txt` content:**
    ```text
    streamlit
    pandas
    matplotlib
    python-dotenv
    langchain
    langchain-openai
    langchain-community
    sqlalchemy
    pyodbc
    ```

## ⚙️ Configuration

### 1. Environment Variables
Create a `.env` file in the root directory of the project and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
