import os
import io
import time
import urllib
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain.chains import LLMMathChain
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Build SQLAlchemy connection string
def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={quoted}"

# Introspect schema
def get_table_info(engine) -> str:
    try:
        with engine.connect() as conn:
            tables = [r[0] for r in conn.execute(text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
            ))]
            parts = []
            for t in tables:
                cols = conn.execute(text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                      FROM INFORMATION_SCHEMA.COLUMNS 
                     WHERE TABLE_NAME = '{t}' ORDER BY ORDINAL_POSITION
                """))
                col_lines = [
                    f"  - {c[0]} ({c[1]}{', nullable' if c[2]=='YES' else ''})"
                    for c in cols
                ]
                fks = conn.execute(text(f"""
                    SELECT COL.COLUMN_NAME, PK.TABLE_NAME, PKC.COLUMN_NAME
                      FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS RC
                      JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE COL 
                        ON RC.CONSTRAINT_NAME = COL.CONSTRAINT_NAME
                      JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK 
                        ON RC.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
                      JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE PKC 
                        ON PK.CONSTRAINT_NAME = PKC.CONSTRAINT_NAME
                     WHERE COL.TABLE_NAME = '{t}'
                """))
                for fk in fks:
                    col_lines.append(f"  - {fk[0]} → {fk[1]}({fk[2]})")
                parts.append(f"Table: {t}\n" + "\n".join(col_lines))
            return "\n\n".join(parts)
    except Exception as e:
        return f"Error fetching schema: {e}"

# Configuration
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER_NAME = ""
DATABASE_NAME = "RentalData"

# Database connection
db_url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, DATABASE_NAME)
engine = create_engine(db_url, echo=False)
schema_description = get_table_info(engine)

# LLM setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt template for SQLDatabaseChain
prompt_template = f"""
You are a T-SQL expert. Here is the database schema:

{schema_description}

If the question includes a time filter such as “last N years”:
- If the table has a DATE or DATETIME column, use: `WHERE DateColumn >= DATEADD(year, -N, GETDATE())`
- If it has only a `Year` integer column, use: `WHERE Year >= YEAR(GETDATE()) - N`
Replace `N` with the specific number mentioned by the user.
When writing SQL queries, do not wrap them in triple backticks or markdown code blocks.
Generate only valid T-SQL for the user’s question below.

User question:
{{input}}
"""
prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

# SQLDatabase and Chain
db = SQLDatabase.from_uri(db_url)
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, prompt=prompt, verbose=True)

# Plotting tool
def plot_sql(query: str, engine) -> tuple:
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None, "Query returned no rows to plot."
        plt.figure(figsize=(10, 6))
        if len(df.columns) == 2 and df.dtypes.iloc[1].kind in 'ifc':
            x_col, y_col = df.columns
            df.plot(x=x_col, y=y_col,
                    kind='bar' if df.dtypes.iloc[0].kind == 'O' else 'line',
                    legend=False)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45)
        elif len(df.columns) > 2 and df.select_dtypes(include=['number']).shape[1] > 0:
            df.select_dtypes(include=['number']).plot()
            plt.legend(loc='best')
        else:
            return None, "Cannot create a meaningful plot from this data."
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf, "✅ Plot generated successfully."
    except Exception as e:
        return None, f"Plot error: {e}"

# Tool wrapper
def plot_and_return_message(query: str) -> str:
    plot_buffer, message = plot_sql(query, engine)
    if plot_buffer:
        plot_filename = f"plot_{int(time.time())}.png"
        with open(plot_filename, "wb") as f:
            f.write(plot_buffer.read())
        return f"{message} Plot saved as `{plot_filename}`"
    return message

# Define Tools
tools = [
    Tool(
        name="SQLQueryTool",
        func=lambda q: db_chain.invoke({"query": q}),
        description="Executes natural language questions by converting them into SQL queries on the database."
    ),
    Tool(
        name="PlotTool",
        func=plot_and_return_message,
        description="Generates a plot from a valid SQL SELECT query. Use it when the user asks to visualize results."
    ),
]

# Memory setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
)

# Sample interaction
if __name__ == "__main__":
    print("Starting the DB agent. Ask a question!")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent_executor.run(user_input)
        print("\nAgent:", response)
