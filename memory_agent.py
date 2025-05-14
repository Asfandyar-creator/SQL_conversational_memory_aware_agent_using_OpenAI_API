import os
import time
import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import re
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import Tool, AgentExecutor
from langchain.agents.conversational.base import ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor

# ------------------------------------------
# 🔌 Database Connection Utilities
# ------------------------------------------
def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={quoted}"

def check_database_exists(server: str, driver: str, db_name: str) -> bool:
    try:
        odbc_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            "DATABASE=master;"
            "Trusted_Connection=yes;"
        )
        quoted = urllib.parse.quote_plus(odbc_str)
        master_url = f"mssql+pyodbc:///?odbc_connect={quoted}"

        engine = create_engine(master_url)
        with engine.connect() as conn:
            # Using the text function to properly execute SQL
            names = [row[0] for row in conn.execute(text("SELECT name FROM sys.databases"))]
            return db_name in names
    except OperationalError as e:
        print(f"❌ Could not connect to SQL Server: {e}")
    except SQLAlchemyError as e:
        print(f"❌ SQLAlchemy failed to connect: {e}")
    except Exception as e:
        print(f"❌ Unknown error occurred while checking the database: {e}")
    return False

# ------------------------------------------
# 📊 Plotting Tool With Schema Awareness
# ------------------------------------------
def plot_sql(query: str, engine) -> str:
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return "Query returned no rows, nothing to plot."

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        plt.figure()

        # Case 1: One categorical/time + one numeric → bar or line chart
        if len(numeric_cols) == 1 and len(non_numeric_cols) == 1:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]

            # Attempt to parse datetime
            try:
                df[x_col] = pd.to_datetime(df[x_col])
                df = df.sort_values(by=x_col)
                chart_type = 'line'
            except Exception:
                df = df.sort_values(by=x_col)
                chart_type = 'bar'

            if chart_type == 'line':
                plt.plot(df[x_col], df[y_col])
            else:
                df.groupby(x_col)[y_col].sum().plot(kind='bar')

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} by {x_col}")

        # Case 2: Two numeric columns → line chart with sorted X
        elif len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[:2]
            df = df.sort_values(by=x_col)
            plt.plot(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} vs {x_col}")

        # Case 3: One numeric only → histogram
        elif len(numeric_cols) == 1:
            col = numeric_cols[0]
            df[col].plot(kind='hist', bins=20)
            plt.xlabel(col)
            plt.title(f"Distribution of {col}")

        else:
            return "❌ Cannot determine a suitable plot. Please ensure the query returns at least one numeric column."

        fname = f"plot_{int(time.time())}.png"
        out_path = os.path.join(os.getcwd(), fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        return f"✅ Plot saved to: {out_path}"

    except Exception as e:
        return f"❌ Failed to generate plot: {e}"


def create_plot_tool_with_schema(engine, db: SQLDatabase):
    def plotting_function(query: str) -> str:
        try:
            return plot_sql(query, engine)
        except Exception as e:
            return f"Plotting failed: {e}"

    return Tool(
        name="plot_sql",
        func=plotting_function,
        description=(
            "Use this tool to generate clean, accurate charts from SQL query results. "
            "Before using it, ensure the query returns a clear X-axis (e.g., years, months, categories) "
            "and a numeric Y-axis (e.g., average, count, totals). "
            "Sort the results by the X-axis if it's time-based (e.g., Year). "
            "Do NOT use this tool if the data has duplicate X values, missing values, or mixed data types. "
            "Avoid plotting identifiers or timestamps directly unless explicitly requested. "
            "Bar charts are good for categories, line charts for trends over time, and pie charts for proportions."
            "Use only the following database schema (tables and columns) when forming queries:\n\n"
            f"{db.table_info}"
        )
    )

# ------------------------------------------
# 🚀 Main Agent Setup
# ------------------------------------------
def launch_conversational_agent(server: str, driver: str, db_name: str):
    try:
        # Database existence is already checked before calling this function
        db_uri = build_sqlalchemy_url(server, driver, db_name)
        engine = create_engine(db_uri)
        db = SQLDatabase(engine)

        print(f"🚀 Initializing AI assistant for database '{db_name}'...")
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatOpenAI(temperature=0, model="gpt-4", verbose=True)

        # Tools
        sql_chain = SQLDatabaseChain.from_llm(llm, db, memory=memory)

        def clean_sql_and_run(query: str) -> str:
            # Remove Markdown formatting like ```sql
            cleaned_query = re.sub(r"```sql|```", "", query, flags=re.IGNORECASE).strip()

            # Split multiple queries by semicolon
            queries = [q.strip() for q in cleaned_query.split(';') if q.strip()]

            results = []
            for sql in queries:
                try:
                    result = sql_chain.invoke(sql)

                    # Convert list of dicts to string for clean output
                    if isinstance(result, list):
                        str_result = "\n".join(str(row) for row in result)
                    else:
                        str_result = str(result)

                    results.append(str_result)

                except Exception as e:
                    results.append(f"❌ Failed to execute: {sql}\nError: {e}")

            return "\n\n".join(results)



        sql_tool = Tool(    
            name="SQL Tool",
            func=clean_sql_and_run,
            description="Generate only a valid SQL query without any explanation or formatting. Do not include markdown or ``` or extra text."
        )

        plot_tool = create_plot_tool_with_schema(engine, db)

        tools = [sql_tool, plot_tool]

        agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=True)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

        print("\n✅ Ready to chat! Type 'exit' or 'quit' to end the session.")
        
        # Start the conversation
        while True:
            try:
                user_input = input("\n🧑‍💻 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Exiting.")
                    break
                response = agent_executor.invoke(user_input)
                print(f"\n🤖 Agent: {response['output']}")
            except Exception as e:
                print(f"❌ Error occurred during the conversation: {e}")
                print("⚠️ Continuing session...")

    except Exception as e:
        print(f"❌ Error setting up the agent: {e}")
        print("Please restart the application and try again.")

# ------------------------------------------
# 🏁 Launch
# ------------------------------------------
if __name__ == "__main__":
    print("🔌 SQL Server Connection Setup")
    print("==============================")
    
    # Fixed server and driver settings (can be modified if needed)
    SERVER = ""
    DRIVER = "ODBC Driver 17 for SQL Server"
    
    # Keep asking for database until a valid one is provided
    database_valid = False
    
    while not database_valid:
        DATABASE = input("Enter database name: ")
        
        print(f"\n🔍 Checking connection to database '{DATABASE}' on server '{SERVER}'...")
        
        if check_database_exists(SERVER, DRIVER, DATABASE):
            print(f"✅ Database '{DATABASE}' found. Connecting...")
            database_valid = True
        else:
            print(f"❌ Database '{DATABASE}' not found on server '{SERVER}'")
            print("Please try again with a valid database name.")
    
    # Now that we have a valid database, launch the agent
    launch_conversational_agent(SERVER, DRIVER, DATABASE)
