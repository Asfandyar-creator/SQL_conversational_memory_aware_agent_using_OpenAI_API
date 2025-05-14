import os
import time
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import Tool

# 🔧 Set these values directly here
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER_NAME = ""
    

def plot_sql(query: str, engine) -> str:
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return "Query returned no rows, nothing to plot."

        plt.figure()
        if len(df.columns) == 2:
            x_col, y_col = df.columns
            df.plot(x=x_col, y=y_col, legend=False)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        else:
            df.plot()

        fname = f"plot_{int(time.time())}.png"
        out_path = os.path.join(os.getcwd(), fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        return out_path
    except Exception:
        return "❌ Failed to generate plot. Please ensure your query returns valid numeric data."


def create_plot_tool(engine):
    return Tool(
        name="plot_sql",
        func=lambda query: plot_sql(query, engine),
        description=(
            "Executes an SQL query and returns the path to a saved plot. "
            "Use this when the user wants a visual representation of query results. "
            "Input: a valid SQL SELECT statement. Output: filepath to a PNG."
        ),
    )


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
        query = "SELECT name FROM sys.databases"
        dbs = pd.read_sql(query, engine)
        return db_name in dbs["name"].values
    except OperationalError:
        print("❌ Could not connect to SQL Server. Please check the server name and driver.")
    except SQLAlchemyError:
        print("❌ SQLAlchemy failed to connect. Check your connection string or SQL Server status.")
    except Exception:
        print("❌ Unknown error occurred while checking the database.")
    return False


def main():
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Please check your .env file.")
        return

    # 🔁 Keep asking for a valid DB name
    while True:
        db_name = input("Enter the database name: ").strip()
        if not db_name:
            print("❌ Database name cannot be empty. Please try again.\n")
            continue
        if check_database_exists(SERVER_NAME, ODBC_DRIVER, db_name):
            break
        else:
            print(f"❌ Error: The database '{db_name}' does not exist or could not be reached.\n")

    try:
        # 🔗 Connect to database
        sqlalchemy_url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, db_name)
        engine = create_engine(sqlalchemy_url)
        db = SQLDatabase(engine, schema="dbo")

        plot_tool = create_plot_tool(engine)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )

        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            extra_tools=[plot_tool],
            verbose=False,
            top_k=10,
        )

        print("\n✅ Connected to the database!")
        print("💬 Start chatting with the database (type 'exit' to quit)\n")

        # 🔁 Keep chatting until user types "exit"
        while True:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() in {"exit", "quit"}:
                print("👋 Exiting. Goodbye!")
                break
            try:
                response = agent.invoke(user_prompt)
                print("Agent:", response["output"])
            except Exception:
                print("❌ Failed to process your request. Please try again or refine your question.")

    except OperationalError:
        print("❌ Could not connect to the specified database. Please verify server and DB name.")
    except SQLAlchemyError:
        print("❌ SQLAlchemy failed during database setup.")
    except Exception:
        print("❌ An unexpected error occurred while setting up the agent.")


if __name__ == "__main__":
    main()
