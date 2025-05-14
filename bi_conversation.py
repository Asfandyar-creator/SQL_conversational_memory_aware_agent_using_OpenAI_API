import os
import time
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import Tool


def plot_sql(query: str, engine) -> str:
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


def build_sqlalchemy_url(db_name: str) -> str:
    odbc_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=;"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={quoted}"


def check_database_exists(server: str, db_name: str) -> bool:
    odbc_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
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


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    server_name = "DESKTOP-QTJL0CH\\SQLEXPRESS"

    # 🔁 Keep asking for a valid DB name
    while True:
        db_name = input("Enter the database name: ")
        if check_database_exists(server_name, db_name):
            break
        else:
            print(f"❌ Error: The database '{db_name}' does not exist on the server. Please try again.\n")

    user_prompt = input("What would you like to ask about the database? ")

    # Setup engine and LangChain agent
    sqlalchemy_url = build_sqlalchemy_url(db_name)
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

    response = agent.invoke(user_prompt)
    print("\n💡 Response:")
    print(response["output"])


if __name__ == "__main__":
    main()
