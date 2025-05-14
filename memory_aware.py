import os
import io
import re
import time
import urllib.parse
from langchain.memory import ConversationBufferMemory
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# LangChain imports
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase

# Load environment
load_dotenv()

# --- Utility Functions ---
def plot_sql(query: str, engine) -> tuple:
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None, "Query returned no rows, nothing to plot."
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
            return None, "❌ Cannot create a meaningful plot with the provided data."
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf, "Here's the visualization of your query results:"
    except Exception as e:
        return None, f"❌ Failed to generate plot: {str(e)}"


def create_plot_tool(engine):
    def plot_and_save(query: str) -> str:
        plot_buffer, message = plot_sql(query, engine)
        if plot_buffer:
            plot_id = str(int(time.time()))
            st.session_state.plot_data[plot_id] = plot_buffer.getvalue()
            st.session_state.latest_plot_id = plot_id
            return f"PLOT_CREATED:{plot_id}"
        return message

    return Tool(
        name="plot_sql",
        func=plot_and_save,
        description="Executes an SQL SELECT query and returns a plot image. Returns PLOT_CREATED:<id> on success.",
    )


def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    odbc = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db_name};Trusted_Connection=yes;"
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc)}"


def check_database_exists(server: str, driver: str, db_name: str) -> bool:
    try:
        odbc = f"DRIVER={{{driver}}};SERVER={server};DATABASE=master;Trusted_Connection=yes;"
        url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc)}"
        eng = create_engine(url)
        with eng.connect() as conn:
            names = [row[0] for row in conn.execute(text("SELECT name FROM sys.databases"))]
            return db_name in names
    except Exception as e:
        st.sidebar.error(f"DB check error: {e}")
        return False


def get_table_info(engine) -> str:
    try:
        with engine.connect() as conn:
            tables = [r[0] for r in conn.execute(text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"))]
            parts = []
            for t in tables:
                cols = conn.execute(text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                      FROM INFORMATION_SCHEMA.COLUMNS 
                     WHERE TABLE_NAME = '{t}' ORDER BY ORDINAL_POSITION
                """))
                col_lines = [f"  - {c[0]} ({c[1]}{', nullable' if c[2]=='YES' else ''})" for c in cols]
                parts.append(f"Table: {t}\n" + "\n".join(col_lines))
            return "\n\n".join(parts)
    except Exception as e:
        return f"Error fetching schema: {e}"

# --- Streamlit UI setup ---
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER_NAME = ""

st.set_page_config(page_title="SQL Chatbot", layout="wide")

# Initialize session state
for key, default in {
    "chat_history": [],
    "db_connected": False,
    "plot_data": {},
    "latest_plot_id": None,
}.items():
    st.session_state.setdefault(key, default)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ No OpenAI key found in .env")
    st.stop()

# Sidebar: DB connection
with st.sidebar:
    st.subheader("Connect to DB")
    db_name = st.text_input("Database Name")
    if st.button("Connect"):
        if not check_database_exists(SERVER_NAME, ODBC_DRIVER, db_name):
            st.error("❌ Database not found")
        else:
            url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, db_name)
            eng = create_engine(url)
            with eng.connect(): pass
            st.session_state.engine = eng
            st.session_state.db_connected = True
            st.session_state.db_schema = get_table_info(eng)
            # Reset memory on new DB
            st.session_state.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
            st.success(f"✅ Connected to {db_name}")
            st.rerun()

# Initialize agent once after connection
if st.session_state.db_connected and "agent" not in st.session_state:
    system_msg = f"""
You are a SQL assistant. The schema is:
{st.session_state.db_schema}

INSTRUCTIONS:
- Use plot_sql for visualizations.
- Do not mention plot IDs; system handles display.
- Recall previous queries for follow-ups.
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_msg),
        MessagesPlaceholder(variable_name="memory"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    engine = st.session_state.engine
    db = SQLDatabase(engine, schema="dbo")
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key))
    sql_tools = toolkit.get_tools()
    plot_tool = create_plot_tool(engine)
    tools = sql_tools + [plot_tool]

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=True,
    )
    st.session_state.agent = agent

# Main chat
if st.session_state.db_connected:
    st.title("SQL Chatbot 🤖")
    for idx, (role, text, plot_id) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(text)
            if plot_id and plot_id in st.session_state.plot_data:
                st.image(st.session_state.plot_data[plot_id], caption="Query Visualization")
                st.download_button(
                    "Download Plot",
                    data=st.session_state.plot_data[plot_id],
                    file_name=f"plot_{plot_id}.png",
                    mime="image/png",
                    key=f"dl_{idx}"
                )

    query = st.chat_input("Ask me about your database:")
    if query:
        st.session_state.chat_history.append(("user", query, None))
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(input=query)
                    # Save to memory explicitly
                    st.session_state.memory.save_context(
                        {"input": query}, {"output": response}
                    )

                    plot_id = None
                    if match := re.search(r"PLOT_CREATED:(\d+)", response):
                        plot_id = match.group(1)
                        response = "Here's the visualization of your query results:"

                    st.markdown(response)
                    if plot_id:
                        st.image(st.session_state.plot_data[plot_id], caption="Query Visualization")
                        st.download_button(
                            "Download Plot",
                            data=st.session_state.plot_data[plot_id],
                            file_name=f"plot_{plot_id}.png",
                            mime="image/png",
                            key=f"dl_plot_{plot_id}"
                        )

                    st.session_state.chat_history.append(("assistant", response, plot_id))
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append(("assistant", err, None))
else:
    st.info("👈 Please connect to a database first.")

# Debugging sidebar
with st.sidebar:
    if st.session_state.db_connected:
        if st.button("Debug Plot Storage"):
            st.write(f"Plots in memory: {len(st.session_state.plot_data)}")
        if st.button("Debug Memory"):
            st.write(st.session_state.memory.load_memory_variables({}))
        if st.button("Clear Conversation Memory"):
            st.session_state.memory.clear()
            st.success("Memory cleared!")
