import os
import io
import re
import time
import urllib.parse
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

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
    def plot_and_save(query):
        plot_buffer, message = plot_sql(query, engine)
        if plot_buffer:
            # Generate a unique ID for the plot
            plot_id = str(int(time.time()))
            
            # Store the plot buffer in session state
            if "plot_data" not in st.session_state:
                st.session_state.plot_data = {}
            
            st.session_state.plot_data[plot_id] = plot_buffer.getvalue()
            
            # Debug output
            if plot_id in st.session_state.plot_data:
                st.session_state.latest_plot_id = plot_id
                return f"PLOT_CREATED:{plot_id} - The plot has been generated successfully."
            else:
                return "Warning: Plot was created but not stored properly."
        return message

    return Tool(
        name="plot_sql",
        func=plot_and_save,
        description="Executes an SQL query and returns a visualization. Input: valid SQL SELECT query. Use this when you need to create a visual representation of data.",
    )

def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    odbc = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db_name};Trusted_Connection=yes;"
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc)}"

def check_database_exists(server, driver, db_name) -> bool:
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
                col_lines = [f"  - {c[0]} ({c[1]}{', nullable' if c[2]=='YES' else ''})"
                             for c in cols]
                # fks
                try:
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
                except:
                    pass
                parts.append(f"Table: {t}\n" + "\n".join(col_lines))
            return "\n\n".join(parts)
    except Exception as e:
        return f"Error fetching schema: {e}"

# --- Streamlit UI setup ---
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER_NAME = ""

st.set_page_config(page_title="SQL Chatbot", layout="wide")

if st.session_state.get("db_connected"):
    st.title("SQL Chatbot 🤖")
else:
    st.title("💬 Let's conversate with your database")


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ No OpenAI key found in .env")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "plot_data" not in st.session_state:
    st.session_state.plot_data = {}
if "latest_plot_id" not in st.session_state:
    st.session_state.latest_plot_id = None

with st.sidebar:
    st.subheader("Connect to DB")
    db_name = st.text_input("Database Name")
    if st.button("Connect"):
        if not check_database_exists(SERVER_NAME, ODBC_DRIVER, db_name):
            st.error("❌ Database not found")
        else:
            url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, db_name)
            eng = create_engine(url)
            with eng.connect() as c: c.execute(text("SELECT 1"))
            st.session_state.engine = eng
            st.session_state.db_connected = True
            st.session_state.db_schema = get_table_info(eng)
            st.success(f"✅ Connected to {db_name}")
            st.rerun()

# Main chat
if st.session_state.get("db_connected"):
    # Show chat history
    for idx, (role, text, plot_id) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(text)
            if plot_id and plot_id in st.session_state.plot_data:
                try:
                    st.image(st.session_state.plot_data[plot_id], caption="Query Visualization")
                    st.download_button(
                        "Download Plot",
                        data=st.session_state.plot_data[plot_id],
                        file_name=f"plot_{plot_id}.png",
                        mime="image/png",
                        key=f"dl_{idx}"
                    )
                except Exception as e:
                    st.error(f"Error displaying plot: {e}")

    query = st.chat_input("Ask me about your database:")
    if query:
        # Add user message to history
        st.session_state.chat_history.append(("user", query, None))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    engine = st.session_state.engine
                    db = SQLDatabase(engine, schema="dbo")
                    llm = ChatOpenAI(
                        model="gpt-4o",
                        temperature=0,
                        openai_api_key=openai_api_key
                    )
                    plot_tool = create_plot_tool(engine)

                    system_message = f"""
You are a SQL assistant. The schema is:
{st.session_state.db_schema}

INSTRUCTIONS FOR VISUALIZATIONS:
1. When asked to create a plot or visualization, ALWAYS use the plot_sql tool with an appropriate SQL query
2. NEVER mention any plot ID in your response
3. After generating a plot, simply say "Here's the visualization of your query results:" followed by any analysis
4. DO NOT suggest that the user needs to view a plot with any ID number
5. DO NOT tell the user that you have created a plot that they can view above - the system will handle this automatically
"""
                    agent = create_sql_agent(
                        llm=llm,
                        db=db,
                        agent_type="tool-calling",
                        extra_tools=[plot_tool],
                        verbose=True,
                        top_k=10,
                        system_message=system_message
                    )
                    
                    result = agent.invoke(query)
                    out = result["output"]
                    
                    # Check for plot ID in the output
                    plot_id = None
                    if "PLOT_CREATED:" in out:
                        match = re.search(r"PLOT_CREATED:(\d+)", out)
                        if match:
                            plot_id = match.group(1)
                            # Clean up the output
                            out = re.sub(
                                r"PLOT_CREATED:\d+.*?\..*?successfully\.|.*?visualization above.*?\.|\s*(?:I have created a plot|You can view the plot).*?\.",
                                "Here's the visualization of your query results:",
                                out
                            ).strip()
                    
                    # If we don't find the ID in the output but we've stored one in session state
                    if not plot_id and st.session_state.latest_plot_id:
                        plot_id = st.session_state.latest_plot_id
                        st.session_state.latest_plot_id = None  # Reset after use
                    
                    # Display the response text
                    st.markdown(out)
                    
                    # Display plot if available
                    if plot_id and plot_id in st.session_state.plot_data:
                        try:
                            st.image(st.session_state.plot_data[plot_id], caption="Query Visualization")
                            st.download_button(
                                "Download Plot",
                                data=st.session_state.plot_data[plot_id],
                                file_name=f"plot_{plot_id}.png",
                                mime="image/png",
                                key=f"dl_plot_{plot_id}"
                            )
                        except Exception as e:
                            st.error(f"Error displaying plot: {e}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append(("assistant", out, plot_id))

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append(("assistant", err, None))
else:
    st.info("👈 Please connect to a database first.")

# Add a debugging section if needed
with st.sidebar:
    if st.session_state.get("db_connected"):
        if st.button("Debug Plot Storage"):
            st.write(f"Plots in memory: {len(st.session_state.plot_data)}")
            st.write(f"Latest plot ID: {st.session_state.latest_plot_id}")
            for pid, data in st.session_state.plot_data.items():
                st.write(f"Plot {pid}: {len(data)} bytes")
