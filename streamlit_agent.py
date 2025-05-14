import os
import io
import time
import urllib.parse
import pandas as pd
import streamlit as st
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

# ------------------------------------------
# 🔌 Database Connection Utilities
# ------------------------------------------
def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    """Build SQLAlchemy connection URL for SQL Server"""
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={quoted}"

def check_database_exists(server: str, driver: str, db_name: str) -> bool:
    """Check if a database exists on the server"""
    engine = None
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
            names = [row[0] for row in conn.execute(text("SELECT name FROM sys.databases"))]
            return db_name in names
    except OperationalError as e:
        st.sidebar.error(f"❌ Could not connect to SQL Server: {e}")
    except SQLAlchemyError as e:
        st.sidebar.error(f"❌ SQLAlchemy failed to connect: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Unknown error occurred: {e}")
    finally:
        if engine:
            engine.dispose()
    return False

def get_table_info(engine) -> str:
    """Get database schema information"""
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
                # Get foreign keys
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

# ------------------------------------------
# 📊 Plotting Tool With Schema Awareness
# ------------------------------------------
def plot_sql(query: str, engine) -> tuple:
    """Generate a plot from SQL query results and return image buffer"""
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None, "Query returned no rows, nothing to plot."

        plt.figure(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        # Case 1: One categorical/time + one numeric → bar or line chart
        if len(numeric_cols) == 1 and len(non_numeric_cols) == 1:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]

            # Attempt to parse datetime
            try:
                df[x_col] = pd.to_datetime(df[x_col])
                df = df.sort_values(by=x_col)
                plt.plot(df[x_col], df[y_col])
            except Exception:
                df = df.sort_values(by=x_col)
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
            return None, "❌ Cannot determine a suitable plot. Please ensure the query returns at least one numeric column."

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf, "Here's the visualization of your query results:"
        
    except Exception as e:
        return None, f"❌ Failed to generate plot: {str(e)}"

def create_plot_tool_with_schema(engine, db):
    """Create a plotting tool for the agent"""
    def plot_and_save(query):
        try:
            plot_buffer, message = plot_sql(query, engine)
            if plot_buffer:
                # Generate a unique ID for the plot
                plot_id = str(int(time.time()))
                
                # Store the plot buffer in session state
                if "plot_data" not in st.session_state:
                    st.session_state.plot_data = {}
                
                st.session_state.plot_data[plot_id] = plot_buffer.getvalue()
                
                # Set latest plot ID
                if plot_id in st.session_state.plot_data:
                    st.session_state.latest_plot_id = plot_id
                    # Use a hidden marker that will be completely removed later
                    return f"PLOT_CREATED:{plot_id} - "
                else:
                    return "Warning: Plot was created but not stored properly."
            return message
        except Exception as e:
            return f"Error in plot creation: {str(e)}"

    return Tool(
        name="plot_sql",
        func=plot_and_save,
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

def disconnect_database():
    """Disconnect from the current database and reset necessary session states"""
    if st.session_state.get("db_connected"):
        # Clear database connection session states
        if "engine" in st.session_state:
            # Close the connection if it's open
            try:
                st.session_state.engine.dispose()
            except:
                pass
            del st.session_state.engine
        
        if "db" in st.session_state:
            del st.session_state.db
            
        if "db_schema" in st.session_state:
            del st.session_state.db_schema
            
        # Reset connection flag
        st.session_state.db_connected = False
        
        # Clear chat history
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        return True
    return False

def ensure_session_state_initialized():
    """Ensure all session state variables are initialized properly"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "db_connected" not in st.session_state:
        st.session_state.db_connected = False
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = {}
    if "latest_plot_id" not in st.session_state:
        st.session_state.latest_plot_id = None
    if "current_db" not in st.session_state:
        st.session_state.current_db = None

# --- Streamlit UI setup ---
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER_NAME = ""

st.set_page_config(page_title="SQL Chatbot", layout="wide")

# Initialize all session state variables
ensure_session_state_initialized()

if st.session_state.get("db_connected"):
    st.title("SQL Conversational Agent 🤖")
else:
    st.title("💬 Let's conversate with your database!")

# Check for OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key

with st.sidebar:
    st.subheader("Database Connection")
    
    # Show current database if connected
    if st.session_state.get("db_connected") and st.session_state.get("current_db"):
        st.success(f"✅ Connected to: {st.session_state.current_db}")
        
        # Add disconnect button
        if st.button("Disconnect"):
            if disconnect_database():
                st.success("Successfully disconnected from database")
                st.rerun()
    
    # Show connection form if not connected
    if not st.session_state.get("db_connected"):
        db_name = st.text_input("Database Name")
        if st.button("Connect"):
            if not db_name:
                st.error("Please enter a database name")
            elif not check_database_exists(SERVER_NAME, ODBC_DRIVER, db_name):
                st.error("❌ Database not found")
            else:
                url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, db_name)
                try:
                    eng = create_engine(url)
                    with eng.connect() as c: c.execute(text("SELECT 1"))
                    st.session_state.engine = eng
                    st.session_state.db = SQLDatabase(eng, schema="dbo")
                    st.session_state.db_connected = True
                    st.session_state.db_schema = get_table_info(eng)
                    st.session_state.current_db = db_name
                    st.success(f"✅ Connected to {db_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")

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
                except Exception:
                    # Silently handle plot display errors without interrupting the session
                    pass

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
                    db = st.session_state.db
                    
                    # Create LLM
                    llm = ChatOpenAI(
                        model="gpt-4",
                        temperature=0,
                        openai_api_key=openai_api_key
                    )
                    
                    # Create SQL Chain
                    sql_chain = SQLDatabaseChain.from_llm(
                        llm=llm,
                        db=db
                    )
                    
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
                    
                    # SQL Tool
                    sql_tool = Tool(    
                        name="SQL Tool",
                        func=clean_sql_and_run,
                        description="Generate only a valid SQL query without any explanation or formatting. Do not include markdown or ``` or extra text."
                    )
                    
                    # Plot Tool
                    plot_tool = create_plot_tool_with_schema(engine, db)
                    
                    tools = [sql_tool, plot_tool]
                    
                    # Create memory, agent, and executor
                    memory = st.session_state.memory
                    
                    agent = ConversationalAgent.from_llm_and_tools(
                        llm=llm,
                        tools=tools
                    )
                    
                    agent_executor = AgentExecutor(
                        agent=agent,
                        tools=tools,
                        memory=memory
                    )
                    
                    # Execute the agent
                    result = agent_executor.invoke({"input": query})
                    out = result.get("output", "I couldn't generate a response.")
                    
                    # Check for plot ID in the output
                    plot_id = None
                    if "PLOT_CREATED:" in out:
                        match = re.search(r"PLOT_CREATED:(\d+)", out)
                        if match:
                            plot_id = match.group(1)
                            # Clean up the output - completely remove any text containing the plot ID
                            out = re.sub(
                                r"PLOT_CREATED:\d+.*?(\.|\n|$)",
                                "",
                                out
                            ).strip()
                            # Replace generic plot creation messages with a consistent message
                            out = re.sub(
                                r"I've generated a plot.*?\..*?|I have created a plot.*?\..*?|You can view the plot.*?\..*?|The plot has been generated.*?\..*?",
                                "Here's the visualization of your query results:\n\n",
                                out
                            ).strip()
                    
                    # If we don't find the ID in the output but we've stored one in session state
                    if not plot_id and st.session_state.latest_plot_id:
                        plot_id = st.session_state.latest_plot_id
                        st.session_state.latest_plot_id = None  # Reset after use
                    
                    # Display the response text
                    st.markdown(out)
                    
                    # Display plot if available with error handling
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
                        except Exception:
                            # Silently handle plot display errors
                            pass
                    
                    # Add to chat history
                    st.session_state.chat_history.append(("assistant", out, plot_id))
                    
                except Exception as e:
                    err = f"I encountered an issue processing your request: {str(e)}\nPlease try again or try a different query."
                    st.error(err)
                    st.session_state.chat_history.append(("assistant", err, None))
else:
    st.info("👈 Please connect to a database first.")