import os
import urllib
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
 


load_dotenv()

# 1) Build the SQLAlchemy URL
def build_sqlalchemy_url(server: str, driver: str, db_name: str) -> str:
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={quoted}"

# 2) Introspect schema: list tables, columns, and FKs
def get_table_info(engine) -> str:
    try:
        with engine.connect() as conn:
            tables = [r[0] for r in conn.execute(text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
            ))]
            parts = []
            for t in tables:
                # columns
                cols = conn.execute(text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                      FROM INFORMATION_SCHEMA.COLUMNS 
                     WHERE TABLE_NAME = '{t}' ORDER BY ORDINAL_POSITION
                """))
                col_lines = [
                    f"  - {c[0]} ({c[1]}{', nullable' if c[2]=='YES' else ''})"
                    for c in cols
                ]
                # foreign keys
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
                except Exception:
                    pass

                parts.append(f"Table: {t}\n" + "\n".join(col_lines))
            return "\n\n".join(parts)
    except Exception as e:
        return f"Error fetching schema: {e}"

# 3) Configuration
ODBC_DRIVER   = "ODBC Driver 17 for SQL Server"
SERVER_NAME   = ""
DATABASE_NAME = "RentalData"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 4) Create URLs, engine, and inspect schema

db_url = build_sqlalchemy_url(SERVER_NAME, ODBC_DRIVER, DATABASE_NAME)
engine = create_engine(db_url, echo=False)
schema_description = get_table_info(engine)

# 5) Build LangChain objects
# Inject the schema into the system prompt for context
full_template = f"""
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
prompt = PromptTemplate(input_variables=["input"], template=full_template)

# Create the SQLDatabase and chain
# Note: SQLDatabase.from_uri will create its own engine, but it points to the same database

db = SQLDatabase.from_uri(db_url)
llm = ChatOpenAI(model='gpt-4o', temperature=0)

db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=prompt,
    verbose=True,
)

# 6) Execute a sample query
if __name__ == "__main__":
    question = "give me most expensive rentals"
    result = db_chain.invoke({"query": question})
    print("Answer:", result)    