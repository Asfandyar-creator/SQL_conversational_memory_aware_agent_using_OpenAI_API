from dotenv import load_dotenv

load_dotenv()

import os

server_name = os.getenv("SERVER_NAME")
driver_name = os.getenv("ODBC_DRIVER")

print(f"SERVER: {server_name}")
print(f"DRIVER: {driver_name}")
