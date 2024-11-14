import os
from dotenv import load_dotenv


ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
print(f"load env: {load_dotenv(dotenv_path=f"./resources/.env")}")

SQL_URI = os.environ.get("SQL_URI", None)
