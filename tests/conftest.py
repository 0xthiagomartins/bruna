import os, sys
import pytest
from dotenv import load_dotenv

print(f'Load .env: {load_dotenv(dotenv_path="./resources/.env")}', flush=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
