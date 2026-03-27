"""Entrypoint script for pipeline skeleton."""
from dotenv import load_dotenv
load_dotenv()  # loads .env from current working directory by default

from content_machine import run_pipeline

if __name__ == "__main__":
    run_pipeline()