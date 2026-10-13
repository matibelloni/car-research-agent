import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    row = conn.execute(
        "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    print(row)
