"""The defender's MCP tool server: one institution's own tables, as of the boundary, read-only.

The harness writes the observer's tables as of the boundary day into `--data` (Parquet, one file per table, only rows
the institution had loaded by then; never truth) and the role's decision points into `--decisions` (JSON). The server
copies the tables into an in-memory DuckDB and then turns off DuckDB's access to files, so a query can't read
anything else on the machine. `export` writes a query's result into the agent's workspace for training.

    python -m fraud_arena.toolserver --data DIR --decisions decisions.json --workspace WS [--format deploy.md]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

MAX_ROWS = 2000
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WRITE = re.compile(r"^\s*(insert|update|delete|create|drop|alter|copy|attach|detach|install|load|set|pragma|"
                    r"export|import|call|checkpoint|vacuum)\b", re.I)


class Tables:
    def __init__(self, data: Path):
        self.con = duckdb.connect(":memory:")
        self.meta = {}
        for f in sorted(Path(data).rglob("*.parquet")):
            name = f.relative_to(data).with_suffix("").as_posix().replace("/", "_").upper()
            if not _IDENT.match(name):
                continue
            self.con.execute(f'create table "{name}" as select * from read_parquet(?)', [str(f)])
            cols = self.con.execute(f'describe "{name}"').fetchall()
            n = self.con.execute(f'select count(*) from "{name}"').fetchone()[0]
            self.meta[name] = {"rows": n, "columns": [{"name": c[0], "type": c[1]} for c in cols]}
        self.con.execute("set enable_external_access = false")
        self.con.execute("set lock_configuration = true")

    def query(self, sql: str, limit: int = MAX_ROWS):
        if _WRITE.match(sql) or ";" in sql.strip().rstrip(";"):
            raise ValueError("one read-only SELECT (or WITH ... SELECT) at a time")
        rel = self.con.sql(sql)
        return rel.limit(limit).fetch_arrow_table() if limit else rel.fetch_arrow_table()


def build(data: Path, decisions: Path, workspace: Path, fmt: Path | None):
    from mcp.server.mcpserver import MCPServer

    tables = Tables(data)
    points = json.loads(Path(decisions).read_text())
    guide = Path(fmt).read_text() if fmt else "See the Runmere deployed-policy format."
    ws = Path(workspace).resolve()
    server = MCPServer("arena", instructions="Your institution's own records as of today, read-only, and its "
                                             "decision points. SQL is DuckDB.")

    @server.tool(description="The tables you can query: name, row count, columns and types.")
    def list_tables() -> dict:
        return tables.meta

    @server.tool(description=f"Run one read-only DuckDB SELECT over your tables. Returns at most {MAX_ROWS} rows "
                             "as {columns, rows}; use `export` for more.")
    def sql(query: str) -> dict:
        t = tables.query(query)
        return {"columns": t.column_names, "rows": [list(r.values()) for r in t.to_pylist()],
                "truncated": t.num_rows >= MAX_ROWS}

    @server.tool(description="Rows of a table where a column equals a value (for example a display id).")
    def lookup(table: str, column: str, value: str) -> dict:
        if table.upper() not in tables.meta or not _IDENT.match(column):
            raise ValueError(f"unknown table or bad column; tables: {sorted(tables.meta)}")
        t = tables.con.execute(f'select * from "{table.upper()}" where cast("{column}" as varchar) = ? limit 100',
                               [value]).fetch_arrow_table()
        return {"columns": t.column_names, "rows": [list(r.values()) for r in t.to_pylist()]}

    @server.tool(description="Write a SELECT's full result to a Parquet file in your workspace (for pandas, "
                             "scikit-learn, XGBoost). `path` is relative to the workspace.")
    def export(query: str, path: str) -> dict:
        out = (ws / path).resolve()
        if ws not in out.parents or out.suffix != ".parquet":
            raise ValueError("path must be a .parquet file inside your workspace")
        t = tables.query(query, limit=0)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        return {"path": str(out.relative_to(ws)), "rows": t.num_rows}

    @server.tool(description="Your role's decision points: when each is made, its actions, its default, and the "
                             "observation features your rules and models can read as obs.<feature>.")
    def decision_points() -> list:
        return points

    @server.tool(description="How to write bundle/policy.yaml and model files.")
    def deploy_format() -> str:
        return guide

    return server


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--format", type=Path)
    a = ap.parse_args(argv)
    build(a.data, a.decisions, a.workspace, a.format).run("stdio")


if __name__ == "__main__":
    main()
