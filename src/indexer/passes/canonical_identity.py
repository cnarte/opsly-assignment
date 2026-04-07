"""Pass 4: Compute qualified_name and source_hash for all entities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter


async def run(
    writer: Neo4jWriter,
    repo_path: str,
    repo_id: str,
    commit_sha: str = "",
) -> dict[str, Any]:
    """Set qualified_name and source_hash on Class/Function/Method nodes."""
    root = Path(repo_path)
    updated = 0

    # Fetch entities that need qualified names
    entities = await writer._client.execute_query(
        "MATCH (e {repo_id: $repo_id}) "
        "WHERE (e:Class OR e:Function OR e:Method) "
        "RETURN e.symbol_id AS sid, e.name AS name, "
        "       e.scope_chain AS scope, e.file_path AS fp, "
        "       e.start_line AS sl, e.end_line AS el",
        {"repo_id": repo_id},
    )

    for ent in entities:
        fp = ent.get("fp", "")
        scope = ent.get("scope", "<module>")
        name = ent.get("name", "")

        # scope_chain already includes the module path (set during AST walk),
        # so just append the entity name. Fallback to mod_path if scope is
        # missing for any reason.
        if scope and scope != "<module>":
            qualified = f"{scope}.{name}"
        else:
            mod_path = fp.replace(os.sep, ".").removesuffix(".py") if fp else ""
            if mod_path.endswith(".__init__"):
                mod_path = mod_path.removesuffix(".__init__")
            qualified = f"{mod_path}.{name}" if mod_path else name

        # Compute source hash from the relevant lines
        source_hash = ""
        sl = ent.get("sl")
        el = ent.get("el")
        if fp and sl and el:
            full_path = root / fp
            try:
                lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                snippet = "\n".join(lines[sl - 1 : el])
                source_hash = hashlib.sha256(snippet.encode()).hexdigest()[:16]
            except OSError:
                pass

        await writer._client.execute_write(
            "MATCH (e {symbol_id: $sid}) "
            "SET e.qualified_name = $qn, e.source_hash = $sh",
            {"sid": ent["sid"], "qn": qualified, "sh": source_hash},
        )
        updated += 1

    return {"updated": updated}
