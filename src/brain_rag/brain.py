import asyncio
from collections.abc import Iterable

from .models import BrainFact


class SupabaseBrainSource:
    """Read-only adapter for TaskTree's Supabase Brain tables."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def fetch_facts(
        self, workspace_id: str, *, entity: str | None = None, section: str | None = None
    ) -> list[BrainFact]:
        return await asyncio.to_thread(self._fetch_sync, workspace_id, entity, section)

    def _fetch_sync(
        self, workspace_id: str, entity: str | None, section: str | None
    ) -> list[BrainFact]:
        import psycopg
        from psycopg.rows import dict_row

        query = """
            SELECT
                o.id::text AS fact_id,
                o.entity_id::text AS entity_id,
                e.kind AS entity_kind,
                e.name AS entity_name,
                o.body,
                COALESCE(o.section, '') AS section,
                o.source,
                o.source_ref,
                o.occurred_at
            FROM observations AS o
            JOIN entities AS e ON e.id = o.entity_id AND e.workspace_id = o.workspace_id
            WHERE o.workspace_id = %s
              AND o.body IS NOT NULL
              AND btrim(o.body) <> ''
              AND o.body NOT LIKE 'Szukaj takze po:%%'
              AND (%s::text IS NULL OR e.name ILIKE '%%' || %s::text || '%%')
              AND (%s::text IS NULL OR COALESCE(o.section, '') = %s::text)
            ORDER BY o.occurred_at DESC, o.id
        """
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                query, (workspace_id, entity, entity, section, section)
            ).fetchall()
        return [BrainFact.model_validate(row) for row in rows]


class StaticBrainSource:
    def __init__(self, facts: Iterable[BrainFact]) -> None:
        self.facts = list(facts)

    async def fetch_facts(
        self, workspace_id: str, *, entity: str | None = None, section: str | None = None
    ) -> list[BrainFact]:
        del workspace_id
        return [
            fact
            for fact in self.facts
            if (entity is None or entity.lower() in fact.entity_name.lower())
            and (section is None or section == fact.section)
        ]
