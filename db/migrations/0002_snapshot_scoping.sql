-- Argus 0002: make the snapshot the unit of scoping, and let chunks outlive it.
--
-- Written while building the pgvector adapter, which is where 0001 stopped
-- being expressible. Two problems, both structural rather than cosmetic:
--
-- 1. Nothing in 0001 scopes a *read* to one snapshot. `symbols`, `symbol_edges`
--    and `chunks` are keyed by `repository_id`, and every snapshot re-creates
--    the repository's symbols, so `SymbolIndexPort.symbols_in_spans(repo, ...)`
--    would return the union of every commit ever indexed. Retrieval must see
--    exactly one commit's graph or the context pack is a lie about the code.
--
-- 2. `chunks` are content-addressed and deduplicated per repository -- UNIQUE
--    (repository_id, content_hash) -- which is the whole reason a one-line push
--    re-embeds one chunk instead of 29,000. But they hung off `files` with
--    ON DELETE CASCADE, and a file row belongs to a single snapshot. Deleting
--    any snapshot therefore deleted paid-for embeddings belonging to chunks that
--    are still live in later snapshots. The dedup key and the ownership FK were
--    describing two different lifetimes.
--
-- Forward-only, per ARCHITECTURE sec. 10. Written to be safe on a populated
-- database even though today's is empty: every backfill runs before its column
-- is made NOT NULL.

BEGIN;

-- --------------------------------------------------------------------------
-- (1) Snapshot scoping for the symbol graph.
--
-- Denormalized onto the row rather than joined through `files` on every query:
-- expansion issues one query per BFS level and each one would otherwise pay an
-- extra join purely to answer "which commit is this". Edges need no such column
-- -- they reference symbol ids, and symbol rows are already snapshot-scoped, so
-- an edge is reachable only from the snapshot whose symbols it connects.
-- --------------------------------------------------------------------------

ALTER TABLE symbols ADD COLUMN snapshot_id UUID
    REFERENCES index_snapshots(id) ON DELETE CASCADE;

UPDATE symbols s SET snapshot_id = f.snapshot_id
    FROM files f WHERE f.id = s.file_id;

ALTER TABLE symbols ALTER COLUMN snapshot_id SET NOT NULL;

-- An fqn is unique within a snapshot by construction (it is how the resolver
-- keys the graph). Stating it here makes the adapter's fqn -> id map a lookup
-- rather than a hope, and turns a parser bug that emits a duplicate fqn into a
-- loud write failure instead of a silently arbitrary edge target.
ALTER TABLE symbols ADD CONSTRAINT symbols_unique_per_snapshot
    UNIQUE (snapshot_id, fqn);

-- Mirrors Symbol.parent_fqn, which 0001 simply did not persist -- chunking and
-- resolution both read it, so a symbol round-tripped through Postgres was not
-- the symbol that went in. Stored as the fqn rather than a self-referencing FK:
-- a parent is always in the same snapshot, so the FK would buy nothing but an
-- insert-ordering constraint on a bulk load.
ALTER TABLE symbols ADD COLUMN parent_fqn TEXT;

-- --------------------------------------------------------------------------
-- (2) Chunks outlive snapshots; membership becomes explicit.
--
-- `file_id` and `symbol_id` degrade to hints (both nullable, neither load
-- bearing). A chunk's path and symbol fqn are intrinsic to it -- they are inputs
-- to content_hash = sha256(repo_id | path | symbol_fqn | normalized_body) -- so
-- storing them on the row is denormalization only in the bookkeeping sense.
-- --------------------------------------------------------------------------

ALTER TABLE chunks ADD COLUMN path TEXT;
ALTER TABLE chunks ADD COLUMN language TEXT;
ALTER TABLE chunks ADD COLUMN symbol_fqn TEXT;

UPDATE chunks c SET path = f.path, language = f.language
    FROM files f WHERE f.id = c.file_id;
UPDATE chunks c SET symbol_fqn = s.fqn
    FROM symbols s WHERE s.id = c.symbol_id;

ALTER TABLE chunks ALTER COLUMN path SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN language SET NOT NULL;

-- Same traversal guard the `files` table carries and CodeSpan enforces at the
-- type boundary: cloned repositories are untrusted input (ARCHITECTURE sec. 8).
ALTER TABLE chunks ADD CONSTRAINT chunks_path_is_relative
    CHECK (path !~ '^/' AND path !~ '(^|/)\.\.(/|$)');

ALTER TABLE chunks ALTER COLUMN file_id DROP NOT NULL;
ALTER TABLE chunks DROP CONSTRAINT chunks_file_id_fkey;
ALTER TABLE chunks ADD CONSTRAINT chunks_file_id_fkey
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE SET NULL;

-- Which chunks are live in which snapshot. A join table rather than a
-- `snapshot_id` column precisely because the interesting case is a chunk that
-- belongs to *many* snapshots: that row, and its embedding, is what a reused
-- chunk reuses.
CREATE TABLE snapshot_chunks (
    snapshot_id     UUID NOT NULL REFERENCES index_snapshots(id) ON DELETE CASCADE,
    chunk_id        UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (snapshot_id, chunk_id)
);

-- The reverse direction ("which snapshots hold this chunk") is what a future
-- garbage collector walks to find chunks no live snapshot references.
CREATE INDEX idx_snapshot_chunks_chunk ON snapshot_chunks (chunk_id);

-- `chunks_for_symbols` is the hottest read in retrieval -- anchors and every BFS
-- level go through it -- and it looks chunks up by fqn now that symbol_id is
-- only a hint. (Lookup by content_hash already rides the UNIQUE constraint's
-- index.)
CREATE INDEX idx_chunks_repo_symbol ON chunks (repository_id, symbol_fqn)
    WHERE symbol_fqn IS NOT NULL;

COMMIT;
