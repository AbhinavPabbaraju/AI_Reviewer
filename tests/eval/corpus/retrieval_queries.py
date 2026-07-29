"""Hand-built retrieval queries — the M2 exit gate's ground truth.

ROADMAP M2: "on 30 hand-built queries, the file a human would need appears in
the context pack >= 90% of the time." Each query below is one such case: a
symbol the diff touches, and the **one file** a reviewer would have to open to
judge the change safely, chosen by reading the corpus.

The corpora are the ones the M1 resolution gate already uses, which is
deliberate: retrieval quality is downstream of resolution quality, and measuring
both against the same repository makes a regression attributable. A resolver
change that quietly stops resolving `CALLS` edges shows up here as a recall
drop, not as a mystery.

A query is labeled with a file the *graph* cannot trivially reach from the
anchor's own module -- naming the anchor's own file would pass for free and
measure nothing. Where the honest answer is a caller, it is a caller; where it
is a test, base class, or the type the symbol returns, it is that.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PY_QUERIES", "TS_QUERIES", "RetrievalQuery"]


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One labeled retrieval case.

    ``changed_symbol`` stands in for a diff hunk: the harness looks the symbol's
    span up in the indexed snapshot and retrieves as if those lines had changed.
    Naming the symbol rather than hard-coding line numbers keeps the labels valid
    when the corpus is edited.
    """

    changed_symbol: str
    needs: str
    """Repo-relative path that must appear in the context pack."""
    why: str
    """Ground-truth reasoning, in one line."""


PY_QUERIES: tuple[RetrievalQuery, ...] = (
    RetrievalQuery(
        "shop.util.text.slugify",
        "shop/services/accounts.py",
        "the caller that feeds user-supplied email straight into it",
    ),
    RetrievalQuery(
        "shop.util.text.truncate",
        "shop/services/notifications.py",
        "the caller that truncates message bodies to 80 chars",
    ),
    RetrievalQuery(
        "shop.util.timing.now",
        "shop/store/cache.py",
        "the cache whose TTL arithmetic depends on this clock",
    ),
    RetrievalQuery(
        "shop.util.timing.timed",
        "shop/api/routes.py",
        "the handlers this decorator wraps",
    ),
    RetrievalQuery(
        "shop.models.Entity.key",
        "shop/store/memory.py",
        "the store that uses the key as its dict index",
    ),
    RetrievalQuery(
        "shop.models.Entity.validate",
        "shop/store/memory.py",
        "the store that calls validate before persisting",
    ),
    RetrievalQuery(
        "shop.models.Order.ensure_open",
        "shop/services/orders.py",
        "the service that gates placing and cancelling on it",
    ),
    RetrievalQuery(
        "shop.models.Order.owner",
        "shop/api/schemas.py",
        "the view builder that dereferences the owner",
    ),
    RetrievalQuery(
        "shop.models.User.label",
        "tests/test_accounts.py",
        "the test that asserts on the label",
    ),
    RetrievalQuery(
        "shop.errors.NotFound",
        "shop/store/memory.py",
        "a store that raises it on a missing key",
    ),
    RetrievalQuery(
        "shop.errors.ValidationFailed",
        "shop/services/accounts.py",
        "the service that raises it for a malformed email",
    ),
    RetrievalQuery(
        "shop.errors.Conflict",
        "shop/models.py",
        "the entity method that raises it",
    ),
    RetrievalQuery(
        "shop.store.base.Repository.get",
        "shop/services/accounts.py",
        "the service that reads through this port",
    ),
    RetrievalQuery(
        "shop.store.base.Repository.put",
        "shop/services/orders.py",
        "the service that writes orders through this port",
    ),
    RetrievalQuery(
        "shop.store.memory.MemoryRepository.put",
        "tests/test_store.py",
        "the test that exercises put/get round-tripping",
    ),
    RetrievalQuery(
        "shop.store.cache.CacheRepository.get",
        "shop/store/base.py",
        "the port it delegates to, and whose contract it must preserve",
    ),
    RetrievalQuery(
        "shop.store.sql.SqlRepository.put",
        "shop/store/base.py",
        "the abstract repository whose contract it implements",
    ),
    RetrievalQuery(
        "shop.services.accounts.AccountService.create",
        "tests/test_accounts.py",
        "the test that covers account creation",
    ),
    RetrievalQuery(
        "shop.services.accounts.AccountService.require",
        "shop/services/orders.py",
        "the caller that requires a user before placing an order",
    ),
    RetrievalQuery(
        "shop.services.orders.OrderService.place",
        "shop/api/routes.py",
        "the HTTP handler that places orders",
    ),
    RetrievalQuery(
        "shop.services.orders.OrderService.cancel",
        "shop/api/routes.py",
        "the handler that cancels, and swallows NotFound while doing it",
    ),
    RetrievalQuery(
        "shop.services.pricing.total",
        "shop/services/orders.py",
        "the service that quotes an order with it",
    ),
    RetrievalQuery(
        "shop.services.pricing.subtotal",
        "shop/models.py",
        "the entity whose total field it reads",
    ),
    RetrievalQuery(
        "shop.services.notifications.LogNotifier.send",
        "shop/util/text.py",
        "the truncation helper the message body passes through",
    ),
    RetrievalQuery(
        "shop.api.schemas.user_view",
        "shop/api/routes.py",
        "the handler that serializes with it",
    ),
    RetrievalQuery(
        "shop.api.schemas.order_view",
        "shop/services/pricing.py",
        "the pricing it calls to fill in the total",
    ),
    RetrievalQuery(
        "shop.api.deps.make_repository",
        "shop/store/cache.py",
        "the concrete repository this composition root wires up",
    ),
    RetrievalQuery(
        "shop.api.deps.make_accounts",
        "shop/api/routes.py",
        "the handler that resolves its dependencies through it",
    ),
    RetrievalQuery(
        "shop.config.load_settings",
        "shop/api/deps.py",
        "the composition root that loads settings to build the graph",
    ),
    RetrievalQuery(
        "shop.config.Settings",
        "shop/api/deps.py",
        "the factories that read debug/dsn off it",
    ),
)


TS_QUERIES: tuple[RetrievalQuery, ...] = (
    RetrievalQuery(
        "src/util/text::slugify",
        "src/services/accounts.ts",
        "the service that slugifies a user-supplied email into an id",
    ),
    RetrievalQuery(
        "src/util/timing::now",
        "src/store/cache.ts",
        "the cache that timestamps entries with it",
    ),
    RetrievalQuery(
        "src/models::Entity.key",
        "src/store/memory.ts",
        "the store that keys its map by it",
    ),
    RetrievalQuery(
        "src/models::Order.ensureOpen",
        "src/services/orders.ts",
        "the service that gates place and cancel on it",
    ),
    RetrievalQuery(
        "src/models::Order.owner",
        "src/api/schemas.ts",
        "the view builder that dereferences the owner",
    ),
    RetrievalQuery(
        "src/store/base::BaseRepository.put",
        "src/services/accounts.ts",
        "the service that persists new users through it",
    ),
    RetrievalQuery(
        "src/store/memory::MemoryRepository.put",
        "tests/store.test.ts",
        "the test that round-trips through it",
    ),
    RetrievalQuery(
        "src/services/accounts::AccountService.require",
        "src/api/routes.ts",
        "the handler that requires a user before responding",
    ),
    RetrievalQuery(
        "src/services/orders::OrderService.place",
        "src/hooks/useOrders.ts",
        "the React hook that places an order in an effect",
    ),
    RetrievalQuery(
        "src/services/pricing::total",
        "src/api/schemas.ts",
        "the view builder that prices the order",
    ),
)
