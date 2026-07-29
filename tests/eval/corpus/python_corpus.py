"""Hand-labeled Python resolution corpus.

A small e-commerce package (`shop`) plus its test suite: 20 modules across four
sub-packages, exercising the import and dispatch shapes that actually occur in
Python code — package ``__init__`` re-exports, relative imports, module aliases,
abstract base classes, decorators, dataclasses, annotations, and stdlib calls
through both bare and dotted receivers.

Every entry in :data:`PY_EXPECTATIONS` is labeled by reading the source. Cases
the resolver currently gets wrong are labeled with the *correct* answer and are
marked ``# known miss:`` — they are the headroom above the 0.85 floor and the
list of what to improve next, not something to quietly relabel.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.indexing.models import EdgeKind
from tests.eval.corpus import EXTERNAL, UNRESOLVED, Expectation

__all__ = ["PY_CORPUS", "PY_EXPECTATIONS", "PY_TEST_FILES"]

CALLS = EdgeKind.CALLS
IMPORTS = EdgeKind.IMPORTS
INHERITS = EdgeKind.INHERITS
REFERENCES = EdgeKind.REFERENCES
TESTS = EdgeKind.TESTS


PY_CORPUS: Mapping[str, str] = {
    # ---------------------------------------------------------------- package
    "shop/__init__.py": '''"""Shop package."""

from .config import Settings, load_settings
from .errors import ShopError

__all__ = ["Settings", "ShopError", "load_settings"]
''',
    "shop/config.py": '''"""Runtime settings."""

import os
from dataclasses import dataclass

from shop.errors import ShopError

DEFAULT_DSN = "sqlite://memory"


@dataclass
class Settings:
    """Process configuration."""

    dsn: str
    debug: bool = False

    def describe(self) -> str:
        return f"{self.dsn} debug={self.debug}"


def load_settings() -> Settings:
    dsn = os.environ.get("SHOP_DSN", DEFAULT_DSN)
    if not dsn:
        raise ShopError("no dsn configured")
    return Settings(dsn=dsn)
''',
    "shop/errors.py": '''"""Error hierarchy."""


class ShopError(Exception):
    """Base class for every error this package raises."""


class NotFound(ShopError):
    def __init__(self, key: str) -> None:
        self.key = key


class Conflict(ShopError):
    """Raised when an operation would violate an invariant."""


class ValidationFailed(Conflict):
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason


def not_found(key: str) -> NotFound:
    return NotFound(key)


def invalid(field: str) -> ValidationFailed:
    return ValidationFailed(field, "invalid")
''',
    "shop/models.py": '''"""Domain entities."""

from dataclasses import dataclass, field

from shop.errors import Conflict, ValidationFailed
from shop.util.text import slugify


@dataclass
class Entity:
    """Anything with a stable identity."""

    id: str

    def key(self) -> str:
        return self.id

    def validate(self) -> None:
        if not self.id:
            raise ValidationFailed("id", "empty")


@dataclass
class User(Entity):
    email: str = ""
    tags: list = field(default_factory=list)

    def label(self) -> str:
        return slugify(self.key() + self.email)

    def validate(self) -> None:
        if "@" not in self.email:
            raise ValidationFailed("email", "missing @")


@dataclass
class Order(Entity):
    user: User = None
    total: int = 0

    def owner(self) -> User:
        return self.user

    def ensure_open(self) -> None:
        if self.total < 0:
            raise Conflict("order is closed")

    def describe(self) -> str:
        return self.owner().label()
''',
    # ------------------------------------------------------------------- util
    "shop/util/__init__.py": '''"""Utility helpers."""
''',
    "shop/util/text.py": '''"""String helpers."""

import re

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


def truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "~"


def summarize(value: str, limit: int) -> str:
    return truncate(slugify(value), limit)
''',
    "shop/util/timing.py": '''"""Timing helpers."""

import time
from functools import wraps

from shop.errors import ShopError


def now() -> float:
    return time.time()


def elapsed(start: float) -> float:
    return now() - start


def deadline(seconds: float) -> float:
    if seconds <= 0:
        raise ShopError("bad deadline")
    return now() + seconds


def timed(fn):
    """Decorator recording wall-clock duration."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = now()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed(start)

    return wrapper
''',
    # ------------------------------------------------------------------ store
    "shop/store/__init__.py": '''"""Storage package."""

from .base import Repository
from .cache import CacheRepository
from .memory import MemoryRepository

__all__ = ["CacheRepository", "MemoryRepository", "Repository"]
''',
    "shop/store/base.py": '''"""Storage interface."""

from abc import ABC, abstractmethod

from shop.errors import NotFound
from shop.models import Entity


class Repository(ABC):
    """The persistence port every store implements."""

    @abstractmethod
    def get(self, key: str) -> Entity:
        raise NotImplementedError

    @abstractmethod
    def put(self, item: Entity) -> None:
        raise NotImplementedError

    def get_many(self, keys: list) -> list:
        return [self.get(key) for key in keys]

    def require(self, key: str) -> Entity:
        item = self.get(key)
        if item is None:
            raise NotFound(key)
        return item
''',
    "shop/store/memory.py": '''"""In-memory store."""

from shop.errors import NotFound
from shop.models import Entity

from .base import Repository


class MemoryRepository(Repository):
    """Dict-backed store; the one the tests use."""

    def __init__(self) -> None:
        self._items = {}

    def get(self, key: str) -> Entity:
        if key not in self._items:
            raise NotFound(key)
        return self._items[key]

    def put(self, item: Entity) -> None:
        item.validate()
        self._items[item.key()] = item

    def seed(self, items: list) -> None:
        for item in items:
            self.put(item)

    def clear(self) -> None:
        self._items = {}
''',
    "shop/store/sql.py": '''"""SQL-backed store."""

import sqlite3

from shop.models import Entity
from shop.util.text import slugify

from .base import Repository


class SqlRepository(Repository):
    """Thin sqlite3 wrapper."""

    def __init__(self, dsn: str) -> None:
        self._conn = sqlite3.connect(dsn)

    def get(self, key: str) -> Entity:
        cursor = self._conn.execute("select body from items where key = ?", (key,))
        return cursor.fetchone()

    def put(self, item: Entity) -> None:
        self._conn.execute("insert into items values (?)", (slugify(item.key()),))

    def close(self) -> None:
        self._conn.close()
''',
    "shop/store/cache.py": '''"""A caching decorator around any repository."""

from shop.models import Entity
from shop.util.timing import now

from .base import Repository


class CacheRepository(Repository):
    """Wraps another repository with a naive TTL cache."""

    def __init__(self, inner: Repository, ttl: float = 60.0) -> None:
        self._inner = inner
        self._ttl = ttl
        self._seen = {}

    def get(self, key: str) -> Entity:
        stamp = self._seen.get(key)
        if stamp is not None and now() - stamp < self._ttl:
            return self._seen[key]
        return self._inner.get(key)

    def put(self, item: Entity) -> None:
        self._seen[item.key()] = now()
        self._inner.put(item)

    def invalidate(self) -> None:
        self._seen.clear()
''',
    # --------------------------------------------------------------- services
    "shop/services/__init__.py": '''"""Application services."""
''',
    "shop/services/accounts.py": '''"""User account service."""

from shop.errors import NotFound, ValidationFailed
from shop.models import User
from shop.store import Repository
from shop.util.text import slugify


class AccountService:
    """Application service coordinating the user store."""

    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    def create(self, email: str) -> User:
        if "@" not in email:
            raise ValidationFailed("email", "missing @")
        user = User(id=slugify(email), email=email)
        self._repo.put(user)
        return user

    def find(self, key: str) -> User:
        return self._repo.get(key)

    def require(self, key: str) -> User:
        user = self.find(key)
        if user is None:
            raise NotFound(key)
        return user

    def rename(self, key: str, email: str) -> User:
        user = self.require(key)
        user.email = email
        self._repo.put(user)
        return user
''',
    "shop/services/pricing.py": '''"""Pricing rules."""

from decimal import Decimal

from shop.errors import ValidationFailed
from shop.models import Order, User

FREE_SHIPPING_THRESHOLD = 5000


def subtotal(order: Order) -> int:
    return order.total


def shipping(order: Order) -> int:
    return 0 if subtotal(order) >= FREE_SHIPPING_THRESHOLD else 500


def total(order: Order) -> int:
    if subtotal(order) < 0:
        raise ValidationFailed("total", "negative")
    return subtotal(order) + shipping(order)


def as_decimal(amount: int) -> Decimal:
    return Decimal(amount) / 100


def discount_for(user: User) -> int:
    return 10 if user.tags else 0
''',
    "shop/services/orders.py": '''"""Order lifecycle service."""

from shop.errors import Conflict
from shop.models import Order, User
from shop.services import pricing
from shop.services.accounts import AccountService
from shop.store.base import Repository
from shop.util import timing


class OrderService:
    """Places and cancels orders on behalf of a user."""

    def __init__(self, repo: Repository, accounts: AccountService) -> None:
        self._repo = repo
        self._accounts = accounts

    def place(self, user_key: str, amount: int) -> Order:
        user = self._accounts.require(user_key)
        order = Order(id=str(timing.now()), user=user, total=amount)
        order.ensure_open()
        self._repo.put(order)
        return order

    def quote(self, order: Order) -> int:
        return pricing.total(order)

    def cancel(self, order: Order) -> None:
        order.ensure_open()
        raise Conflict("order cancelled")

    def owner_of(self, order: Order) -> User:
        return order.owner()
''',
    "shop/services/notifications.py": '''"""Outbound notifications."""

import json
from typing import Protocol

from shop.models import User
from shop.util.text import truncate


class Notifier(Protocol):
    """Anything that can deliver a message to a user."""

    def send(self, user: User, body: str) -> None: ...


class LogNotifier:
    """Notifier that writes to stdout."""

    def send(self, user: User, body: str) -> None:
        payload = json.dumps({"to": user.email, "body": truncate(body, 80)})
        self._write(payload)

    def _write(self, payload: str) -> None:
        print(payload)


def notify_all(notifier: Notifier, users: list, body: str) -> None:
    for user in users:
        notifier.send(user, body)
''',
    # -------------------------------------------------------------------- api
    "shop/api/__init__.py": '''"""HTTP layer."""
''',
    "shop/api/schemas.py": '''"""Wire shapes for the HTTP layer."""

from dataclasses import dataclass

from shop.models import Order, User
from shop.services import pricing


@dataclass
class UserView:
    id: str
    email: str


@dataclass
class OrderView:
    id: str
    owner: UserView
    total: int


def user_view(user: User) -> UserView:
    return UserView(id=user.key(), email=user.email)


def order_view(order: Order) -> OrderView:
    return OrderView(
        id=order.key(),
        owner=user_view(order.owner()),
        total=pricing.total(order),
    )
''',
    "shop/api/deps.py": '''"""Composition root."""

from shop.config import Settings, load_settings
from shop.services.accounts import AccountService
from shop.services.orders import OrderService
from shop.store.cache import CacheRepository
from shop.store.memory import MemoryRepository
from shop.store.sql import SqlRepository


def make_repository(settings: Settings):
    if settings.debug:
        return MemoryRepository()
    return CacheRepository(SqlRepository(settings.dsn))


def make_accounts(settings: Settings) -> AccountService:
    return AccountService(make_repository(settings))


def make_orders(settings: Settings) -> OrderService:
    return OrderService(make_repository(settings), make_accounts(settings))


def default_orders() -> OrderService:
    return make_orders(load_settings())
''',
    "shop/api/routes.py": '''"""HTTP handlers."""

from shop.api import deps, schemas
from shop.config import load_settings
from shop.errors import NotFound
from shop.models import Order
from shop.services.orders import OrderService
from shop.util.timing import timed


@timed
def get_user(key: str) -> schemas.UserView:
    service = deps.make_accounts(load_settings())
    return schemas.user_view(service.require(key))


@timed
def place_order(key: str, amount: int) -> schemas.OrderView:
    service = deps.default_orders()
    return schemas.order_view(service.place(key, amount))


def cancel_order(service: OrderService, order: Order) -> None:
    try:
        service.cancel(order)
    except NotFound:
        return None
''',
    # ------------------------------------------------------------------ tests
    "tests/test_accounts.py": '''"""Account service tests."""

from shop.errors import ValidationFailed
from shop.services.accounts import AccountService
from shop.store.memory import MemoryRepository


def make_service() -> AccountService:
    return AccountService(MemoryRepository())


def test_create() -> None:
    service = make_service()
    user = service.create("a@example.com")
    assert user.label()


def test_rename() -> None:
    service = make_service()
    service.create("a@example.com")
    service.rename("a-example-com", "b@example.com")


def test_require() -> None:
    service = make_service()
    try:
        service.require("missing")
    except ValidationFailed:
        return None
''',
    "tests/test_orders.py": '''"""Order service tests."""

from shop.models import Order, User
from shop.services.accounts import AccountService
from shop.services.orders import OrderService
from shop.store.memory import MemoryRepository


def build() -> OrderService:
    repo = MemoryRepository()
    return OrderService(repo, AccountService(repo))


def test_place() -> None:
    service = build()
    order = service.place("a-example-com", 100)
    assert order.owner()


def test_cancel() -> None:
    service = build()
    order = Order(id="1", user=User(id="u", email="u@example.com"), total=1)
    service.cancel(order)


def test_quote() -> None:
    service = build()
    service.quote(Order(id="2", user=None, total=10))
''',
    "tests/test_pricing.py": '''"""Pricing tests."""

from shop.models import Order
from shop.services import pricing


def test_total() -> None:
    order = Order(id="1", user=None, total=100)
    assert pricing.total(order)


def test_shipping() -> None:
    order = Order(id="2", user=None, total=9000)
    assert pricing.shipping(order) == 0


def test_as_decimal() -> None:
    assert pricing.as_decimal(100)
''',
    "tests/test_store.py": '''"""Store tests."""

from shop.errors import NotFound
from shop.models import Entity
from shop.store.cache import CacheRepository
from shop.store.memory import MemoryRepository


def test_put_and_get() -> None:
    repo = MemoryRepository()
    repo.put(Entity(id="x"))
    assert repo.get("x")


def test_cache_delegates() -> None:
    inner = MemoryRepository()
    cache = CacheRepository(inner)
    cache.put(Entity(id="y"))
    cache.get("y")


def test_missing() -> None:
    repo = MemoryRepository()
    try:
        repo.get("missing")
    except NotFound:
        return None
''',
}

PY_TEST_FILES: frozenset[str] = frozenset(
    {
        "tests/test_accounts.py",
        "tests/test_orders.py",
        "tests/test_pricing.py",
        "tests/test_store.py",
    }
)


PY_EXPECTATIONS: list[Expectation] = [
    # -- shop/__init__.py: relative imports from a package's own __init__ ----
    (IMPORTS, "shop", "shop.config.Settings"),
    (IMPORTS, "shop", "shop.config.load_settings"),
    (IMPORTS, "shop", "shop.errors.ShopError"),
    # -- shop/config.py ------------------------------------------------------
    (IMPORTS, "shop.config", EXTERNAL("os")),
    (IMPORTS, "shop.config", EXTERNAL("dataclasses.dataclass")),
    (IMPORTS, "shop.config", "shop.errors.ShopError"),
    (REFERENCES, "shop.config.Settings", EXTERNAL("dataclasses.dataclass")),
    (REFERENCES, "shop.config.load_settings", "shop.config.Settings"),
    (CALLS, "shop.config.load_settings", EXTERNAL("os.environ.get")),
    (CALLS, "shop.config.load_settings", "shop.errors.ShopError"),
    (CALLS, "shop.config.load_settings", "shop.config.Settings"),
    # -- shop/errors.py ------------------------------------------------------
    (INHERITS, "shop.errors.NotFound", "shop.errors.ShopError"),
    (INHERITS, "shop.errors.Conflict", "shop.errors.ShopError"),
    (INHERITS, "shop.errors.ValidationFailed", "shop.errors.Conflict"),
    (REFERENCES, "shop.errors.not_found", "shop.errors.NotFound"),
    (REFERENCES, "shop.errors.invalid", "shop.errors.ValidationFailed"),
    (CALLS, "shop.errors.not_found", "shop.errors.NotFound"),
    (CALLS, "shop.errors.invalid", "shop.errors.ValidationFailed"),
    # -- shop/models.py ------------------------------------------------------
    (IMPORTS, "shop.models", EXTERNAL("dataclasses.dataclass")),
    (IMPORTS, "shop.models", EXTERNAL("dataclasses.field")),
    (IMPORTS, "shop.models", "shop.errors.Conflict"),
    (IMPORTS, "shop.models", "shop.errors.ValidationFailed"),
    (IMPORTS, "shop.models", "shop.util.text.slugify"),
    (REFERENCES, "shop.models.Entity", EXTERNAL("dataclasses.dataclass")),
    (REFERENCES, "shop.models.User", EXTERNAL("dataclasses.dataclass")),
    (REFERENCES, "shop.models.Order", EXTERNAL("dataclasses.dataclass")),
    (INHERITS, "shop.models.User", "shop.models.Entity"),
    (INHERITS, "shop.models.Order", "shop.models.Entity"),
    (CALLS, "shop.models.Entity.validate", "shop.errors.ValidationFailed"),
    (CALLS, "shop.models.User", EXTERNAL("dataclasses.field")),  # class-body call
    (CALLS, "shop.models.User.label", "shop.util.text.slugify"),
    (CALLS, "shop.models.User.label", "shop.models.Entity.key"),  # self, inherited
    (CALLS, "shop.models.User.validate", "shop.errors.ValidationFailed"),
    (REFERENCES, "shop.models.Order.owner", "shop.models.User"),
    (CALLS, "shop.models.Order.ensure_open", "shop.errors.Conflict"),
    (CALLS, "shop.models.Order.describe", "shop.models.Order.owner"),
    (CALLS, "shop.models.Order.describe", "shop.models.User.label"),  # chained call
    # -- shop/util/text.py ---------------------------------------------------
    (IMPORTS, "shop.util.text", EXTERNAL("re")),
    (CALLS, "shop.util.text", EXTERNAL("re.compile")),  # module-level call
    (CALLS, "shop.util.text.slugify", UNRESOLVED("sub")),  # re.Pattern method
    (CALLS, "shop.util.text.slugify", UNRESOLVED("lower")),  # str method
    (CALLS, "shop.util.text.slugify", UNRESOLVED("strip")),  # str method
    (CALLS, "shop.util.text.summarize", "shop.util.text.truncate"),
    (CALLS, "shop.util.text.summarize", "shop.util.text.slugify"),
    # -- shop/util/timing.py -------------------------------------------------
    (IMPORTS, "shop.util.timing", EXTERNAL("time")),
    (IMPORTS, "shop.util.timing", EXTERNAL("functools.wraps")),
    (IMPORTS, "shop.util.timing", "shop.errors.ShopError"),
    (CALLS, "shop.util.timing.now", EXTERNAL("time.time")),
    (CALLS, "shop.util.timing.elapsed", "shop.util.timing.now"),
    (CALLS, "shop.util.timing.deadline", "shop.util.timing.now"),
    (CALLS, "shop.util.timing.deadline", "shop.errors.ShopError"),
    (REFERENCES, "shop.util.timing.timed.wrapper", EXTERNAL("functools.wraps")),
    (CALLS, "shop.util.timing.timed.wrapper", "shop.util.timing.now"),
    (CALLS, "shop.util.timing.timed.wrapper", "shop.util.timing.elapsed"),
    (CALLS, "shop.util.timing.timed.wrapper", UNRESOLVED("fn")),  # parameter
    # -- shop/store/__init__.py ----------------------------------------------
    (IMPORTS, "shop.store", "shop.store.base.Repository"),
    (IMPORTS, "shop.store", "shop.store.cache.CacheRepository"),
    (IMPORTS, "shop.store", "shop.store.memory.MemoryRepository"),
    # -- shop/store/base.py --------------------------------------------------
    (IMPORTS, "shop.store.base", EXTERNAL("abc.ABC")),
    (IMPORTS, "shop.store.base", EXTERNAL("abc.abstractmethod")),
    (IMPORTS, "shop.store.base", "shop.errors.NotFound"),
    (IMPORTS, "shop.store.base", "shop.models.Entity"),
    (INHERITS, "shop.store.base.Repository", EXTERNAL("abc.ABC")),
    (REFERENCES, "shop.store.base.Repository.get", EXTERNAL("abc.abstractmethod")),
    (REFERENCES, "shop.store.base.Repository.put", EXTERNAL("abc.abstractmethod")),
    (REFERENCES, "shop.store.base.Repository.get", "shop.models.Entity"),
    (REFERENCES, "shop.store.base.Repository.put", "shop.models.Entity"),
    (REFERENCES, "shop.store.base.Repository.require", "shop.models.Entity"),
    (CALLS, "shop.store.base.Repository.get_many", "shop.store.base.Repository.get"),
    (CALLS, "shop.store.base.Repository.require", "shop.store.base.Repository.get"),
    (CALLS, "shop.store.base.Repository.require", "shop.errors.NotFound"),
    # -- shop/store/memory.py ------------------------------------------------
    (IMPORTS, "shop.store.memory", "shop.errors.NotFound"),
    (IMPORTS, "shop.store.memory", "shop.models.Entity"),
    (IMPORTS, "shop.store.memory", "shop.store.base.Repository"),  # relative
    (INHERITS, "shop.store.memory.MemoryRepository", "shop.store.base.Repository"),
    (REFERENCES, "shop.store.memory.MemoryRepository.get", "shop.models.Entity"),
    (REFERENCES, "shop.store.memory.MemoryRepository.put", "shop.models.Entity"),
    (CALLS, "shop.store.memory.MemoryRepository.get", "shop.errors.NotFound"),
    (CALLS, "shop.store.memory.MemoryRepository.put", "shop.models.Entity.validate"),
    (CALLS, "shop.store.memory.MemoryRepository.put", "shop.models.Entity.key"),
    (
        CALLS,
        "shop.store.memory.MemoryRepository.seed",
        "shop.store.memory.MemoryRepository.put",
    ),
    # -- shop/store/sql.py ---------------------------------------------------
    (IMPORTS, "shop.store.sql", EXTERNAL("sqlite3")),
    (IMPORTS, "shop.store.sql", "shop.models.Entity"),
    (IMPORTS, "shop.store.sql", "shop.util.text.slugify"),
    (IMPORTS, "shop.store.sql", "shop.store.base.Repository"),
    (INHERITS, "shop.store.sql.SqlRepository", "shop.store.base.Repository"),
    (REFERENCES, "shop.store.sql.SqlRepository.get", "shop.models.Entity"),
    (REFERENCES, "shop.store.sql.SqlRepository.put", "shop.models.Entity"),
    (CALLS, "shop.store.sql.SqlRepository.__init__", EXTERNAL("sqlite3.connect")),
    (CALLS, "shop.store.sql.SqlRepository.get", UNRESOLVED("execute")),  # sqlite3 conn
    (CALLS, "shop.store.sql.SqlRepository.get", UNRESOLVED("fetchone")),
    (CALLS, "shop.store.sql.SqlRepository.put", UNRESOLVED("execute")),
    (CALLS, "shop.store.sql.SqlRepository.put", "shop.util.text.slugify"),
    (CALLS, "shop.store.sql.SqlRepository.put", "shop.models.Entity.key"),
    # `self._conn.close()` must not bind to SqlRepository.close, which is a
    # different object's method that merely shares the name.
    (CALLS, "shop.store.sql.SqlRepository.close", UNRESOLVED("close")),
    # -- shop/store/cache.py -------------------------------------------------
    (IMPORTS, "shop.store.cache", "shop.models.Entity"),
    (IMPORTS, "shop.store.cache", "shop.util.timing.now"),
    (IMPORTS, "shop.store.cache", "shop.store.base.Repository"),
    (INHERITS, "shop.store.cache.CacheRepository", "shop.store.base.Repository"),
    (REFERENCES, "shop.store.cache.CacheRepository.__init__", "shop.store.base.Repository"),
    (REFERENCES, "shop.store.cache.CacheRepository.get", "shop.models.Entity"),
    (REFERENCES, "shop.store.cache.CacheRepository.put", "shop.models.Entity"),
    (CALLS, "shop.store.cache.CacheRepository.get", "shop.util.timing.now"),
    (CALLS, "shop.store.cache.CacheRepository.get", "shop.store.base.Repository.get"),
    # known miss: `self._seen.get(key)` is a dict lookup; the resolver has no
    # type for the attribute and falls back to the same-named repository method.
    (CALLS, "shop.store.cache.CacheRepository.get", UNRESOLVED("get")),
    (CALLS, "shop.store.cache.CacheRepository.put", "shop.models.Entity.key"),
    (CALLS, "shop.store.cache.CacheRepository.put", "shop.util.timing.now"),
    (CALLS, "shop.store.cache.CacheRepository.put", "shop.store.base.Repository.put"),
    # known miss: `self._seen.clear()` vs MemoryRepository.clear, same reason.
    (CALLS, "shop.store.cache.CacheRepository.invalidate", UNRESOLVED("clear")),
    # -- shop/services/accounts.py (imports through the store barrel) --------
    (IMPORTS, "shop.services.accounts", "shop.errors.NotFound"),
    (IMPORTS, "shop.services.accounts", "shop.errors.ValidationFailed"),
    (IMPORTS, "shop.services.accounts", "shop.models.User"),
    (IMPORTS, "shop.services.accounts", "shop.store.base.Repository"),  # re-export
    (IMPORTS, "shop.services.accounts", "shop.util.text.slugify"),
    (
        REFERENCES,
        "shop.services.accounts.AccountService.__init__",
        "shop.store.base.Repository",
    ),
    (REFERENCES, "shop.services.accounts.AccountService.create", "shop.models.User"),
    (REFERENCES, "shop.services.accounts.AccountService.find", "shop.models.User"),
    (REFERENCES, "shop.services.accounts.AccountService.require", "shop.models.User"),
    (REFERENCES, "shop.services.accounts.AccountService.rename", "shop.models.User"),
    (
        CALLS,
        "shop.services.accounts.AccountService.create",
        "shop.errors.ValidationFailed",
    ),
    (CALLS, "shop.services.accounts.AccountService.create", "shop.models.User"),
    (CALLS, "shop.services.accounts.AccountService.create", "shop.util.text.slugify"),
    (
        CALLS,
        "shop.services.accounts.AccountService.create",
        "shop.store.base.Repository.put",
    ),
    (
        CALLS,
        "shop.services.accounts.AccountService.find",
        "shop.store.base.Repository.get",
    ),
    (
        CALLS,
        "shop.services.accounts.AccountService.require",
        "shop.services.accounts.AccountService.find",
    ),
    (CALLS, "shop.services.accounts.AccountService.require", "shop.errors.NotFound"),
    (
        CALLS,
        "shop.services.accounts.AccountService.rename",
        "shop.services.accounts.AccountService.require",
    ),
    (
        CALLS,
        "shop.services.accounts.AccountService.rename",
        "shop.store.base.Repository.put",
    ),
    # -- shop/services/pricing.py --------------------------------------------
    (IMPORTS, "shop.services.pricing", EXTERNAL("decimal.Decimal")),
    (IMPORTS, "shop.services.pricing", "shop.errors.ValidationFailed"),
    (IMPORTS, "shop.services.pricing", "shop.models.Order"),
    (IMPORTS, "shop.services.pricing", "shop.models.User"),
    (REFERENCES, "shop.services.pricing.subtotal", "shop.models.Order"),
    (REFERENCES, "shop.services.pricing.shipping", "shop.models.Order"),
    (REFERENCES, "shop.services.pricing.total", "shop.models.Order"),
    (REFERENCES, "shop.services.pricing.as_decimal", EXTERNAL("decimal.Decimal")),
    (REFERENCES, "shop.services.pricing.discount_for", "shop.models.User"),
    (CALLS, "shop.services.pricing.shipping", "shop.services.pricing.subtotal"),
    (CALLS, "shop.services.pricing.total", "shop.services.pricing.subtotal"),
    (CALLS, "shop.services.pricing.total", "shop.services.pricing.shipping"),
    (CALLS, "shop.services.pricing.total", "shop.errors.ValidationFailed"),
    (CALLS, "shop.services.pricing.as_decimal", EXTERNAL("decimal.Decimal")),
    # -- shop/services/orders.py ---------------------------------------------
    (IMPORTS, "shop.services.orders", "shop.errors.Conflict"),
    (IMPORTS, "shop.services.orders", "shop.models.Order"),
    (IMPORTS, "shop.services.orders", "shop.models.User"),
    (IMPORTS, "shop.services.orders", "shop.services.pricing"),  # submodule import
    (IMPORTS, "shop.services.orders", "shop.services.accounts.AccountService"),
    (IMPORTS, "shop.services.orders", "shop.store.base.Repository"),
    (IMPORTS, "shop.services.orders", "shop.util.timing"),
    (
        REFERENCES,
        "shop.services.orders.OrderService.__init__",
        "shop.store.base.Repository",
    ),
    (
        REFERENCES,
        "shop.services.orders.OrderService.__init__",
        "shop.services.accounts.AccountService",
    ),
    (REFERENCES, "shop.services.orders.OrderService.place", "shop.models.Order"),
    (REFERENCES, "shop.services.orders.OrderService.quote", "shop.models.Order"),
    (REFERENCES, "shop.services.orders.OrderService.cancel", "shop.models.Order"),
    (REFERENCES, "shop.services.orders.OrderService.owner_of", "shop.models.Order"),
    (REFERENCES, "shop.services.orders.OrderService.owner_of", "shop.models.User"),
    (
        CALLS,
        "shop.services.orders.OrderService.place",
        "shop.services.accounts.AccountService.require",
    ),
    (CALLS, "shop.services.orders.OrderService.place", "shop.models.Order"),
    (CALLS, "shop.services.orders.OrderService.place", "shop.util.timing.now"),
    (CALLS, "shop.services.orders.OrderService.place", "shop.models.Order.ensure_open"),
    (
        CALLS,
        "shop.services.orders.OrderService.place",
        "shop.store.base.Repository.put",
    ),
    (CALLS, "shop.services.orders.OrderService.quote", "shop.services.pricing.total"),
    (CALLS, "shop.services.orders.OrderService.cancel", "shop.models.Order.ensure_open"),
    (CALLS, "shop.services.orders.OrderService.cancel", "shop.errors.Conflict"),
    (CALLS, "shop.services.orders.OrderService.owner_of", "shop.models.Order.owner"),
    # -- shop/services/notifications.py --------------------------------------
    (IMPORTS, "shop.services.notifications", EXTERNAL("json")),
    (IMPORTS, "shop.services.notifications", EXTERNAL("typing.Protocol")),
    (IMPORTS, "shop.services.notifications", "shop.models.User"),
    (IMPORTS, "shop.services.notifications", "shop.util.text.truncate"),
    (
        INHERITS,
        "shop.services.notifications.Notifier",
        EXTERNAL("typing.Protocol"),
    ),
    (REFERENCES, "shop.services.notifications.Notifier.send", "shop.models.User"),
    (REFERENCES, "shop.services.notifications.LogNotifier.send", "shop.models.User"),
    (
        REFERENCES,
        "shop.services.notifications.notify_all",
        "shop.services.notifications.Notifier",
    ),
    (CALLS, "shop.services.notifications.LogNotifier.send", EXTERNAL("json.dumps")),
    (
        CALLS,
        "shop.services.notifications.LogNotifier.send",
        "shop.util.text.truncate",
    ),
    (
        CALLS,
        "shop.services.notifications.LogNotifier.send",
        "shop.services.notifications.LogNotifier._write",
    ),
    # known miss: dispatch through a parameter annotated `Notifier`; the
    # resolver ranks by name only and picks the implementation.
    (
        CALLS,
        "shop.services.notifications.notify_all",
        "shop.services.notifications.Notifier.send",
    ),
    # -- shop/api/schemas.py -------------------------------------------------
    (IMPORTS, "shop.api.schemas", EXTERNAL("dataclasses.dataclass")),
    (IMPORTS, "shop.api.schemas", "shop.models.Order"),
    (IMPORTS, "shop.api.schemas", "shop.models.User"),
    (IMPORTS, "shop.api.schemas", "shop.services.pricing"),
    (REFERENCES, "shop.api.schemas.UserView", EXTERNAL("dataclasses.dataclass")),
    (REFERENCES, "shop.api.schemas.OrderView", EXTERNAL("dataclasses.dataclass")),
    (REFERENCES, "shop.api.schemas.user_view", "shop.models.User"),
    (REFERENCES, "shop.api.schemas.user_view", "shop.api.schemas.UserView"),
    (REFERENCES, "shop.api.schemas.order_view", "shop.models.Order"),
    (REFERENCES, "shop.api.schemas.order_view", "shop.api.schemas.OrderView"),
    (CALLS, "shop.api.schemas.user_view", "shop.api.schemas.UserView"),
    (CALLS, "shop.api.schemas.user_view", "shop.models.Entity.key"),
    (CALLS, "shop.api.schemas.order_view", "shop.api.schemas.OrderView"),
    (CALLS, "shop.api.schemas.order_view", "shop.api.schemas.user_view"),
    (CALLS, "shop.api.schemas.order_view", "shop.models.Entity.key"),
    (CALLS, "shop.api.schemas.order_view", "shop.models.Order.owner"),
    (CALLS, "shop.api.schemas.order_view", "shop.services.pricing.total"),
    # -- shop/api/deps.py ----------------------------------------------------
    (IMPORTS, "shop.api.deps", "shop.config.Settings"),
    (IMPORTS, "shop.api.deps", "shop.config.load_settings"),
    (IMPORTS, "shop.api.deps", "shop.services.accounts.AccountService"),
    (IMPORTS, "shop.api.deps", "shop.services.orders.OrderService"),
    (IMPORTS, "shop.api.deps", "shop.store.cache.CacheRepository"),
    (IMPORTS, "shop.api.deps", "shop.store.memory.MemoryRepository"),
    (IMPORTS, "shop.api.deps", "shop.store.sql.SqlRepository"),
    (REFERENCES, "shop.api.deps.make_repository", "shop.config.Settings"),
    (REFERENCES, "shop.api.deps.make_accounts", "shop.config.Settings"),
    (REFERENCES, "shop.api.deps.make_accounts", "shop.services.accounts.AccountService"),
    (REFERENCES, "shop.api.deps.make_orders", "shop.config.Settings"),
    (REFERENCES, "shop.api.deps.make_orders", "shop.services.orders.OrderService"),
    (REFERENCES, "shop.api.deps.default_orders", "shop.services.orders.OrderService"),
    (CALLS, "shop.api.deps.make_repository", "shop.store.memory.MemoryRepository"),
    (CALLS, "shop.api.deps.make_repository", "shop.store.cache.CacheRepository"),
    (CALLS, "shop.api.deps.make_repository", "shop.store.sql.SqlRepository"),
    (CALLS, "shop.api.deps.make_accounts", "shop.services.accounts.AccountService"),
    (CALLS, "shop.api.deps.make_accounts", "shop.api.deps.make_repository"),
    (CALLS, "shop.api.deps.make_orders", "shop.services.orders.OrderService"),
    (CALLS, "shop.api.deps.make_orders", "shop.api.deps.make_repository"),
    (CALLS, "shop.api.deps.make_orders", "shop.api.deps.make_accounts"),
    (CALLS, "shop.api.deps.default_orders", "shop.api.deps.make_orders"),
    (CALLS, "shop.api.deps.default_orders", "shop.config.load_settings"),
    # -- shop/api/routes.py --------------------------------------------------
    (IMPORTS, "shop.api.routes", "shop.api.deps"),
    (IMPORTS, "shop.api.routes", "shop.api.schemas"),
    (IMPORTS, "shop.api.routes", "shop.config.load_settings"),
    (IMPORTS, "shop.api.routes", "shop.errors.NotFound"),
    (IMPORTS, "shop.api.routes", "shop.models.Order"),
    (IMPORTS, "shop.api.routes", "shop.services.orders.OrderService"),
    (IMPORTS, "shop.api.routes", "shop.util.timing.timed"),
    (REFERENCES, "shop.api.routes.get_user", "shop.util.timing.timed"),  # decorator
    (REFERENCES, "shop.api.routes.get_user", "shop.api.schemas.UserView"),
    (REFERENCES, "shop.api.routes.place_order", "shop.util.timing.timed"),
    (REFERENCES, "shop.api.routes.place_order", "shop.api.schemas.OrderView"),
    (REFERENCES, "shop.api.routes.cancel_order", "shop.services.orders.OrderService"),
    (REFERENCES, "shop.api.routes.cancel_order", "shop.models.Order"),
    (CALLS, "shop.api.routes.get_user", "shop.api.deps.make_accounts"),
    (CALLS, "shop.api.routes.get_user", "shop.config.load_settings"),
    (CALLS, "shop.api.routes.get_user", "shop.api.schemas.user_view"),
    (
        CALLS,
        "shop.api.routes.get_user",
        "shop.services.accounts.AccountService.require",
    ),
    (CALLS, "shop.api.routes.place_order", "shop.api.deps.default_orders"),
    (CALLS, "shop.api.routes.place_order", "shop.api.schemas.order_view"),
    (
        CALLS,
        "shop.api.routes.place_order",
        "shop.services.orders.OrderService.place",
    ),
    (
        CALLS,
        "shop.api.routes.cancel_order",
        "shop.services.orders.OrderService.cancel",
    ),
    # -- tests/test_accounts.py ----------------------------------------------
    (IMPORTS, "tests.test_accounts", "shop.errors.ValidationFailed"),
    (IMPORTS, "tests.test_accounts", "shop.services.accounts.AccountService"),
    (IMPORTS, "tests.test_accounts", "shop.store.memory.MemoryRepository"),
    (
        REFERENCES,
        "tests.test_accounts.make_service",
        "shop.services.accounts.AccountService",
    ),
    (
        CALLS,
        "tests.test_accounts.make_service",
        "shop.services.accounts.AccountService",
    ),
    (
        CALLS,
        "tests.test_accounts.make_service",
        "shop.store.memory.MemoryRepository",
    ),
    (
        TESTS,
        "tests.test_accounts.test_create",
        "shop.services.accounts.AccountService.create",
    ),
    (
        TESTS,
        "tests.test_accounts.test_rename",
        "shop.services.accounts.AccountService.rename",
    ),
    (
        TESTS,
        "tests.test_accounts.test_require",
        "shop.services.accounts.AccountService.require",
    ),
    (CALLS, "tests.test_accounts.test_create", "tests.test_accounts.make_service"),
    (
        CALLS,
        "tests.test_accounts.test_create",
        "shop.services.accounts.AccountService.create",
    ),
    (CALLS, "tests.test_accounts.test_create", "shop.models.User.label"),
    (CALLS, "tests.test_accounts.test_rename", "tests.test_accounts.make_service"),
    (
        CALLS,
        "tests.test_accounts.test_rename",
        "shop.services.accounts.AccountService.create",
    ),
    (
        CALLS,
        "tests.test_accounts.test_rename",
        "shop.services.accounts.AccountService.rename",
    ),
    (CALLS, "tests.test_accounts.test_require", "tests.test_accounts.make_service"),
    (
        CALLS,
        "tests.test_accounts.test_require",
        "shop.services.accounts.AccountService.require",
    ),
    # -- tests/test_orders.py ------------------------------------------------
    (IMPORTS, "tests.test_orders", "shop.models.Order"),
    (IMPORTS, "tests.test_orders", "shop.models.User"),
    (IMPORTS, "tests.test_orders", "shop.services.accounts.AccountService"),
    (IMPORTS, "tests.test_orders", "shop.services.orders.OrderService"),
    (REFERENCES, "tests.test_orders.build", "shop.services.orders.OrderService"),
    (CALLS, "tests.test_orders.build", "shop.store.memory.MemoryRepository"),
    (CALLS, "tests.test_orders.build", "shop.services.orders.OrderService"),
    (CALLS, "tests.test_orders.build", "shop.services.accounts.AccountService"),
    (TESTS, "tests.test_orders.test_place", "shop.services.orders.OrderService.place"),
    (TESTS, "tests.test_orders.test_cancel", "shop.services.orders.OrderService.cancel"),
    (TESTS, "tests.test_orders.test_quote", "shop.services.orders.OrderService.quote"),
    (CALLS, "tests.test_orders.test_place", "tests.test_orders.build"),
    (
        CALLS,
        "tests.test_orders.test_place",
        "shop.services.orders.OrderService.place",
    ),
    (CALLS, "tests.test_orders.test_place", "shop.models.Order.owner"),
    (CALLS, "tests.test_orders.test_cancel", "tests.test_orders.build"),
    (CALLS, "tests.test_orders.test_cancel", "shop.models.Order"),
    (CALLS, "tests.test_orders.test_cancel", "shop.models.User"),
    (
        CALLS,
        "tests.test_orders.test_cancel",
        "shop.services.orders.OrderService.cancel",
    ),
    (CALLS, "tests.test_orders.test_quote", "tests.test_orders.build"),
    (CALLS, "tests.test_orders.test_quote", "shop.models.Order"),
    (
        CALLS,
        "tests.test_orders.test_quote",
        "shop.services.orders.OrderService.quote",
    ),
    # -- tests/test_pricing.py -----------------------------------------------
    (IMPORTS, "tests.test_pricing", "shop.models.Order"),
    (IMPORTS, "tests.test_pricing", "shop.services.pricing"),
    (TESTS, "tests.test_pricing.test_total", "shop.services.pricing.total"),
    (TESTS, "tests.test_pricing.test_shipping", "shop.services.pricing.shipping"),
    (TESTS, "tests.test_pricing.test_as_decimal", "shop.services.pricing.as_decimal"),
    (CALLS, "tests.test_pricing.test_total", "shop.models.Order"),
    (CALLS, "tests.test_pricing.test_total", "shop.services.pricing.total"),
    (CALLS, "tests.test_pricing.test_shipping", "shop.models.Order"),
    (CALLS, "tests.test_pricing.test_shipping", "shop.services.pricing.shipping"),
    (CALLS, "tests.test_pricing.test_as_decimal", "shop.services.pricing.as_decimal"),
    # -- tests/test_store.py -------------------------------------------------
    (IMPORTS, "tests.test_store", "shop.errors.NotFound"),
    (IMPORTS, "tests.test_store", "shop.models.Entity"),
    (IMPORTS, "tests.test_store", "shop.store.cache.CacheRepository"),
    (IMPORTS, "tests.test_store", "shop.store.memory.MemoryRepository"),
    # No symbol is named `put_and_get` / `cache_delegates` / `missing`: the
    # TESTS heuristic must decline rather than invent a target.
    (TESTS, "tests.test_store.test_put_and_get", UNRESOLVED("put_and_get")),
    (TESTS, "tests.test_store.test_cache_delegates", UNRESOLVED("cache_delegates")),
    (TESTS, "tests.test_store.test_missing", UNRESOLVED("missing")),
    (CALLS, "tests.test_store.test_put_and_get", "shop.models.Entity"),
    (CALLS, "tests.test_store.test_put_and_get", "shop.store.memory.MemoryRepository"),
    # known miss: `repo = MemoryRepository()` then `repo.put(...)`. Correct
    # target requires local-variable type inference, which M1 does not do; the
    # resolver ranks by name and lands on the abstract base.
    (
        CALLS,
        "tests.test_store.test_put_and_get",
        "shop.store.memory.MemoryRepository.put",
    ),
    (
        CALLS,
        "tests.test_store.test_put_and_get",
        "shop.store.memory.MemoryRepository.get",
    ),
    (CALLS, "tests.test_store.test_cache_delegates", "shop.models.Entity"),
    (
        CALLS,
        "tests.test_store.test_cache_delegates",
        "shop.store.memory.MemoryRepository",
    ),
    (
        CALLS,
        "tests.test_store.test_cache_delegates",
        "shop.store.cache.CacheRepository",
    ),
    (
        CALLS,
        "tests.test_store.test_cache_delegates",
        "shop.store.cache.CacheRepository.put",
    ),
    (
        CALLS,
        "tests.test_store.test_cache_delegates",
        "shop.store.cache.CacheRepository.get",
    ),
    (CALLS, "tests.test_store.test_missing", "shop.store.memory.MemoryRepository"),
    (CALLS, "tests.test_store.test_missing", "shop.store.memory.MemoryRepository.get"),
]
