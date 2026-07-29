"""Hand-labeled TypeScript resolution corpus.

The Python corpus's application, rebuilt with TypeScript's idioms: barrel
``index.ts`` re-exports (named, typed, and wildcard), namespace imports,
interfaces with ``implements``, abstract-ish base classes, parameter properties,
arrow-function exports, and a React hook importing an external package.

TypeScript fqns are ``<module-path>::<Nested.Name>`` (a TS module id *is* its
path), so they are written out literally here rather than translated.

Two M1 modeling decisions the labels respect:

* Interface *members* are not symbols — only the interface itself is. A call
  dispatched through an interface-typed value is therefore labeled with the
  concrete implementation it can only reach, not with a phantom member.
* No ``TESTS`` edges are inferred for TypeScript (a Jest ``describe`` block is
  not the strong naming signal Python's ``test_*`` is), so the test modules here
  contribute ordinary ``CALLS`` and ``IMPORTS`` labels only.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.indexing.models import EdgeKind
from tests.eval.corpus import EXTERNAL, UNRESOLVED, Expectation

__all__ = ["TS_CORPUS", "TS_EXPECTATIONS", "TS_TEST_FILES"]

CALLS = EdgeKind.CALLS
IMPORTS = EdgeKind.IMPORTS
INHERITS = EdgeKind.INHERITS
REFERENCES = EdgeKind.REFERENCES


TS_CORPUS: Mapping[str, str] = {
    # ------------------------------------------------------------- top barrel
    "src/index.ts": """export * from "./models";
export * from "./errors";
export { OrderService } from "./services/orders";
export { AccountService } from "./services/accounts";
""",
    "src/errors.ts": """export class ShopError extends Error {
  constructor(readonly reason: string) {
    super(reason);
  }
}

export class NotFound extends ShopError {}

export class Conflict extends ShopError {}

export class ValidationFailed extends Conflict {
  constructor(readonly field: string) {
    super(`invalid ${field}`);
  }
}

export function notFound(key: string): NotFound {
  return new NotFound(key);
}

export function invalid(field: string): ValidationFailed {
  return new ValidationFailed(field);
}
""",
    "src/models.ts": """import { Conflict, ValidationFailed } from "./errors";
import { slugify } from "./util/text";

export interface Identified {
  id: string;
}

export class Entity implements Identified {
  constructor(readonly id: string) {}

  key(): string {
    return this.id;
  }

  validate(): void {
    if (!this.id) {
      throw new ValidationFailed("id");
    }
  }
}

export class User extends Entity {
  constructor(id: string, readonly email: string) {
    super(id);
  }

  label(): string {
    return slugify(this.key() + this.email);
  }
}

export class Order extends Entity {
  constructor(id: string, readonly user: User, readonly total: number) {
    super(id);
  }

  owner(): User {
    return this.user;
  }

  ensureOpen(): void {
    if (this.total < 0) {
      throw new Conflict("closed");
    }
  }

  describe(): string {
    return this.owner().label();
  }
}
""",
    # ------------------------------------------------------------------- util
    "src/util/index.ts": """export * from "./text";
export * from "./timing";
""",
    "src/util/text.ts": """const SLUG = /[^a-z0-9]+/g;

export function slugify(value: string): string {
  return value.toLowerCase().replace(SLUG, "-");
}

export function truncate(value: string, limit: number): string {
  return value.length <= limit ? value : value.slice(0, limit - 1) + "~";
}

export function summarize(value: string, limit: number): string {
  return truncate(slugify(value), limit);
}
""",
    "src/util/timing.ts": """export function now(): number {
  return Date.now();
}

export function elapsed(start: number): number {
  return now() - start;
}

export const deadline = (seconds: number): number => now() + seconds * 1000;
""",
    # ------------------------------------------------------------------ store
    "src/store/index.ts": """export { BaseRepository } from "./base";
export type { Repository } from "./base";
export { MemoryRepository } from "./memory";
export { CacheRepository } from "./cache";
""",
    "src/store/base.ts": """import { NotFound } from "../errors";
import { Entity } from "../models";

export interface Repository {
  get(key: string): Entity;
  put(item: Entity): void;
}

export class BaseRepository implements Repository {
  get(key: string): Entity {
    throw new NotFound(key);
  }

  put(item: Entity): void {
    throw new NotFound(item.key());
  }

  getMany(keys: string[]): Entity[] {
    return keys.map((key) => this.get(key));
  }

  require(key: string): Entity {
    const item = this.get(key);
    if (!item) {
      throw new NotFound(key);
    }
    return item;
  }
}
""",
    "src/store/memory.ts": """import { NotFound } from "../errors";
import { Entity } from "../models";

import { BaseRepository } from "./base";

export class MemoryRepository extends BaseRepository {
  private items = new Map<string, Entity>();

  get(key: string): Entity {
    const found = this.items.get(key);
    if (!found) {
      throw new NotFound(key);
    }
    return found;
  }

  put(item: Entity): void {
    item.validate();
    this.items.set(item.key(), item);
  }

  seed(items: Entity[]): void {
    items.forEach((item) => this.put(item));
  }
}
""",
    "src/store/cache.ts": """import { Entity } from "../models";
import { now } from "../util/timing";

import { BaseRepository } from "./base";
import type { Repository } from "./base";

export class CacheRepository extends BaseRepository {
  private stamps = new Map<string, number>();

  constructor(private readonly inner: Repository) {
    super();
  }

  get(key: string): Entity {
    return this.inner.get(key);
  }

  put(item: Entity): void {
    this.stamps.set(item.key(), now());
    this.inner.put(item);
  }
}
""",
    # --------------------------------------------------------------- services
    "src/services/accounts.ts": """import { NotFound, ValidationFailed } from "../errors";
import { User } from "../models";
import { Repository } from "../store";
import { slugify } from "../util";

export class AccountService {
  constructor(private readonly repo: Repository) {}

  create(email: string): User {
    if (!email.includes("@")) {
      throw new ValidationFailed("email");
    }
    const user = new User(slugify(email), email);
    this.repo.put(user);
    return user;
  }

  find(key: string): User {
    return this.repo.get(key) as User;
  }

  require(key: string): User {
    const user = this.find(key);
    if (!user) {
      throw new NotFound(key);
    }
    return user;
  }
}
""",
    "src/services/pricing.ts": """import { ValidationFailed } from "../errors";
import { Order, User } from "../models";

export const FREE_SHIPPING = 5000;

export function subtotal(order: Order): number {
  return order.total;
}

export function shipping(order: Order): number {
  return subtotal(order) >= FREE_SHIPPING ? 0 : 500;
}

export function total(order: Order): number {
  if (subtotal(order) < 0) {
    throw new ValidationFailed("total");
  }
  return subtotal(order) + shipping(order);
}

export function discountFor(user: User): number {
  return user.email.endsWith("@vip.example") ? 10 : 0;
}
""",
    "src/services/orders.ts": """import { Conflict } from "../errors";
import { Order, User } from "../models";
import { Repository } from "../store";
import { now } from "../util/timing";

import { AccountService } from "./accounts";
import * as pricing from "./pricing";

export class OrderService {
  constructor(
    private readonly repo: Repository,
    private readonly accounts: AccountService,
  ) {}

  place(userKey: string, amount: number): Order {
    const user = this.accounts.require(userKey);
    const order = new Order(`${now()}`, user, amount);
    order.ensureOpen();
    this.repo.put(order);
    return order;
  }

  quote(order: Order): number {
    return pricing.total(order);
  }

  cancel(order: Order): void {
    order.ensureOpen();
    throw new Conflict("cancelled");
  }

  ownerOf(order: Order): User {
    return order.owner();
  }
}
""",
    "src/services/notifications.ts": """import { User } from "../models";
import { truncate } from "../util";

export interface Notifier {
  send(user: User, body: string): void;
}

export class LogNotifier implements Notifier {
  send(user: User, body: string): void {
    this.write(`${user.email}: ${truncate(body, 80)}`);
  }

  private write(line: string): void {
    console.log(line);
  }
}

export function notifyAll(notifier: Notifier, users: User[], body: string): void {
  users.forEach((user) => notifier.send(user, body));
}
""",
    # -------------------------------------------------------------------- api
    "src/api/schemas.ts": """import { Order, User } from "../models";
import * as pricing from "../services/pricing";

export interface UserView {
  id: string;
  email: string;
}

export interface OrderView {
  id: string;
  owner: UserView;
  total: number;
}

export function userView(user: User): UserView {
  return { id: user.key(), email: user.email };
}

export function orderView(order: Order): OrderView {
  return {
    id: order.key(),
    owner: userView(order.owner()),
    total: pricing.total(order),
  };
}
""",
    "src/api/deps.ts": """import { AccountService } from "../services/accounts";
import { OrderService } from "../services/orders";
import { CacheRepository, MemoryRepository, Repository } from "../store";

export function makeRepository(): Repository {
  return new CacheRepository(new MemoryRepository());
}

export function makeAccounts(): AccountService {
  return new AccountService(makeRepository());
}

export function makeOrders(): OrderService {
  return new OrderService(makeRepository(), makeAccounts());
}
""",
    "src/api/routes.ts": """import { Order } from "../models";
import { OrderService } from "../services/orders";

import * as deps from "./deps";
import { OrderView, UserView, orderView, userView } from "./schemas";

export function getUser(key: string): UserView {
  const service = deps.makeAccounts();
  return userView(service.require(key));
}

export function placeOrder(key: string, amount: number): OrderView {
  const service = deps.makeOrders();
  return orderView(service.place(key, amount));
}

export function cancelOrder(service: OrderService, order: Order): void {
  service.cancel(order);
}
""",
    "src/hooks/useOrders.ts": """import { useEffect, useState } from "react";

import { makeOrders } from "../api/deps";
import { Order } from "../models";

export const useOrders = (key: string) => {
  const [orders, setOrders] = useState<Order[]>([]);
  useEffect(() => {
    const service = makeOrders();
    setOrders([service.place(key, 0)]);
  }, [key]);
  return orders;
};
""",
    # ------------------------------------------------------------------ tests
    "tests/orders.test.ts": """import { Order, User } from "../src";
import { AccountService } from "../src/services/accounts";
import { OrderService } from "../src/services/orders";
import { MemoryRepository } from "../src/store";

function build(): OrderService {
  const repo = new MemoryRepository();
  return new OrderService(repo, new AccountService(repo));
}

export function testPlace(): void {
  const service = build();
  const order = service.place("k", 100);
  order.owner();
}

export function testCancel(): void {
  const service = build();
  service.cancel(new Order("1", new User("u", "u@example.com"), 1));
}
""",
    "tests/store.test.ts": """import { Entity, User } from "../src/models";
import { CacheRepository } from "../src/store/cache";
import { MemoryRepository } from "../src/store/memory";

export function testPutAndGet(): void {
  const repo = new MemoryRepository();
  repo.put(new User("x", "x@example.com"));
  repo.get("x");
}

export function testCacheDelegates(): void {
  const cache = new CacheRepository(new MemoryRepository());
  cache.put(new User("y", "y@example.com"));
  cache.get("y");
}

export function items(): Entity[] {
  return [];
}
""",
}

TS_TEST_FILES: frozenset[str] = frozenset(
    {"tests/orders.test.ts", "tests/store.test.ts"}
)


TS_EXPECTATIONS: list[Expectation] = [
    # -- src/index.ts: wildcard + named barrel --------------------------------
    (IMPORTS, "src/index", "src/models"),
    (IMPORTS, "src/index", "src/errors"),
    (IMPORTS, "src/index", "src/services/orders::OrderService"),
    (IMPORTS, "src/index", "src/services/accounts::AccountService"),
    # -- src/errors.ts --------------------------------------------------------
    (INHERITS, "src/errors::NotFound", "src/errors::ShopError"),
    (INHERITS, "src/errors::Conflict", "src/errors::ShopError"),
    (INHERITS, "src/errors::ValidationFailed", "src/errors::Conflict"),
    (REFERENCES, "src/errors::notFound", "src/errors::NotFound"),
    (REFERENCES, "src/errors::invalid", "src/errors::ValidationFailed"),
    (CALLS, "src/errors::notFound", "src/errors::NotFound"),
    (CALLS, "src/errors::invalid", "src/errors::ValidationFailed"),
    # -- src/models.ts --------------------------------------------------------
    (IMPORTS, "src/models", "src/errors::Conflict"),
    (IMPORTS, "src/models", "src/errors::ValidationFailed"),
    (IMPORTS, "src/models", "src/util/text::slugify"),
    (INHERITS, "src/models::Entity", "src/models::Identified"),  # implements
    (INHERITS, "src/models::User", "src/models::Entity"),
    (INHERITS, "src/models::Order", "src/models::Entity"),
    (CALLS, "src/models::Entity.validate", "src/errors::ValidationFailed"),
    (CALLS, "src/models::User.label", "src/util/text::slugify"),
    (CALLS, "src/models::User.label", "src/models::Entity.key"),  # this, inherited
    (REFERENCES, "src/models::Order.constructor", "src/models::User"),
    (REFERENCES, "src/models::Order.owner", "src/models::User"),
    (CALLS, "src/models::Order.ensureOpen", "src/errors::Conflict"),
    (CALLS, "src/models::Order.describe", "src/models::Order.owner"),
    (CALLS, "src/models::Order.describe", "src/models::User.label"),  # chained
    # -- src/util --------------------------------------------------------------
    (IMPORTS, "src/util/index", "src/util/text"),  # two wildcards, both kept
    (IMPORTS, "src/util/index", "src/util/timing"),
    (CALLS, "src/util/text::slugify", UNRESOLVED("toLowerCase")),
    (CALLS, "src/util/text::slugify", UNRESOLVED("replace")),
    (CALLS, "src/util/text::truncate", UNRESOLVED("slice")),
    (CALLS, "src/util/text::summarize", "src/util/text::truncate"),
    (CALLS, "src/util/text::summarize", "src/util/text::slugify"),
    (CALLS, "src/util/timing::elapsed", "src/util/timing::now"),
    (CALLS, "src/util/timing::deadline", "src/util/timing::now"),  # arrow export
    # -- src/store/index.ts ----------------------------------------------------
    (IMPORTS, "src/store/index", "src/store/base::BaseRepository"),
    (IMPORTS, "src/store/index", "src/store/base::Repository"),  # `export type`
    (IMPORTS, "src/store/index", "src/store/memory::MemoryRepository"),
    (IMPORTS, "src/store/index", "src/store/cache::CacheRepository"),
    # -- src/store/base.ts -----------------------------------------------------
    (IMPORTS, "src/store/base", "src/errors::NotFound"),
    (IMPORTS, "src/store/base", "src/models::Entity"),
    (INHERITS, "src/store/base::BaseRepository", "src/store/base::Repository"),
    (REFERENCES, "src/store/base::BaseRepository.get", "src/models::Entity"),
    (REFERENCES, "src/store/base::BaseRepository.put", "src/models::Entity"),
    (REFERENCES, "src/store/base::BaseRepository.getMany", "src/models::Entity"),
    (REFERENCES, "src/store/base::BaseRepository.require", "src/models::Entity"),
    (CALLS, "src/store/base::BaseRepository.get", "src/errors::NotFound"),
    (CALLS, "src/store/base::BaseRepository.put", "src/errors::NotFound"),
    (CALLS, "src/store/base::BaseRepository.put", "src/models::Entity.key"),
    (
        CALLS,
        "src/store/base::BaseRepository.getMany",
        "src/store/base::BaseRepository.get",
    ),
    (CALLS, "src/store/base::BaseRepository.getMany", UNRESOLVED("map")),
    (
        CALLS,
        "src/store/base::BaseRepository.require",
        "src/store/base::BaseRepository.get",
    ),
    (CALLS, "src/store/base::BaseRepository.require", "src/errors::NotFound"),
    # -- src/store/memory.ts ---------------------------------------------------
    (IMPORTS, "src/store/memory", "src/errors::NotFound"),
    (IMPORTS, "src/store/memory", "src/models::Entity"),
    (IMPORTS, "src/store/memory", "src/store/base::BaseRepository"),
    (INHERITS, "src/store/memory::MemoryRepository", "src/store/base::BaseRepository"),
    (REFERENCES, "src/store/memory::MemoryRepository.get", "src/models::Entity"),
    (REFERENCES, "src/store/memory::MemoryRepository.put", "src/models::Entity"),
    (REFERENCES, "src/store/memory::MemoryRepository.seed", "src/models::Entity"),
    (CALLS, "src/store/memory::MemoryRepository.get", "src/errors::NotFound"),
    # known miss: `this.items` is a Map; no type for the attribute, so the
    # resolver name-matches against the repository classes' `get`.
    (CALLS, "src/store/memory::MemoryRepository.get", UNRESOLVED("get")),
    (CALLS, "src/store/memory::MemoryRepository.put", "src/models::Entity.validate"),
    (CALLS, "src/store/memory::MemoryRepository.put", "src/models::Entity.key"),
    (CALLS, "src/store/memory::MemoryRepository.put", UNRESOLVED("set")),
    (
        CALLS,
        "src/store/memory::MemoryRepository.seed",
        "src/store/memory::MemoryRepository.put",
    ),
    (CALLS, "src/store/memory::MemoryRepository.seed", UNRESOLVED("forEach")),
    # -- src/store/cache.ts ----------------------------------------------------
    (IMPORTS, "src/store/cache", "src/models::Entity"),
    (IMPORTS, "src/store/cache", "src/util/timing::now"),
    (IMPORTS, "src/store/cache", "src/store/base::BaseRepository"),
    (IMPORTS, "src/store/cache", "src/store/base::Repository"),  # `import type`
    (INHERITS, "src/store/cache::CacheRepository", "src/store/base::BaseRepository"),
    (REFERENCES, "src/store/cache::CacheRepository.constructor", "src/store/base::Repository"),
    (REFERENCES, "src/store/cache::CacheRepository.get", "src/models::Entity"),
    (REFERENCES, "src/store/cache::CacheRepository.put", "src/models::Entity"),
    # `this.inner` is typed `Repository`, whose members are not symbols; the
    # reachable implementation is the base class's.
    (
        CALLS,
        "src/store/cache::CacheRepository.get",
        "src/store/base::BaseRepository.get",
    ),
    (CALLS, "src/store/cache::CacheRepository.put", "src/models::Entity.key"),
    (CALLS, "src/store/cache::CacheRepository.put", "src/util/timing::now"),
    (CALLS, "src/store/cache::CacheRepository.put", UNRESOLVED("set")),
    (
        CALLS,
        "src/store/cache::CacheRepository.put",
        "src/store/base::BaseRepository.put",
    ),
    # -- src/services/accounts.ts (imports through two barrels) ---------------
    (IMPORTS, "src/services/accounts", "src/errors::NotFound"),
    (IMPORTS, "src/services/accounts", "src/errors::ValidationFailed"),
    (IMPORTS, "src/services/accounts", "src/models::User"),
    (IMPORTS, "src/services/accounts", "src/store/base::Repository"),  # named barrel
    (IMPORTS, "src/services/accounts", "src/util/text::slugify"),  # wildcard barrel
    (
        REFERENCES,
        "src/services/accounts::AccountService.constructor",
        "src/store/base::Repository",
    ),
    (REFERENCES, "src/services/accounts::AccountService.create", "src/models::User"),
    (REFERENCES, "src/services/accounts::AccountService.find", "src/models::User"),
    (REFERENCES, "src/services/accounts::AccountService.require", "src/models::User"),
    (CALLS, "src/services/accounts::AccountService.create", UNRESOLVED("includes")),
    (
        CALLS,
        "src/services/accounts::AccountService.create",
        "src/errors::ValidationFailed",
    ),
    (CALLS, "src/services/accounts::AccountService.create", "src/models::User"),
    (CALLS, "src/services/accounts::AccountService.create", "src/util/text::slugify"),
    (
        CALLS,
        "src/services/accounts::AccountService.create",
        "src/store/base::BaseRepository.put",
    ),
    (
        CALLS,
        "src/services/accounts::AccountService.find",
        "src/store/base::BaseRepository.get",
    ),
    (
        CALLS,
        "src/services/accounts::AccountService.require",
        "src/services/accounts::AccountService.find",
    ),
    (CALLS, "src/services/accounts::AccountService.require", "src/errors::NotFound"),
    # -- src/services/pricing.ts ----------------------------------------------
    (IMPORTS, "src/services/pricing", "src/errors::ValidationFailed"),
    (IMPORTS, "src/services/pricing", "src/models::Order"),
    (IMPORTS, "src/services/pricing", "src/models::User"),
    (REFERENCES, "src/services/pricing::subtotal", "src/models::Order"),
    (REFERENCES, "src/services/pricing::shipping", "src/models::Order"),
    (REFERENCES, "src/services/pricing::total", "src/models::Order"),
    (REFERENCES, "src/services/pricing::discountFor", "src/models::User"),
    (CALLS, "src/services/pricing::shipping", "src/services/pricing::subtotal"),
    (CALLS, "src/services/pricing::total", "src/services/pricing::subtotal"),
    (CALLS, "src/services/pricing::total", "src/services/pricing::shipping"),
    (CALLS, "src/services/pricing::total", "src/errors::ValidationFailed"),
    (CALLS, "src/services/pricing::discountFor", UNRESOLVED("endsWith")),
    # -- src/services/orders.ts ------------------------------------------------
    (IMPORTS, "src/services/orders", "src/errors::Conflict"),
    (IMPORTS, "src/services/orders", "src/models::Order"),
    (IMPORTS, "src/services/orders", "src/models::User"),
    (IMPORTS, "src/services/orders", "src/store/base::Repository"),
    (IMPORTS, "src/services/orders", "src/util/timing::now"),
    (IMPORTS, "src/services/orders", "src/services/accounts::AccountService"),
    (IMPORTS, "src/services/orders", "src/services/pricing"),  # namespace import
    (
        REFERENCES,
        "src/services/orders::OrderService.constructor",
        "src/store/base::Repository",
    ),
    (
        REFERENCES,
        "src/services/orders::OrderService.constructor",
        "src/services/accounts::AccountService",
    ),
    (REFERENCES, "src/services/orders::OrderService.place", "src/models::Order"),
    (REFERENCES, "src/services/orders::OrderService.quote", "src/models::Order"),
    (REFERENCES, "src/services/orders::OrderService.cancel", "src/models::Order"),
    (REFERENCES, "src/services/orders::OrderService.ownerOf", "src/models::Order"),
    (REFERENCES, "src/services/orders::OrderService.ownerOf", "src/models::User"),
    (
        CALLS,
        "src/services/orders::OrderService.place",
        "src/services/accounts::AccountService.require",
    ),
    (CALLS, "src/services/orders::OrderService.place", "src/models::Order"),
    (CALLS, "src/services/orders::OrderService.place", "src/util/timing::now"),
    (
        CALLS,
        "src/services/orders::OrderService.place",
        "src/models::Order.ensureOpen",
    ),
    (
        CALLS,
        "src/services/orders::OrderService.place",
        "src/store/base::BaseRepository.put",
    ),
    (CALLS, "src/services/orders::OrderService.quote", "src/services/pricing::total"),
    (
        CALLS,
        "src/services/orders::OrderService.cancel",
        "src/models::Order.ensureOpen",
    ),
    (CALLS, "src/services/orders::OrderService.cancel", "src/errors::Conflict"),
    (CALLS, "src/services/orders::OrderService.ownerOf", "src/models::Order.owner"),
    # -- src/services/notifications.ts -----------------------------------------
    (IMPORTS, "src/services/notifications", "src/models::User"),
    (IMPORTS, "src/services/notifications", "src/util/text::truncate"),
    (
        INHERITS,
        "src/services/notifications::LogNotifier",
        "src/services/notifications::Notifier",
    ),
    (REFERENCES, "src/services/notifications::LogNotifier.send", "src/models::User"),
    (
        REFERENCES,
        "src/services/notifications::notifyAll",
        "src/services/notifications::Notifier",
    ),
    (REFERENCES, "src/services/notifications::notifyAll", "src/models::User"),
    (
        CALLS,
        "src/services/notifications::LogNotifier.send",
        "src/util/text::truncate",
    ),
    (
        CALLS,
        "src/services/notifications::LogNotifier.send",
        "src/services/notifications::LogNotifier.write",
    ),
    (
        CALLS,
        "src/services/notifications::notifyAll",
        "src/services/notifications::LogNotifier.send",
    ),
    (CALLS, "src/services/notifications::notifyAll", UNRESOLVED("forEach")),
    # -- src/api/schemas.ts ----------------------------------------------------
    (IMPORTS, "src/api/schemas", "src/models::Order"),
    (IMPORTS, "src/api/schemas", "src/models::User"),
    (IMPORTS, "src/api/schemas", "src/services/pricing"),
    (REFERENCES, "src/api/schemas::userView", "src/models::User"),
    (REFERENCES, "src/api/schemas::userView", "src/api/schemas::UserView"),
    (REFERENCES, "src/api/schemas::orderView", "src/models::Order"),
    (REFERENCES, "src/api/schemas::orderView", "src/api/schemas::OrderView"),
    (CALLS, "src/api/schemas::userView", "src/models::Entity.key"),
    (CALLS, "src/api/schemas::orderView", "src/models::Entity.key"),
    (CALLS, "src/api/schemas::orderView", "src/api/schemas::userView"),
    (CALLS, "src/api/schemas::orderView", "src/models::Order.owner"),
    (CALLS, "src/api/schemas::orderView", "src/services/pricing::total"),
    # -- src/api/deps.ts -------------------------------------------------------
    (IMPORTS, "src/api/deps", "src/services/accounts::AccountService"),
    (IMPORTS, "src/api/deps", "src/services/orders::OrderService"),
    (IMPORTS, "src/api/deps", "src/store/cache::CacheRepository"),
    (IMPORTS, "src/api/deps", "src/store/memory::MemoryRepository"),
    (IMPORTS, "src/api/deps", "src/store/base::Repository"),
    (REFERENCES, "src/api/deps::makeRepository", "src/store/base::Repository"),
    (REFERENCES, "src/api/deps::makeAccounts", "src/services/accounts::AccountService"),
    (REFERENCES, "src/api/deps::makeOrders", "src/services/orders::OrderService"),
    (CALLS, "src/api/deps::makeRepository", "src/store/cache::CacheRepository"),
    (CALLS, "src/api/deps::makeRepository", "src/store/memory::MemoryRepository"),
    (CALLS, "src/api/deps::makeAccounts", "src/services/accounts::AccountService"),
    (CALLS, "src/api/deps::makeAccounts", "src/api/deps::makeRepository"),
    (CALLS, "src/api/deps::makeOrders", "src/services/orders::OrderService"),
    (CALLS, "src/api/deps::makeOrders", "src/api/deps::makeRepository"),
    (CALLS, "src/api/deps::makeOrders", "src/api/deps::makeAccounts"),
    # -- src/api/routes.ts -----------------------------------------------------
    (IMPORTS, "src/api/routes", "src/models::Order"),
    (IMPORTS, "src/api/routes", "src/services/orders::OrderService"),
    (IMPORTS, "src/api/routes", "src/api/deps"),  # namespace import
    (IMPORTS, "src/api/routes", "src/api/schemas::UserView"),
    (IMPORTS, "src/api/routes", "src/api/schemas::OrderView"),
    (IMPORTS, "src/api/routes", "src/api/schemas::userView"),
    (IMPORTS, "src/api/routes", "src/api/schemas::orderView"),
    (REFERENCES, "src/api/routes::getUser", "src/api/schemas::UserView"),
    (REFERENCES, "src/api/routes::placeOrder", "src/api/schemas::OrderView"),
    (REFERENCES, "src/api/routes::cancelOrder", "src/services/orders::OrderService"),
    (REFERENCES, "src/api/routes::cancelOrder", "src/models::Order"),
    (CALLS, "src/api/routes::getUser", "src/api/deps::makeAccounts"),
    (CALLS, "src/api/routes::getUser", "src/api/schemas::userView"),
    (
        CALLS,
        "src/api/routes::getUser",
        "src/services/accounts::AccountService.require",
    ),
    (CALLS, "src/api/routes::placeOrder", "src/api/deps::makeOrders"),
    (CALLS, "src/api/routes::placeOrder", "src/api/schemas::orderView"),
    (
        CALLS,
        "src/api/routes::placeOrder",
        "src/services/orders::OrderService.place",
    ),
    (
        CALLS,
        "src/api/routes::cancelOrder",
        "src/services/orders::OrderService.cancel",
    ),
    # -- src/hooks/useOrders.ts (external package) -----------------------------
    (IMPORTS, "src/hooks/useOrders", EXTERNAL("react.useEffect")),
    (IMPORTS, "src/hooks/useOrders", EXTERNAL("react.useState")),
    (IMPORTS, "src/hooks/useOrders", "src/api/deps::makeOrders"),
    (IMPORTS, "src/hooks/useOrders", "src/models::Order"),
    (CALLS, "src/hooks/useOrders::useOrders", EXTERNAL("react.useState")),
    (CALLS, "src/hooks/useOrders::useOrders", EXTERNAL("react.useEffect")),
    (CALLS, "src/hooks/useOrders::useOrders", "src/api/deps::makeOrders"),
    (
        CALLS,
        "src/hooks/useOrders::useOrders",
        "src/services/orders::OrderService.place",
    ),
    (CALLS, "src/hooks/useOrders::useOrders", UNRESOLVED("setOrders")),
    # -- tests/orders.test.ts --------------------------------------------------
    (IMPORTS, "tests/orders.test", "src/models::Order"),  # via the top barrel
    (IMPORTS, "tests/orders.test", "src/models::User"),
    (IMPORTS, "tests/orders.test", "src/services/accounts::AccountService"),
    (IMPORTS, "tests/orders.test", "src/services/orders::OrderService"),
    (IMPORTS, "tests/orders.test", "src/store/memory::MemoryRepository"),
    (REFERENCES, "tests/orders.test::build", "src/services/orders::OrderService"),
    (CALLS, "tests/orders.test::build", "src/store/memory::MemoryRepository"),
    (CALLS, "tests/orders.test::build", "src/services/orders::OrderService"),
    (CALLS, "tests/orders.test::build", "src/services/accounts::AccountService"),
    (CALLS, "tests/orders.test::testPlace", "tests/orders.test::build"),
    (
        CALLS,
        "tests/orders.test::testPlace",
        "src/services/orders::OrderService.place",
    ),
    (CALLS, "tests/orders.test::testPlace", "src/models::Order.owner"),
    (CALLS, "tests/orders.test::testCancel", "tests/orders.test::build"),
    (
        CALLS,
        "tests/orders.test::testCancel",
        "src/services/orders::OrderService.cancel",
    ),
    (CALLS, "tests/orders.test::testCancel", "src/models::Order"),
    (CALLS, "tests/orders.test::testCancel", "src/models::User"),
    # -- tests/store.test.ts ---------------------------------------------------
    (IMPORTS, "tests/store.test", "src/models::Entity"),
    (IMPORTS, "tests/store.test", "src/models::User"),
    (IMPORTS, "tests/store.test", "src/store/cache::CacheRepository"),
    (IMPORTS, "tests/store.test", "src/store/memory::MemoryRepository"),
    (REFERENCES, "tests/store.test::items", "src/models::Entity"),
    (CALLS, "tests/store.test::testPutAndGet", "src/store/memory::MemoryRepository"),
    (CALLS, "tests/store.test::testPutAndGet", "src/models::User"),
    # known miss: `const repo = new MemoryRepository()` then `repo.put(...)`
    # needs local-variable type inference, which M1 does not do.
    (
        CALLS,
        "tests/store.test::testPutAndGet",
        "src/store/memory::MemoryRepository.put",
    ),
    (
        CALLS,
        "tests/store.test::testPutAndGet",
        "src/store/memory::MemoryRepository.get",
    ),
    (
        CALLS,
        "tests/store.test::testCacheDelegates",
        "src/store/cache::CacheRepository",
    ),
    (
        CALLS,
        "tests/store.test::testCacheDelegates",
        "src/store/memory::MemoryRepository",
    ),
    (CALLS, "tests/store.test::testCacheDelegates", "src/models::User"),
    (
        CALLS,
        "tests/store.test::testCacheDelegates",
        "src/store/cache::CacheRepository.put",
    ),
    (
        CALLS,
        "tests/store.test::testCacheDelegates",
        "src/store/cache::CacheRepository.get",
    ),
]
