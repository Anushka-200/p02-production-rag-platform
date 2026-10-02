"""Passwords, JWT tokens and the login endpoint."""

import time
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.security import TokenError, create_token, hash_password, read_token, verify_password
from app.config import Settings
from tests.test_api import FakeEmbedder, SharedConnection
from tests.test_search import World

SECRET = "x" * 40


def settings(**kw) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(_env_file=None, jwt_secret=SECRET, **kw)  # type: ignore[call-arg]


# ---- passwords ---------------------------------------------------------------------------------
def test_a_password_verifies_and_a_wrong_one_does_not() -> None:
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored)


def test_the_same_password_hashes_differently_each_time() -> None:
    assert hash_password("same") != hash_password("same")  # random salt


@pytest.mark.parametrize("bad", [None, "", "plain", "scrypt$x", "md5$a$b", "scrypt$!!$!!"])
def test_a_missing_or_malformed_hash_is_a_failed_login_not_a_crash(bad) -> None:  # type: ignore[no-untyped-def]
    assert verify_password("anything", bad) is False


# ---- tokens ------------------------------------------------------------------------------------
def test_a_token_round_trips_to_the_user_and_org() -> None:
    user, org = uuid.uuid4(), uuid.uuid4()
    token = create_token(settings(), user, org, "member")
    assert read_token(settings(), token) == (user, org)


def test_an_expired_token_is_rejected() -> None:
    s = settings()
    claims = {"sub": str(uuid.uuid4()), "org": str(uuid.uuid4()), "iss": s.jwt_issuer,
              "exp": int(time.time()) - 5}  # fmt: skip
    with pytest.raises(TokenError):
        read_token(s, jwt.encode(claims, SECRET, algorithm="HS256"))


def test_a_token_signed_with_another_secret_is_rejected() -> None:
    forged = create_token(
        Settings(_env_file=None, jwt_secret="y" * 40),  # type: ignore[call-arg]
        uuid.uuid4(),
        uuid.uuid4(),
        "admin",
    )
    with pytest.raises(TokenError):
        read_token(settings(), forged)


def test_a_token_that_claims_no_algorithm_is_rejected() -> None:
    claims = {"sub": str(uuid.uuid4()), "org": str(uuid.uuid4()), "iss": "p02-rag",
              "exp": int(time.time()) + 600}  # fmt: skip
    with pytest.raises(TokenError):
        read_token(settings(), jwt.encode(claims, None, algorithm="none"))  # type: ignore[arg-type]


def test_a_token_for_another_issuer_or_missing_claims_is_rejected() -> None:
    other = jwt.encode(
        {"sub": str(uuid.uuid4()), "org": str(uuid.uuid4()), "iss": "evil", "exp": int(time.time()) + 60},
        SECRET, algorithm="HS256",
    )  # fmt: skip
    with pytest.raises(TokenError):
        read_token(settings(), other)
    no_org = jwt.encode(
        {"sub": "x", "iss": "p02-rag", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256"
    )
    with pytest.raises(TokenError):
        read_token(settings(), no_org)


def test_a_missing_or_short_secret_refuses_to_issue_or_read_tokens() -> None:
    with pytest.raises(TokenError, match="not set"):
        create_token(Settings(_env_file=None), uuid.uuid4(), uuid.uuid4(), "member")  # type: ignore[call-arg]
    with pytest.raises(TokenError, match="at least"):
        create_token(
            Settings(_env_file=None, jwt_secret="short"), uuid.uuid4(), uuid.uuid4(), "member"
        )  # type: ignore[call-arg]


# ---- the endpoints -----------------------------------------------------------------------------
@pytest.fixture
def client(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann = w.person("acme", "ann")
    bob = w.person("globex", "bob")
    db_schema.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password("ann-pass"), ann.user_id),
    )
    db_schema.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password("bob-pass"), bob.user_id),
    )

    def make(env="dev", **kw):  # type: ignore[no-untyped-def]
        app = create_app(
            settings(environment=env, reranker_provider="none", **kw),
            connect=lambda: SharedConnection(db_schema), embedder=FakeEmbedder(), reranker=None,
        )  # fmt: skip
        return TestClient(app)

    return make, ann, bob, w


def test_login_returns_a_token_that_identifies_the_user(client) -> None:  # type: ignore[no-untyped-def]
    make, ann, _bob, _w = client
    r = make().post("/auth/login", json={"email": "ann@x.com", "password": "ann-pass"})
    assert r.status_code == 200
    assert read_token(settings(), r.json()["access_token"]) == (ann.user_id, ann.org_id)


def test_wrong_password_and_unknown_user_look_identical(client) -> None:  # type: ignore[no-untyped-def]
    make, *_ = client
    c = make()
    wrong = c.post("/auth/login", json={"email": "ann@x.com", "password": "nope"})
    unknown = c.post("/auth/login", json={"email": "ghost@x.com", "password": "nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_a_bearer_token_is_accepted_in_production_and_the_dev_header_is_not(client) -> None:  # type: ignore[no-untyped-def]
    make, ann, _bob, _w = client
    token = create_token(settings(), ann.user_id, ann.org_id, "member")
    prod = make(env="prod")
    # a valid token works anywhere: it gets past login (this member is just not an admin)
    assert prod.post("/admin/inspect", json={"question": "q"},
                     headers={"Authorization": f"Bearer {token}"}).status_code == 403  # fmt: skip
    # the dev shortcut does nothing in production
    assert prod.post("/admin/inspect", json={"question": "q"},
                     headers={"X-Dev-User": "ann@x.com"}).status_code == 401  # fmt: skip


def test_a_token_whose_org_is_not_the_users_real_org_is_rejected(client) -> None:  # type: ignore[no-untyped-def]
    make, ann, bob, _w = client
    forged = create_token(settings(), ann.user_id, bob.org_id, "admin")  # ann claiming globex
    r = make().post("/admin/inspect", json={"question": "q"},
                    headers={"Authorization": f"Bearer {forged}"})  # fmt: skip
    assert r.status_code == 401


def test_a_token_for_a_deleted_user_stops_working_at_once(client) -> None:  # type: ignore[no-untyped-def]
    make, ann, _bob, w = client
    token = create_token(settings(), ann.user_id, ann.org_id, "member")
    w.conn.execute("DELETE FROM users WHERE id = %s", (ann.user_id,))
    r = make().post("/chat", json={"question": "q"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_garbage_tokens_are_401(client) -> None:  # type: ignore[no-untyped-def]
    make, *_ = client
    for header in ("Bearer nonsense", "Bearer ", "Basic abc", "Bearer a.b.c"):
        r = make().post("/chat", json={"question": "q"}, headers={"Authorization": header})
        assert r.status_code == 401, header


def test_the_same_email_in_two_orgs_logs_into_the_one_whose_password_matches(client) -> None:  # type: ignore[no-untyped-def]
    make, ann, _bob, w = client
    other = w.person("initech", "ann")  # ann@x.com again, in another organisation
    w.conn.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password("initech-pass"), other.user_id),
    )
    c = make()
    first = c.post("/auth/login", json={"email": "ann@x.com", "password": "ann-pass"})
    second = c.post("/auth/login", json={"email": "ann@x.com", "password": "initech-pass"})
    assert read_token(settings(), first.json()["access_token"]) == (ann.user_id, ann.org_id)
    assert read_token(settings(), second.json()["access_token"]) == (other.user_id, other.org_id)


def test_an_ambiguous_login_is_refused_not_guessed(client) -> None:  # type: ignore[no-untyped-def]
    make, _ann, _bob, w = client
    other = w.person("initech", "ann")
    w.conn.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password("ann-pass"), other.user_id),
    )  # the same email AND the same password in two organisations
    r = make().post("/auth/login", json={"email": "ann@x.com", "password": "ann-pass"})
    assert r.status_code == 401
