"""Durable Campaign aggregates and one-time confirmation challenges."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from agentqa.conventions import utc_now
from agentqa.errors import AgentQAError
from agentqa.specs.hashing import canonical_json_bytes
from agentqa.workbench.models import Campaign, CampaignConfirmationChallenge, CampaignSubrun

CAMPAIGN_STORE_SCHEMA_VERSION = "agentqa.campaign-store/v1alpha1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    project_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    campaign_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(workflow_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_campaigns_workflow
    ON campaigns(workflow_id, created_at DESC, campaign_id DESC);
CREATE TABLE IF NOT EXISTS campaign_subruns (
    subrun_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    subrun_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaign_subruns_campaign
    ON campaign_subruns(campaign_id, subrun_id);
CREATE TABLE IF NOT EXISTS campaign_challenges (
    challenge_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id) ON DELETE CASCADE,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    challenge_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaign_challenges_campaign
    ON campaign_challenges(campaign_id, created_at DESC);
"""


class CampaignStoreError(AgentQAError):
    def __init__(self, message: str, *, code: str = "CAMPAIGN_STORE_ERROR") -> None:
        super().__init__(message, code=code)


def _json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


class CampaignStore:
    """SQLite-backed campaign state with atomic aggregate updates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @staticmethod
    def _parse_campaign(payload: str) -> Campaign:
        try:
            return Campaign.model_validate_json(payload)
        except ValidationError:
            raise CampaignStoreError(
                "stored campaign is invalid", code="CAMPAIGN_STORE_FORMAT_ERROR"
            ) from None

    @staticmethod
    def _parse_subrun(payload: str) -> CampaignSubrun:
        try:
            return CampaignSubrun.model_validate_json(payload)
        except ValidationError:
            raise CampaignStoreError(
                "stored campaign subrun is invalid", code="CAMPAIGN_STORE_FORMAT_ERROR"
            ) from None

    @staticmethod
    def _parse_challenge(payload: str) -> CampaignConfirmationChallenge:
        try:
            return CampaignConfirmationChallenge.model_validate_json(payload)
        except ValidationError:
            raise CampaignStoreError(
                "stored campaign challenge is invalid", code="CAMPAIGN_STORE_FORMAT_ERROR"
            ) from None

    def _initialize_sync(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = ?",
                    ("campaign_store_schema_version",),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                        ("campaign_store_schema_version", CAMPAIGN_STORE_SCHEMA_VERSION),
                    )
                elif row["value"] != CAMPAIGN_STORE_SCHEMA_VERSION:
                    raise CampaignStoreError(
                        "campaign store schema is not supported",
                        code="CAMPAIGN_STORE_SCHEMA_ERROR",
                    )
                now = utc_now()
                rows = connection.execute(
                    "SELECT campaign_json FROM campaigns WHERE status IN ('QUEUED', 'RUNNING')"
                ).fetchall()
                for row in rows:
                    campaign = self._parse_campaign(cast(str, row["campaign_json"]))
                    interrupted = campaign.model_copy(
                        update={
                            "status": "INTERRUPTED",
                            "updated_at": now,
                            "finished_at": now,
                        }
                    )
                    self._write_campaign(connection, interrupted)
                subrun_rows = connection.execute(
                    "SELECT subrun_json FROM campaign_subruns WHERE status IN ('QUEUED', 'RUNNING')"
                ).fetchall()
                for row in subrun_rows:
                    subrun = self._parse_subrun(cast(str, row["subrun_json"]))
                    interrupted_subrun = subrun.model_copy(
                        update={"status": "INTERRUPTED", "finished_at": now}
                    )
                    self._write_subrun(connection, interrupted_subrun)
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    @staticmethod
    def _write_campaign(connection: sqlite3.Connection, campaign: Campaign) -> None:
        connection.execute(
            "UPDATE campaigns SET status = ?, campaign_json = ?, updated_at = ? "
            "WHERE campaign_id = ?",
            (
                campaign.status,
                _json(campaign.model_copy(update={"subruns": []}).model_dump(mode="json")),
                campaign.updated_at.isoformat(),
                campaign.campaign_id,
            ),
        )

    @staticmethod
    def _write_subrun(connection: sqlite3.Connection, subrun: CampaignSubrun) -> None:
        connection.execute(
            "UPDATE campaign_subruns SET run_id = ?, status = ?, subrun_json = ?, "
            "updated_at = ? WHERE subrun_id = ?",
            (
                subrun.run_id,
                subrun.status,
                _json(subrun.model_dump(mode="json")),
                utc_now().isoformat(),
                subrun.subrun_id,
            ),
        )

    def _hydrate(self, connection: sqlite3.Connection, payload: str) -> Campaign:
        campaign = self._parse_campaign(payload)
        rows = connection.execute(
            "SELECT subrun_json FROM campaign_subruns WHERE campaign_id = ? ORDER BY subrun_id",
            (campaign.campaign_id,),
        ).fetchall()
        return campaign.model_copy(
            update={"subruns": [self._parse_subrun(cast(str, row["subrun_json"])) for row in rows]}
        )

    def _create_sync(self, campaign: Campaign) -> Campaign:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT campaign_json FROM campaigns WHERE workflow_id = ? AND idempotency_key = ?",
                (campaign.workflow_id, campaign.idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._hydrate(connection, cast(str, row["campaign_json"]))
                if existing.config_hash != campaign.config_hash:
                    raise CampaignStoreError(
                        "idempotency key already exists with a different campaign configuration",
                        code="CAMPAIGN_IDEMPOTENCY_CONFLICT",
                    )
                connection.rollback()
                return existing
            connection.execute(
                """INSERT INTO campaigns(
                       campaign_id, workflow_id, project_id, idempotency_key, config_hash,
                       status, campaign_json, created_at, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    campaign.campaign_id,
                    campaign.workflow_id,
                    campaign.project_id,
                    campaign.idempotency_key,
                    campaign.config_hash,
                    campaign.status,
                    _json(campaign.model_copy(update={"subruns": []}).model_dump(mode="json")),
                    campaign.created_at.isoformat(),
                    campaign.updated_at.isoformat(),
                ),
            )
            for subrun in campaign.subruns:
                connection.execute(
                    """INSERT INTO campaign_subruns(
                           subrun_id, campaign_id, run_id, status, subrun_json, updated_at
                       ) VALUES(?, ?, ?, ?, ?, ?)""",
                    (
                        subrun.subrun_id,
                        campaign.campaign_id,
                        subrun.run_id,
                        subrun.status,
                        _json(subrun.model_dump(mode="json")),
                        campaign.updated_at.isoformat(),
                    ),
                )
            connection.commit()
            return campaign
        except CampaignStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def create(self, campaign: Campaign) -> Campaign:
        return await asyncio.to_thread(self._create_sync, campaign)

    def _get_sync(self, campaign_id: str) -> Campaign | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT campaign_json FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            return None if row is None else self._hydrate(connection, cast(str, row[0]))
        finally:
            connection.close()

    async def get(self, campaign_id: str) -> Campaign | None:
        return await asyncio.to_thread(self._get_sync, campaign_id)

    def _list_sync(self, workflow_id: str) -> list[Campaign]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT campaign_json FROM campaigns WHERE workflow_id = ? "
                "ORDER BY created_at DESC, campaign_id DESC",
                (workflow_id,),
            ).fetchall()
            return [self._hydrate(connection, cast(str, row[0])) for row in rows]
        finally:
            connection.close()

    async def list(self, workflow_id: str) -> list[Campaign]:
        return await asyncio.to_thread(self._list_sync, workflow_id)

    def _update_sync(self, campaign: Campaign) -> Campaign:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT config_hash FROM campaigns WHERE campaign_id = ?",
                (campaign.campaign_id,),
            ).fetchone()
            if row is None:
                raise CampaignStoreError("campaign does not exist", code="CAMPAIGN_NOT_FOUND")
            if row["config_hash"] != campaign.config_hash:
                raise CampaignStoreError(
                    "campaign configuration is immutable", code="CAMPAIGN_CONFIG_IMMUTABLE"
                )
            self._write_campaign(connection, campaign)
            connection.commit()
            return campaign
        except CampaignStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update(self, campaign: Campaign) -> Campaign:
        return await asyncio.to_thread(self._update_sync, campaign)

    def _update_subrun_sync(self, subrun: CampaignSubrun) -> CampaignSubrun:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM campaign_subruns WHERE subrun_id = ?", (subrun.subrun_id,)
                ).fetchone()
                is None
            ):
                raise CampaignStoreError(
                    "campaign subrun does not exist", code="CAMPAIGN_SUBRUN_NOT_FOUND"
                )
            self._write_subrun(connection, subrun)
            connection.commit()
            return subrun
        except CampaignStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update_subrun(self, subrun: CampaignSubrun) -> CampaignSubrun:
        return await asyncio.to_thread(self._update_subrun_sync, subrun)

    def _save_challenge_sync(
        self, challenge: CampaignConfirmationChallenge
    ) -> CampaignConfirmationChallenge:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute(
                "SELECT challenge_id, challenge_json FROM campaign_challenges "
                "WHERE campaign_id = ? AND status = 'PENDING'",
                (challenge.campaign_id,),
            ).fetchall()
            for row in pending:
                expired = self._parse_challenge(cast(str, row["challenge_json"])).model_copy(
                    update={"status": "EXPIRED"}
                )
                connection.execute(
                    "UPDATE campaign_challenges SET status = 'EXPIRED', challenge_json = ? "
                    "WHERE challenge_id = ?",
                    (_json(expired.model_dump(mode="json")), row["challenge_id"]),
                )
            connection.execute(
                """INSERT INTO campaign_challenges(
                       challenge_id, campaign_id, config_hash, status, challenge_json, created_at
                   ) VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    challenge.challenge_id,
                    challenge.campaign_id,
                    challenge.config_hash,
                    challenge.status,
                    _json(challenge.model_dump(mode="json")),
                    challenge.created_at.isoformat(),
                ),
            )
            connection.commit()
            return challenge
        finally:
            connection.close()

    async def save_challenge(
        self, challenge: CampaignConfirmationChallenge
    ) -> CampaignConfirmationChallenge:
        return await asyncio.to_thread(self._save_challenge_sync, challenge)

    def _consume_challenge_sync(
        self, challenge_id: str, campaign_id: str, config_hash: str, phrase_hash: str
    ) -> CampaignConfirmationChallenge:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT challenge_json FROM campaign_challenges WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
            if row is None:
                raise CampaignStoreError(
                    "confirmation challenge does not exist", code="CAMPAIGN_CHALLENGE_NOT_FOUND"
                )
            challenge = self._parse_challenge(cast(str, row[0]))
            now = utc_now()
            if (
                challenge.campaign_id != campaign_id
                or challenge.config_hash != config_hash
                or challenge.phrase_hash != phrase_hash
                or challenge.status != "PENDING"
                or challenge.expires_at <= now
            ):
                raise CampaignStoreError(
                    "confirmation challenge is invalid, expired, or already used",
                    code="CAMPAIGN_CHALLENGE_INVALID",
                )
            consumed = challenge.model_copy(update={"status": "CONSUMED", "consumed_at": now})
            cursor = connection.execute(
                "UPDATE campaign_challenges SET status = ?, challenge_json = ? "
                "WHERE challenge_id = ? AND status = 'PENDING'",
                ("CONSUMED", _json(consumed.model_dump(mode="json")), challenge_id),
            )
            if cursor.rowcount != 1:
                raise CampaignStoreError(
                    "confirmation challenge was already consumed",
                    code="CAMPAIGN_CHALLENGE_INVALID",
                )
            connection.commit()
            return consumed
        except CampaignStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def consume_challenge(
        self, challenge_id: str, campaign_id: str, config_hash: str, phrase_hash: str
    ) -> CampaignConfirmationChallenge:
        return await asyncio.to_thread(
            self._consume_challenge_sync,
            challenge_id,
            campaign_id,
            config_hash,
            phrase_hash,
        )

    async def close(self) -> None:
        return None


__all__ = ["CAMPAIGN_STORE_SCHEMA_VERSION", "CampaignStore", "CampaignStoreError"]
