"""
Redis Transaction Store — maintains sender history for real-time feature computation.

Uses Redis sorted sets (scored by Unix timestamp) to efficiently query
recent transactions within sliding windows (1h, 6h, 24h, 7d).

Falls back to an in-memory dictionary if Redis is unavailable.
"""
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TransactionStore:
    """Transaction history store using Redis sorted sets with in-memory fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        self._redis = None
        self._memory: Dict[str, list] = {}  # fallback: sender_cif -> [tx_dicts]
        self._pair_counts: Dict[str, int] = {}  # fallback: "sender|receiver" -> count

        try:
            import redis
            self._redis = redis.from_url(redis_url, db=db, decode_responses=True)
            self._redis.ping()
            logger.info("Redis connected at %s (db=%d)", redis_url, db)
        except Exception as e:
            logger.warning("Redis unavailable (%s). Using in-memory fallback.", e)
            self._redis = None

    @property
    def is_redis_connected(self) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    # ─── Store ───────────────────────────────────────────────────────────

    def store_transaction(self, tx: dict) -> None:
        """Store a transaction for later history queries.

        Args:
            tx: dict with keys: sender_cif, receiver_cif, amount_vnd,
                receiver_bank_code, transaction_type, device_mac_hash, timestamp
        """
        sender = tx["sender_cif"]
        receiver = tx["receiver_cif"]
        ts_epoch = self._to_epoch(tx.get("timestamp"))
        tx_record = {
            "amount_vnd": tx["amount_vnd"],
            "receiver_cif": receiver,
            "receiver_bank_code": tx["receiver_bank_code"],
            "transaction_type": tx["transaction_type"],
            "device_mac_hash": tx["device_mac_hash"],
            "timestamp": ts_epoch,
        }

        if self._redis:
            try:
                key = f"sender:{sender}:txns"
                self._redis.zadd(key, {json.dumps(tx_record): ts_epoch})
                # Expire after 8 days (slightly longer than our 7-day window)
                self._redis.expire(key, 8 * 86400)

                # Track sender→receiver pair count
                pair_key = f"pair:{sender}|{receiver}"
                self._redis.incr(pair_key)
                self._redis.expire(pair_key, 30 * 86400)  # 30-day pair tracking

                # Track receiver incoming count
                recv_key = f"receiver:{receiver}:senders"
                self._redis.sadd(recv_key, sender)
                self._redis.expire(recv_key, 30 * 86400)

                # Track sender outgoing bank diversity
                bank_key = f"sender:{sender}:banks"
                self._redis.sadd(bank_key, tx["receiver_bank_code"])
                self._redis.expire(bank_key, 30 * 86400)

                return
            except Exception as e:
                logger.warning("Redis write failed (%s), falling back to memory.", e)

        # In-memory fallback
        self._memory.setdefault(sender, []).append(tx_record)
        pair_id = f"{sender}|{receiver}"
        self._pair_counts[pair_id] = self._pair_counts.get(pair_id, 0) + 1

    # ─── Query: Sender History ───────────────────────────────────────────

    def get_sender_history(
        self, sender_cif: str, window_seconds: int, reference_ts: Optional[float] = None
    ) -> List[dict]:
        """Get all transactions from a sender within a time window.

        Args:
            sender_cif: sender ID
            window_seconds: how far back to look (e.g., 3600 for 1 hour)
            reference_ts: reference unix timestamp (defaults to now)

        Returns:
            List of transaction dicts within the window.
        """
        now = reference_ts or time.time()
        cutoff = now - window_seconds

        if self._redis:
            try:
                key = f"sender:{sender_cif}:txns"
                # ZRANGEBYSCORE returns members with score between cutoff and now
                raw = self._redis.zrangebyscore(key, cutoff, now)
                return [json.loads(r) for r in raw]
            except Exception as e:
                logger.warning("Redis read failed (%s), falling back.", e)

        # In-memory fallback
        txns = self._memory.get(sender_cif, [])
        return [t for t in txns if t["timestamp"] >= cutoff]

    # ─── Query: Pair Count ───────────────────────────────────────────────

    def get_pair_count(self, sender_cif: str, receiver_cif: str) -> int:
        """Count prior transactions between this sender→receiver pair."""
        if self._redis:
            try:
                val = self._redis.get(f"pair:{sender_cif}|{receiver_cif}")
                return int(val) if val else 0
            except Exception:
                pass

        return self._pair_counts.get(f"{sender_cif}|{receiver_cif}", 0)

    # ─── Query: Unique Receivers ─────────────────────────────────────────

    def get_unique_receivers(
        self, sender_cif: str, window_seconds: int, reference_ts: Optional[float] = None
    ) -> int:
        """Count distinct receivers this sender has sent to within a window."""
        history = self.get_sender_history(sender_cif, window_seconds, reference_ts)
        return len(set(t["receiver_cif"] for t in history))

    # ─── Query: Unique Devices ───────────────────────────────────────────

    def get_unique_devices(
        self, sender_cif: str, window_seconds: int, reference_ts: Optional[float] = None
    ) -> int:
        """Count distinct devices this sender has used within a window."""
        history = self.get_sender_history(sender_cif, window_seconds, reference_ts)
        return len(set(t["device_mac_hash"] for t in history))

    # ─── Query: Sender Out-Degree ────────────────────────────────────────

    def get_sender_out_degree(self, sender_cif: str) -> int:
        """Total distinct receivers this sender has ever sent to (within TTL)."""
        if self._redis:
            try:
                key = f"sender:{sender_cif}:txns"
                raw = self._redis.zrange(key, 0, -1)
                receivers = set()
                for r in raw:
                    receivers.add(json.loads(r)["receiver_cif"])
                return len(receivers)
            except Exception:
                pass

        txns = self._memory.get(sender_cif, [])
        return len(set(t["receiver_cif"] for t in txns))

    # ─── Query: Receiver In-Degree ───────────────────────────────────────

    def get_receiver_in_degree(self, receiver_cif: str) -> int:
        """Total distinct senders that have sent to this receiver."""
        if self._redis:
            try:
                return self._redis.scard(f"receiver:{receiver_cif}:senders")
            except Exception:
                pass

        # In-memory: count senders that sent to this receiver
        senders = set()
        for sender, txns in self._memory.items():
            for t in txns:
                if t["receiver_cif"] == receiver_cif:
                    senders.add(sender)
                    break
        return len(senders)

    # ─── Query: Bank Diversity ───────────────────────────────────────────

    def get_sender_bank_diversity(self, sender_cif: str) -> int:
        """Number of different banks this sender has sent money to."""
        if self._redis:
            try:
                return self._redis.scard(f"sender:{sender_cif}:banks")
            except Exception:
                pass

        txns = self._memory.get(sender_cif, [])
        return len(set(t["receiver_bank_code"] for t in txns))

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_epoch(ts_str: Optional[str]) -> float:
        """Convert timestamp string to Unix epoch seconds."""
        if ts_str is None:
            return time.time()
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            return time.time()
