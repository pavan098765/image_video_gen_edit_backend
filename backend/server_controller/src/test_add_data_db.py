"""
Test utility script to manually add credits to a user.

Usage:
    python test_add_data_db.py

Edit the USER_IDENTIFIER and CREDITS_TO_ADD before running.
"""

from db import (
    get_db_manager,
    User,
    CreditTransaction,
    utc_naive_now,
)

# =========================
# CONFIG (EDIT THIS)
# =========================

# You can use either:
# 1) internal user.id
# 2) firebase_uid
USER_IDENTIFIER = {
    "user_id": None,                     # e.g. "user_123"
    "firebase_uid": "DZBArGWWs5W79XChhvMde3bsepr2"  # e.g. "abc123xyz"
}

CREDITS_TO_ADD = 500
REASON = "manual_test_credit_grant"


# =========================
# SCRIPT
# =========================

def add_credits():
    db_manager = get_db_manager()

    with db_manager.get_session() as session:
        # -------------------------
        # Find user
        # -------------------------
        query = session.query(User)

        if USER_IDENTIFIER.get("user_id"):
            user = query.filter(User.id == USER_IDENTIFIER["user_id"]).first()
        elif USER_IDENTIFIER.get("firebase_uid"):
            user = query.filter(
                User.firebase_uid == USER_IDENTIFIER["firebase_uid"]
            ).first()
        else:
            raise ValueError("Provide either user_id or firebase_uid")

        if not user:
            raise RuntimeError("❌ User not found")

        # -------------------------
        # Update credits
        # -------------------------
        before = user.total_credits
        user.total_credits += CREDITS_TO_ADD

        # -------------------------
        # Create transaction log
        # -------------------------
        tx = CreditTransaction(
            user_id=user.id,
            delta=CREDITS_TO_ADD,
            reason=REASON,
            job_id=None,
            created_at=utc_naive_now(),
        )

        session.add(tx)
        session.commit()

        print("✅ Credits added successfully")
        print(f"User ID        : {user.id}")
        print(f"Before credits : {before}")
        print(f"Added credits  : {CREDITS_TO_ADD}")
        print(f"After credits  : {user.total_credits}")
        print(f"Transaction ID : {tx.id}")


if __name__ == "__main__":
    add_credits()
