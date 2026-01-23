# Backend In-App Purchase Implementation Guide

## Overview

This document provides comprehensive implementation notes for the backend to securely verify and process in-app purchases from the Flutter frontend. The system follows a **"Never Trust Client"** approach - all purchases must be verified server-side before crediting gems or enabling ad-free status.

---

## Critical Security Architecture

### Prices
- **Prices are NOT stored in the app or config files**
- Prices come directly from **Google Play Store** via `in_app_purchase` plugin
- The `ProductDetails.price` field contains the localized, region-appropriate price
- Users see the exact price Google charges - no hardcoding, no client manipulation possible

### Gem Amounts
- **The backend has its own `PRODUCT_ID → GEMS` mapping** (see below)
- The client ONLY sends `product_id` (e.g., `"credits_500"`)
- The client NEVER sends the gem amount to award
- Backend looks up gem amount from its own secure mapping
- Even if client is tampered, backend awards correct gem amount based on product_id

### Config Integrity (tool_config.json)
- Frontend downloads config from R2 storage
- Config is verified via SHA256 hash comparison
- Server should provide `tool_config.json.sha256` file
- Any tampering in transit or locally is detected
- Config contains AI model costs/info for display only - actual costs are validated server-side

---

## Product IDs (Must Match Play Console)

| Product ID | Type | Gems | Description |
|------------|------|------|-------------|
| `remove_ads_lifetime` | Non-consumable | - | One-time purchase to remove all ads |
| `credits_50` | Consumable | 50 | Tiny Pack |
| `credits_100` | Consumable | 100 | Starter Pack |
| `credits_500` | Consumable | 500 | Power Pack (Popular) |
| `credits_1000` | Consumable | 1,000 | Mega Pack |
| `credits_2500` | Consumable | 2,500 | Super Pack |
| `credits_5000` | Consumable | 5,000 | Ultra Pack |
| `credits_10000` | Consumable | 10,000 | Legendary Pack (Best Value) |

---

## API Endpoints Required

### 1. Verify Gem Purchase

**Endpoint:** `POST /auth/verify-gem-purchase`

**Purpose:** Verify a consumable gem pack purchase with Google Play, prevent duplicates, and credit gems to user account.

**Request Headers:**
```
Authorization: Bearer <firebase_id_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "purchase_token": "string",    // Required: Google Play purchase token
  "product_id": "credits_500",   // Required: Product ID from kProductIds
  "platform": "android",          // Required: "android" or "ios"
  "device_id": "string",          // Optional: Device ID for fraud detection
  "order_id": "string"            // Optional (Android): Order ID for duplicate detection
}
```

**Successful Response (200):**
```json
{
  "success": true,
  "gems_awarded": 500,
  "new_balance": 1250,
  "order_id": "GPA.3313-1234-5678-90123",
  "message": "Purchase verified and gems credited"
}
```

**Duplicate Purchase Response (409):**
```json
{
  "success": false,
  "detail": "Purchase already processed",
  "current_balance": 1250,
  "order_id": "GPA.3313-1234-5678-90123"
}
```

**Error Responses:**
- `400` - Invalid request (missing fields, invalid product_id)
- `401` - Unauthorized (invalid/expired token)
- `402` - Payment verification failed with Google
- `404` - User account not found
- `409` - Duplicate purchase (already processed)
- `500` - Server error

---

### 2. Verify Ad-Free Purchase

**Endpoint:** `POST /auth/verify-ad-free-purchase`

**Purpose:** Verify the one-time ad-free purchase and enable ad-free status for user.

**Request Body:**
```json
{
  "purchase_token": "string",
  "product_id": "remove_ads_lifetime",
  "platform": "android",
  "device_id": "string"
}
```

**Successful Response (200):**
```json
{
  "verified": true,
  "message": "Ad-free purchase verified and activated"
}
```

---

### 3. Check Ad-Free Status

**Endpoint:** `GET /auth/ad-free-status`

**Purpose:** Check if user has ad-free status (called on app startup).

**Response (200):**
```json
{
  "is_ad_free": true,
  "purchased_at": "2026-01-13T10:30:00Z"
}
```

---

## Google Play Verification Implementation

### Prerequisites

1. **Enable Google Play Developer API** in Google Cloud Console
2. **Create Service Account** with access to Google Play Developer API
3. **Link Service Account** to Play Console (Users & Permissions)
4. **Download Service Account JSON** key file

### Python Implementation

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
from datetime import datetime, timezone

# Load service account credentials
SCOPES = ['https://www.googleapis.com/auth/androidpublisher']
SERVICE_ACCOUNT_FILE = 'path/to/service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# Build the Android Publisher API client
androidpublisher = build('androidpublisher', 'v3', credentials=credentials)

PACKAGE_NAME = 'com.shambhutech.kalaai'  # Your app's package name

# Product ID to gems mapping
PRODUCT_GEMS_MAP = {
    'credits_50': 50,
    'credits_100': 100,
    'credits_500': 500,
    'credits_1000': 1000,
    'credits_2500': 2500,
    'credits_5000': 5000,
    'credits_10000': 10000,
}


async def verify_google_play_purchase(
    purchase_token: str,
    product_id: str,
    expected_package: str = PACKAGE_NAME
) -> dict:
    """
    Verify a purchase with Google Play Developer API.
    
    Returns:
        dict with keys: valid, order_id, purchase_state, purchase_time, consumed
    """
    try:
        # Call Google Play API to verify the purchase
        result = androidpublisher.purchases().products().get(
            packageName=expected_package,
            productId=product_id,
            token=purchase_token
        ).execute()
        
        """
        Response fields:
        - kind: "androidpublisher#productPurchase"
        - purchaseTimeMillis: Time of purchase (ms since epoch)
        - purchaseState: 0 = Purchased, 1 = Canceled, 2 = Pending
        - consumptionState: 0 = Yet to be consumed, 1 = Consumed
        - developerPayload: Custom payload (if any)
        - orderId: The order ID
        - purchaseType: 0 = Test, 1 = Promo, not present = Normal purchase
        - acknowledgementState: 0 = Not acknowledged, 1 = Acknowledged
        - purchaseToken: The purchase token
        - productId: The product ID
        - quantity: Quantity (usually 1)
        - regionCode: ISO 3166-1 alpha-2 country code
        """
        
        purchase_state = result.get('purchaseState', -1)
        consumption_state = result.get('consumptionState', 0)
        order_id = result.get('orderId', '')
        purchase_time_ms = result.get('purchaseTimeMillis', 0)
        
        return {
            'valid': purchase_state == 0,  # 0 = Purchased successfully
            'order_id': order_id,
            'purchase_state': purchase_state,
            'purchase_time': datetime.fromtimestamp(int(purchase_time_ms) / 1000, tz=timezone.utc),
            'consumed': consumption_state == 1,
            'raw_response': result
        }
        
    except Exception as e:
        print(f"Google Play verification error: {e}")
        return {
            'valid': False,
            'error': str(e)
        }


async def acknowledge_purchase(purchase_token: str, product_id: str) -> bool:
    """
    Acknowledge a purchase (required within 3 days or purchase is refunded).
    Note: For consumables, you should consume instead of acknowledge.
    """
    try:
        androidpublisher.purchases().products().acknowledge(
            packageName=PACKAGE_NAME,
            productId=product_id,
            token=purchase_token,
            body={}
        ).execute()
        return True
    except Exception as e:
        print(f"Acknowledge error: {e}")
        return False


async def consume_purchase(purchase_token: str, product_id: str) -> bool:
    """
    Consume a consumable purchase (allows re-purchase).
    """
    try:
        androidpublisher.purchases().products().consume(
            packageName=PACKAGE_NAME,
            productId=product_id,
            token=purchase_token
        ).execute()
        return True
    except Exception as e:
        print(f"Consume error: {e}")
        return False
```

---

## Database Schema

### Required Tables

```sql
-- Store all processed purchases for audit & duplicate prevention
CREATE TABLE purchases (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL REFERENCES users(firebase_uid),
    order_id VARCHAR(255) UNIQUE,  -- Google Play order ID (for duplicate detection)
    purchase_token TEXT NOT NULL,   -- Store hashed or encrypted
    product_id VARCHAR(100) NOT NULL,
    platform VARCHAR(20) NOT NULL,  -- 'android' or 'ios'
    gems_awarded INTEGER DEFAULT 0,
    amount_paid DECIMAL(10, 2),     -- If available from receipt
    currency VARCHAR(10),
    purchase_state VARCHAR(50),     -- 'verified', 'pending', 'refunded'
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_response JSONB,             -- Store full Google response for auditing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for fast lookups
    INDEX idx_purchases_user_id (user_id),
    INDEX idx_purchases_order_id (order_id),
    INDEX idx_purchases_product_id (product_id)
);

-- Track ad-free purchases separately (non-consumable)
CREATE TABLE ad_free_purchases (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) UNIQUE NOT NULL REFERENCES users(firebase_uid),
    order_id VARCHAR(255),
    purchase_token_hash VARCHAR(64),  -- SHA256 hash for duplicate detection
    platform VARCHAR(20) NOT NULL,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Add is_ad_free column to users table if not exists
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_ad_free BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS ad_free_since TIMESTAMP WITH TIME ZONE;
```

---

## Endpoint Implementation (FastAPI Example)

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import hashlib

router = APIRouter()

class GemPurchaseRequest(BaseModel):
    purchase_token: str
    product_id: str
    platform: str
    device_id: Optional[str] = None
    order_id: Optional[str] = None

class GemPurchaseResponse(BaseModel):
    success: bool
    gems_awarded: int = 0
    new_balance: int = 0
    order_id: Optional[str] = None
    message: Optional[str] = None


@router.post("/auth/verify-gem-purchase", response_model=GemPurchaseResponse)
async def verify_gem_purchase(
    request: GemPurchaseRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """
    Verify a gem pack purchase with Google Play and credit gems.
    
    Security measures:
    1. Verify purchase token with Google Play API
    2. Check for duplicate order_id to prevent replay attacks
    3. Credit gems atomically in database transaction
    4. Consume the purchase on Google Play
    """
    
    # 1. Validate product_id
    if request.product_id not in PRODUCT_GEMS_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid product_id: {request.product_id}")
    
    gems_to_award = PRODUCT_GEMS_MAP[request.product_id]
    
    # 2. Check for duplicate purchase (by order_id if provided)
    if request.order_id:
        existing = await db.fetch_one(
            "SELECT id FROM purchases WHERE order_id = :order_id",
            {"order_id": request.order_id}
        )
        if existing:
            # Already processed - return current balance (not an error, just already done)
            current_balance = await get_user_credits(current_user.firebase_uid, db)
            raise HTTPException(
                status_code=409,
                detail="Purchase already processed",
                headers={"X-Current-Balance": str(current_balance)}
            )
    
    # 3. Verify with Google Play API
    verification = await verify_google_play_purchase(
        purchase_token=request.purchase_token,
        product_id=request.product_id
    )
    
    if not verification.get('valid'):
        raise HTTPException(
            status_code=402,
            detail=f"Purchase verification failed: {verification.get('error', 'Invalid purchase')}"
        )
    
    # 4. Double-check for duplicate using verified order_id
    verified_order_id = verification.get('order_id')
    if verified_order_id:
        existing = await db.fetch_one(
            "SELECT id FROM purchases WHERE order_id = :order_id",
            {"order_id": verified_order_id}
        )
        if existing:
            current_balance = await get_user_credits(current_user.firebase_uid, db)
            return GemPurchaseResponse(
                success=True,  # Not an error - already credited
                gems_awarded=0,
                new_balance=current_balance,
                order_id=verified_order_id,
                message="Purchase already processed"
            )
    
    # 5. Credit gems in atomic transaction
    async with db.transaction():
        # Insert purchase record
        await db.execute(
            """
            INSERT INTO purchases 
            (user_id, order_id, purchase_token, product_id, platform, 
             gems_awarded, purchase_state, raw_response)
            VALUES 
            (:user_id, :order_id, :token_hash, :product_id, :platform,
             :gems, 'verified', :raw_response)
            """,
            {
                "user_id": current_user.firebase_uid,
                "order_id": verified_order_id,
                "token_hash": hashlib.sha256(request.purchase_token.encode()).hexdigest(),
                "product_id": request.product_id,
                "platform": request.platform,
                "gems": gems_to_award,
                "raw_response": json.dumps(verification.get('raw_response', {}))
            }
        )
        
        # Update user's gem balance
        await db.execute(
            """
            UPDATE users 
            SET credits = credits + :gems 
            WHERE firebase_uid = :uid
            """,
            {"gems": gems_to_award, "uid": current_user.firebase_uid}
        )
        
        # Get new balance
        new_balance = await get_user_credits(current_user.firebase_uid, db)
    
    # 6. Consume the purchase on Google Play (allows re-purchase of consumables)
    # Do this AFTER crediting to ensure user gets their gems even if consume fails
    try:
        await consume_purchase(request.purchase_token, request.product_id)
    except Exception as e:
        # Log but don't fail - user already got their gems
        print(f"Warning: Failed to consume purchase: {e}")
    
    return GemPurchaseResponse(
        success=True,
        gems_awarded=gems_to_award,
        new_balance=new_balance,
        order_id=verified_order_id,
        message="Purchase verified and gems credited"
    )


@router.post("/auth/verify-ad-free-purchase")
async def verify_ad_free_purchase(
    request: GemPurchaseRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """
    Verify the one-time ad-free purchase.
    """
    
    if request.product_id != 'remove_ads_lifetime':
        raise HTTPException(status_code=400, detail="Invalid product_id for ad-free")
    
    # Check if already ad-free
    if current_user.is_ad_free:
        return {"verified": True, "message": "Already ad-free"}
    
    # Verify with Google Play
    verification = await verify_google_play_purchase(
        purchase_token=request.purchase_token,
        product_id=request.product_id
    )
    
    if not verification.get('valid'):
        raise HTTPException(
            status_code=402,
            detail=f"Purchase verification failed: {verification.get('error', 'Invalid purchase')}"
        )
    
    # Check for duplicate
    token_hash = hashlib.sha256(request.purchase_token.encode()).hexdigest()
    existing = await db.fetch_one(
        "SELECT id FROM ad_free_purchases WHERE purchase_token_hash = :hash",
        {"hash": token_hash}
    )
    
    if existing:
        # Already processed - ensure user is marked ad-free
        await db.execute(
            "UPDATE users SET is_ad_free = TRUE WHERE firebase_uid = :uid",
            {"uid": current_user.firebase_uid}
        )
        return {"verified": True, "message": "Ad-free already active"}
    
    # Enable ad-free
    async with db.transaction():
        await db.execute(
            """
            INSERT INTO ad_free_purchases 
            (user_id, order_id, purchase_token_hash, platform)
            VALUES (:uid, :order_id, :hash, :platform)
            """,
            {
                "uid": current_user.firebase_uid,
                "order_id": verification.get('order_id'),
                "hash": token_hash,
                "platform": request.platform
            }
        )
        
        await db.execute(
            """
            UPDATE users 
            SET is_ad_free = TRUE, ad_free_since = NOW() 
            WHERE firebase_uid = :uid
            """,
            {"uid": current_user.firebase_uid}
        )
    
    # Acknowledge the purchase (non-consumable)
    try:
        await acknowledge_purchase(request.purchase_token, request.product_id)
    except Exception as e:
        print(f"Warning: Failed to acknowledge ad-free purchase: {e}")
    
    return {"verified": True, "message": "Ad-free purchase verified and activated"}


@router.get("/auth/ad-free-status")
async def get_ad_free_status(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Check if user has ad-free status."""
    
    user = await db.fetch_one(
        "SELECT is_ad_free, ad_free_since FROM users WHERE firebase_uid = :uid",
        {"uid": current_user.firebase_uid}
    )
    
    return {
        "is_ad_free": user['is_ad_free'] if user else False,
        "purchased_at": user['ad_free_since'].isoformat() if user and user['ad_free_since'] else None
    }
```

---

## Security Best Practices

### 1. Never Trust Client Data
- Always verify purchase tokens with Google Play API
- Don't rely on client-sent gem amounts - use server-side mapping

### 2. Duplicate Prevention
- Store `order_id` and check before processing
- Use database transactions for atomic operations
- Hash purchase tokens for duplicate detection

### 3. Fraud Detection
- Log device_id for pattern analysis
- Monitor for unusual purchase patterns
- Implement rate limiting on verification endpoints

### 4. Error Handling
- Return 409 for duplicates (not errors - purchase was already credited)
- Log all verification attempts for auditing
- Store raw Google response for disputes

### 5. Timing Considerations
- Verify BEFORE crediting gems
- Acknowledge/Consume AFTER crediting (user gets gems even if this fails)
- Google requires acknowledgment within 3 days or purchase is auto-refunded

---

## Testing

### Test Purchases
1. Add license testers in Play Console
2. Use test purchase accounts
3. Test purchases return `purchaseType: 0` (test purchase)

### Test Scenarios
1. ✅ Valid purchase → Credit gems
2. ✅ Duplicate purchase → Return 409, show current balance
3. ✅ Invalid token → Return 402
4. ✅ Wrong product_id → Return 400
5. ✅ Unsigned user → Return 401
6. ✅ Account not found → Return 404

---

## Refund Handling (Optional but Recommended)

Google Play sends real-time developer notifications (RTDN) for refunds:

```python
@router.post("/webhooks/google-play")
async def handle_google_play_webhook(request: Request, db: Database = Depends(get_db)):
    """
    Handle Google Play real-time developer notifications.
    Useful for handling refunds and subscription changes.
    """
    body = await request.json()
    
    # Decode the notification
    # See: https://developer.android.com/google/play/billing/rtdn-reference
    
    notification_type = body.get('oneTimeProductNotification', {}).get('notificationType')
    
    if notification_type == 2:  # ONE_TIME_PRODUCT_CANCELED (refunded)
        purchase_token = body.get('oneTimeProductNotification', {}).get('purchaseToken')
        
        # Find and mark the purchase as refunded
        await db.execute(
            """
            UPDATE purchases 
            SET purchase_state = 'refunded' 
            WHERE purchase_token = :token
            """,
            {"token": hashlib.sha256(purchase_token.encode()).hexdigest()}
        )
        
        # Optionally: Deduct gems from user
        # This depends on your business policy
    
    return {"status": "ok"}
```

---

## Checklist

- [ ] Enable Google Play Developer API in Google Cloud Console
- [ ] Create and link service account
- [ ] Download service account JSON key
- [ ] Create `purchases` table with unique `order_id` constraint
- [ ] Create `ad_free_purchases` table
- [ ] Add `is_ad_free` column to users table
- [ ] Implement `/auth/verify-gem-purchase` endpoint
- [ ] Implement `/auth/verify-ad-free-purchase` endpoint
- [ ] Implement `/auth/ad-free-status` endpoint
- [ ] Add license testers in Play Console for testing
- [ ] Test all purchase scenarios
- [ ] Set up RTDN webhooks for refund handling (optional)
- [ ] Add monitoring/alerting for verification failures

---

## Frontend Reference

The frontend sends purchases with this flow:
1. User selects gem pack → Initiates purchase via Google Play
2. Google Play handles payment → Returns `PurchaseDetails`
3. Frontend extracts `purchaseToken` and `orderId`
4. Frontend calls `/auth/verify-gem-purchase` with token
5. Backend verifies, credits gems, returns new balance
6. Frontend updates UI with new balance

**Key fields from frontend:**
- `purchase_token`: From `GooglePlayPurchaseDetails.billingClientPurchase.purchaseToken`
- `order_id`: From `GooglePlayPurchaseDetails.billingClientPurchase.orderId`
- `product_id`: The product ID (e.g., `credits_500`)
- `platform`: Always `"android"` for Google Play

---

## Config Integrity (tool_config.json)

### Why It Matters
The `tool_config.json` contains AI model configurations, costs, and capabilities. While the backend validates actual costs during job submission, we want to prevent users from seeing tampered/incorrect info.

### Implementation

**1. Generate hash when uploading config:**
```bash
# When uploading tool_config.json to R2
sha256sum tool_config.json | awk '{print $1}' > tool_config.json.sha256
# Upload both files to R2
```

**2. Or via Python script:**
```python
import hashlib
import json

def generate_config_hash(config_path: str) -> str:
    """Generate SHA256 hash for tool_config.json"""
    with open(config_path, 'rb') as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()

# Generate and save
config_hash = generate_config_hash('tool_config.json')
with open('tool_config.json.sha256', 'w') as f:
    f.write(config_hash)

print(f"Hash: {config_hash}")
```

**3. Upload both files to R2:**
```
https://your-r2-bucket.r2.dev/tool_config.json
https://your-r2-bucket.r2.dev/tool_config.json.sha256
```

### Frontend Verification Flow
1. Download `tool_config.json` and `tool_config.json.sha256`
2. Compute SHA256 of downloaded config
3. Compare with server-provided hash
4. If mismatch → reject config (possible tampering)
5. Cache verified config with hash for future integrity checks

### What This Prevents
- **Man-in-the-middle attacks**: Tampered config during download is detected
- **Local tampering**: Modified cached config fails integrity check
- **Spoofed configs**: Only configs signed by your server are accepted

### Limitations
- This is client-side verification (defense in depth)
- Backend MUST still validate all costs/parameters during job submission
- Backend is the ultimate source of truth for pricing
