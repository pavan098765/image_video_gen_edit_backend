# purchase_verification.py
"""
In-App Purchase Verification Service

Provides server-side verification for:
- Google Play (Android) purchases
- App Store (iOS) purchases

CRITICAL: Never trust client-side purchase claims.
Always verify with the store before granting entitlements.

Purchase Types:
- Consumables (gem/credit packs): Verify → Credit → Consume
- Non-consumables (ad-free): Verify → Activate → Acknowledge
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger("purchase_verification")


# =============================================================================
# Configuration
# =============================================================================

# Google Play
GOOGLE_PLAY_PACKAGE_NAME = os.getenv("GOOGLE_PLAY_PACKAGE_NAME", "com.shambhutech.kalaai")
GOOGLE_PLAY_SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_PLAY_SERVICE_ACCOUNT_PATH",
    "/app/secrets/google-play-service-account.json"
)

# App Store
APP_STORE_SHARED_SECRET = os.getenv("APP_STORE_SHARED_SECRET", "")
APP_STORE_SANDBOX = os.getenv("APP_STORE_SANDBOX", "false").lower() == "true"

# =============================================================================
# Product Configuration - CRITICAL: Server-side gem amounts
# =============================================================================

# Product ID to credits/gems mapping (NEVER trust client-sent amounts)
# These must match Play Console product IDs exactly
PRODUCT_CREDITS_MAP = {
    "credits_50": 50,
    "credits_100": 100,
    "credits_500": 500,
    "credits_1000": 1000,
    "credits_2500": 2500,
    "credits_5000": 5000,
    "credits_10000": 10000,
}

# Valid product IDs for ad-free purchase
# NOTE: remove_ads_lifetime_v2 is CONSUMABLE on Play Store but controlled by backend
# This allows each Firebase account to purchase independently
VALID_AD_FREE_PRODUCT_IDS = [
    "remove_ads_lifetime_v2",  # NEW: Consumable product - one per Firebase account
    "remove_ads_lifetime",     # Legacy: Non-consumable (deprecated)
    "remove_ads",              # Legacy
    "ad_free_lifetime",        # Legacy
]

# The primary ad-free product ID (consumable)
PRIMARY_AD_FREE_PRODUCT_ID = "remove_ads_lifetime_v2"

# =============================================================================
# VIP Subscription Products
# =============================================================================

# VIP product IDs to duration mapping (in days)
# VIP benefits: unlimited concurrent generations, bulk edit, free upscale
# VIP is TIME-LIMITED unlike ad-free which is permanent
VIP_PRODUCT_DURATION_MAP = {
    "vip_pack_weekly": 7,           # 7 days VIP
    "vip_pack_monthly": 30,         # 30 days VIP
    "vip_pack_yearly": 365,         # 365 days VIP (best value)
}

# List of valid VIP product IDs
VALID_VIP_PRODUCT_IDS = list(VIP_PRODUCT_DURATION_MAP.keys())

# Primary VIP product ID
PRIMARY_VIP_PRODUCT_ID = "vip_monthly"


def get_vip_duration_days(product_id: str) -> int:
    """
    Get the VIP duration in days for a product ID.
    Returns 0 if product is not a VIP product.
    
    CRITICAL: This is the authoritative source for VIP duration.
    Never use client-provided duration.
    """
    return VIP_PRODUCT_DURATION_MAP.get(product_id, 0)


def is_vip_product(product_id: str) -> bool:
    """Check if a product is a VIP subscription product"""
    return product_id in VIP_PRODUCT_DURATION_MAP


# All valid product IDs (combined)
ALL_VALID_PRODUCT_IDS = (
    list(PRODUCT_CREDITS_MAP.keys()) + 
    VALID_AD_FREE_PRODUCT_IDS + 
    VALID_VIP_PRODUCT_IDS
)


def get_credits_for_product(product_id: str) -> int:
    """
    Get the credit amount for a product ID.
    Returns 0 if product is not a credit pack.
    
    CRITICAL: This is the authoritative source for credit amounts.
    Never use client-provided credit amounts.
    """
    return PRODUCT_CREDITS_MAP.get(product_id, 0)


def is_consumable_product(product_id: str) -> bool:
    """Check if a product is consumable (credits) vs non-consumable (ad-free)"""
    return product_id in PRODUCT_CREDITS_MAP


def is_ad_free_product(product_id: str) -> bool:
    """Check if a product is the ad-free purchase"""
    return product_id in VALID_AD_FREE_PRODUCT_IDS


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class VerificationResult:
    """Result of purchase verification"""
    valid: bool
    order_id: Optional[str] = None
    purchase_time: Optional[str] = None
    purchase_time_ms: Optional[int] = None  # Milliseconds since epoch
    reason: Optional[str] = None
    store_data: Optional[Dict[str, Any]] = None
    consumed: bool = False  # Whether the purchase was already consumed
    acknowledged: bool = False  # Whether the purchase was acknowledged
    is_test_purchase: bool = False  # purchaseType == 0 indicates test purchase
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "order_id": self.order_id,
            "purchase_time": self.purchase_time,
            "purchase_time_ms": self.purchase_time_ms,
            "reason": self.reason,
            "consumed": self.consumed,
            "acknowledged": self.acknowledged,
            "is_test_purchase": self.is_test_purchase,
        }


# =============================================================================
# Google Play Verification
# =============================================================================

_google_play_service = None


def _get_google_play_service():
    """
    Get or create Google Play Developer API service.
    
    Uses service account credentials for server-to-server auth.
    """
    global _google_play_service
    
    if _google_play_service is not None:
        return _google_play_service
    
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        # Check if service account file exists
        if not os.path.exists(GOOGLE_PLAY_SERVICE_ACCOUNT_PATH):
            logger.error(f"Google Play service account file not found: {GOOGLE_PLAY_SERVICE_ACCOUNT_PATH}")
            return None
        
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_PLAY_SERVICE_ACCOUNT_PATH,
            scopes=['https://www.googleapis.com/auth/androidpublisher']
        )
        
        _google_play_service = build('androidpublisher', 'v3', credentials=credentials)
        logger.info("Google Play Developer API service initialized")
        return _google_play_service
        
    except ImportError as e:
        logger.error(f"Google API client libraries not installed: {e}")
        logger.error("Install with: pip install google-auth google-auth-oauthlib google-api-python-client")
        return None
    except Exception as e:
        logger.exception(f"Failed to initialize Google Play service: {e}")
        return None


def verify_google_play_purchase(
    product_id: str,
    purchase_token: str,
    package_name: Optional[str] = None,
) -> VerificationResult:
    """
    Verify a Google Play one-time purchase.
    
    Args:
        product_id: The product ID (e.g., 'remove_ads_lifetime' or 'credits_500')
        purchase_token: The purchase token from the client
        package_name: Android package name (optional, uses default)
    
    Returns:
        VerificationResult with validation status and details
        
    Response fields from Google:
        - purchaseState: 0=Purchased, 1=Canceled, 2=Pending
        - consumptionState: 0=Not consumed, 1=Consumed
        - acknowledgementState: 0=Not acknowledged, 1=Acknowledged
        - purchaseType: 0=Test, 1=Promo, absent=Normal purchase
        - orderId: The Google Play order ID (unique per purchase)
        - purchaseTimeMillis: Timestamp in milliseconds
    """
    package_name = package_name or GOOGLE_PLAY_PACKAGE_NAME
    
    logger.info(f"Verifying Google Play purchase: product={product_id}, package={package_name}")
    
    service = _get_google_play_service()
    if service is None:
        return VerificationResult(
            valid=False,
            reason="Google Play API not configured"
        )
    
    try:
        # For non-subscription (one-time) purchases, use purchases().products().get()
        result = service.purchases().products().get(
            packageName=package_name,
            productId=product_id,
            token=purchase_token
        ).execute()
        
        logger.debug(f"Google Play API response: {result}")
        
        # Parse response fields
        purchase_state = result.get('purchaseState')
        consumption_state = result.get('consumptionState', 0)
        acknowledgement_state = result.get('acknowledgementState', 0)
        purchase_type = result.get('purchaseType')  # 0=Test, 1=Promo, absent=Normal
        order_id = result.get('orderId', '')
        purchase_time_ms = result.get('purchaseTimeMillis')
        
        # Determine if this is a test purchase
        is_test = purchase_type == 0
        
        if purchase_state == 0:
            # Purchase is valid (state 0 = Purchased)
            return VerificationResult(
                valid=True,
                order_id=order_id,
                purchase_time=str(purchase_time_ms) if purchase_time_ms else None,
                purchase_time_ms=int(purchase_time_ms) if purchase_time_ms else None,
                consumed=consumption_state == 1,
                acknowledged=acknowledgement_state == 1,
                is_test_purchase=is_test,
                store_data={
                    "purchaseState": purchase_state,
                    "consumptionState": consumption_state,
                    "acknowledgementState": acknowledgement_state,
                    "purchaseType": purchase_type,
                    "developerPayload": result.get('developerPayload'),
                    "orderId": order_id,
                    "purchaseTimeMillis": purchase_time_ms,
                    "regionCode": result.get('regionCode'),
                    "quantity": result.get('quantity', 1),
                }
            )
        elif purchase_state == 1:
            return VerificationResult(
                valid=False,
                reason="Purchase was cancelled",
                order_id=order_id,
                store_data={"purchaseState": purchase_state, "orderId": order_id}
            )
        elif purchase_state == 2:
            return VerificationResult(
                valid=False,
                reason="Purchase is pending payment",
                order_id=order_id,
                store_data={"purchaseState": purchase_state, "orderId": order_id}
            )
        else:
            return VerificationResult(
                valid=False,
                reason=f"Unknown purchase state: {purchase_state}",
                store_data=result
            )
            
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Google Play verification failed: {error_msg}")
        
        # Check for common errors
        if "404" in error_msg or "not found" in error_msg.lower():
            return VerificationResult(
                valid=False,
                reason="Purchase not found - invalid token or product ID"
            )
        elif "401" in error_msg or "403" in error_msg:
            return VerificationResult(
                valid=False,
                reason="API authentication failed - check service account"
            )
        
        return VerificationResult(
            valid=False,
            reason=f"Verification error: {error_msg}"
        )


def acknowledge_google_play_purchase(
    product_id: str,
    purchase_token: str,
    package_name: Optional[str] = None,
    developer_payload: Optional[str] = None,
) -> bool:
    """
    Acknowledge a Google Play purchase.
    
    IMPORTANT: Non-consumable purchases MUST be acknowledged within 3 days
    or Google will automatically refund the purchase.
    
    For consumables, use consume_google_play_purchase() instead.
    
    Args:
        product_id: The product ID
        purchase_token: The purchase token
        package_name: Android package name (optional)
        developer_payload: Optional payload to store with acknowledgement
    
    Returns:
        True if acknowledged successfully, False otherwise
    """
    package_name = package_name or GOOGLE_PLAY_PACKAGE_NAME
    
    logger.info(f"Acknowledging Google Play purchase: product={product_id}")
    
    service = _get_google_play_service()
    if service is None:
        logger.error("Cannot acknowledge: Google Play API not configured")
        return False
    
    try:
        body = {}
        if developer_payload:
            body['developerPayload'] = developer_payload
            
        service.purchases().products().acknowledge(
            packageName=package_name,
            productId=product_id,
            token=purchase_token,
            body=body
        ).execute()
        
        logger.info(f"Successfully acknowledged purchase: product={product_id}")
        return True
        
    except Exception as e:
        error_str = str(e)
        
        # Expected errors - log as simple warning without traceback
        expected_errors = [
            "productNotOwnedByUser",      # Client already handled
            "alreadyAcknowledged",        # Already acknowledged
            "purchaseAlreadyAcknowledged", # Duplicate acknowledge
        ]
        
        is_expected = any(err in error_str for err in expected_errors)
        
        if is_expected:
            logger.warning(
                f"Acknowledge skipped (already done): product={product_id} - "
                f"This is expected when the app handles acknowledgement"
            )
        else:
            # Unexpected error - log full traceback
            logger.exception(f"Failed to acknowledge purchase: {e}")
        return False


def consume_google_play_purchase(
    product_id: str,
    purchase_token: str,
    package_name: Optional[str] = None,
) -> bool:
    """
    Consume a Google Play consumable purchase.
    
    IMPORTANT: Consumable purchases (like credit packs) MUST be consumed
    to allow the user to purchase the same product again.
    
    Consuming also implicitly acknowledges the purchase.
    
    Args:
        product_id: The product ID
        purchase_token: The purchase token
        package_name: Android package name (optional)
    
    Returns:
        True if consumed successfully, False otherwise
    """
    package_name = package_name or GOOGLE_PLAY_PACKAGE_NAME
    
    logger.info(f"Consuming Google Play purchase: product={product_id}")
    
    service = _get_google_play_service()
    if service is None:
        logger.error("Cannot consume: Google Play API not configured")
        return False
    
    try:
        service.purchases().products().consume(
            packageName=package_name,
            productId=product_id,
            token=purchase_token
        ).execute()
        
        logger.info(f"Successfully consumed purchase: product={product_id}")
        return True
        
    except Exception as e:
        error_str = str(e)
        
        # Expected errors - log as simple warning without traceback
        # These occur when client already consumed/completed the purchase
        expected_errors = [
            "productNotOwnedByUser",      # Client already consumed
            "alreadyOwned",               # Already processed
            "productAlreadyOwned",        # Duplicate consume attempt
        ]
        
        is_expected = any(err in error_str for err in expected_errors)
        
        if is_expected:
            # Simple warning - this is normal when client handles consumption
            logger.warning(
                f"Consume skipped (client likely handled): product={product_id} - "
                f"This is expected when the app calls completePurchase()"
            )
        else:
            # Unexpected error - log full traceback for debugging
            logger.exception(f"Failed to consume purchase: {e}")
        
        return False


# =============================================================================
# App Store Verification
# =============================================================================

def verify_app_store_purchase(
    receipt_data: str,
    product_id: Optional[str] = None,
    is_sandbox: Optional[bool] = None,
) -> VerificationResult:
    """
    Verify an App Store purchase using receipt validation.
    
    Args:
        receipt_data: Base64-encoded receipt data from the client
        product_id: Expected product ID to verify (optional)
        is_sandbox: Force sandbox mode (optional, auto-detects if None)
    
    Returns:
        VerificationResult with validation status and details
    """
    if is_sandbox is None:
        is_sandbox = APP_STORE_SANDBOX
    
    logger.info(f"Verifying App Store purchase: sandbox={is_sandbox}")
    
    # App Store receipt verification endpoints
    production_url = "https://buy.itunes.apple.com/verifyReceipt"
    sandbox_url = "https://sandbox.itunes.apple.com/verifyReceipt"
    
    url = sandbox_url if is_sandbox else production_url
    
    payload = {
        "receipt-data": receipt_data,
        "exclude-old-transactions": True,
    }
    
    # Add shared secret if configured
    if APP_STORE_SHARED_SECRET:
        payload["password"] = APP_STORE_SHARED_SECRET
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        status = result.get("status")
        logger.debug(f"App Store API response status: {status}")
        
        # Status codes:
        # 0 = Valid receipt
        # 21007 = Sandbox receipt sent to production (retry with sandbox)
        # 21008 = Production receipt sent to sandbox
        # Other = Various errors
        
        if status == 0:
            # Valid receipt - find the specific product
            receipt = result.get("receipt", {})
            in_app_purchases = receipt.get("in_app", [])
            
            # If product_id specified, verify it exists in receipt
            if product_id:
                for purchase in in_app_purchases:
                    if purchase.get("product_id") == product_id:
                        return VerificationResult(
                            valid=True,
                            order_id=purchase.get("transaction_id"),
                            purchase_time=purchase.get("purchase_date"),
                            store_data={
                                "transaction_id": purchase.get("transaction_id"),
                                "original_transaction_id": purchase.get("original_transaction_id"),
                                "product_id": purchase.get("product_id"),
                                "purchase_date": purchase.get("purchase_date"),
                                "quantity": purchase.get("quantity"),
                            }
                        )
                
                return VerificationResult(
                    valid=False,
                    reason=f"Product {product_id} not found in receipt"
                )
            
            # No specific product requested, check for any valid ad-free product
            for purchase in in_app_purchases:
                if purchase.get("product_id") in VALID_AD_FREE_PRODUCT_IDS:
                    return VerificationResult(
                        valid=True,
                        order_id=purchase.get("transaction_id"),
                        purchase_time=purchase.get("purchase_date"),
                        store_data={
                            "transaction_id": purchase.get("transaction_id"),
                            "original_transaction_id": purchase.get("original_transaction_id"),
                            "product_id": purchase.get("product_id"),
                            "purchase_date": purchase.get("purchase_date"),
                        }
                    )
            
            return VerificationResult(
                valid=False,
                reason="No valid ad-free product found in receipt"
            )
        
        elif status == 21007:
            # Sandbox receipt sent to production - retry with sandbox
            logger.info("Sandbox receipt detected, retrying with sandbox endpoint")
            return verify_app_store_purchase(receipt_data, product_id, is_sandbox=True)
        
        elif status == 21008:
            # Production receipt sent to sandbox - retry with production
            logger.info("Production receipt detected, retrying with production endpoint")
            return verify_app_store_purchase(receipt_data, product_id, is_sandbox=False)
        
        else:
            # Map common status codes to error messages
            status_messages = {
                21000: "App Store could not read receipt",
                21002: "Receipt data malformed",
                21003: "Receipt could not be authenticated",
                21004: "Shared secret mismatch",
                21005: "Receipt server unavailable",
                21006: "Receipt valid but subscription expired",
                21010: "This receipt could not be authorized",
            }
            
            reason = status_messages.get(status, f"Unknown status: {status}")
            return VerificationResult(
                valid=False,
                reason=reason,
                store_data={"status": status}
            )
            
    except requests.exceptions.Timeout:
        return VerificationResult(
            valid=False,
            reason="App Store verification timed out"
        )
    except requests.exceptions.RequestException as e:
        return VerificationResult(
            valid=False,
            reason=f"Network error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"App Store verification failed: {e}")
        return VerificationResult(
            valid=False,
            reason=f"Verification error: {str(e)}"
        )


# =============================================================================
# Unified Verification Function
# =============================================================================

def verify_purchase(
    purchase_token: str,
    product_id: str,
    platform: str,
    package_name: Optional[str] = None,
) -> VerificationResult:
    """
    Unified purchase verification for any platform.
    
    Validates the product_id against known valid products and verifies
    the purchase with the appropriate store API.
    
    Args:
        purchase_token: Purchase token or receipt data
        product_id: Product ID to verify
        platform: 'android' or 'ios'
        package_name: Android package name (optional)
    
    Returns:
        VerificationResult with validation status
    """
    # Validate product ID against all valid products
    if product_id not in ALL_VALID_PRODUCT_IDS:
        logger.warning(f"Invalid product ID attempted: {product_id}")
        return VerificationResult(
            valid=False,
            reason=f"Invalid product ID: {product_id}"
        )
    
    platform = platform.lower().strip()
    
    if platform == "android":
        return verify_google_play_purchase(
            product_id=product_id,
            purchase_token=purchase_token,
            package_name=package_name,
        )
    elif platform == "ios":
        return verify_app_store_purchase(
            receipt_data=purchase_token,
            product_id=product_id,
        )
    else:
        return VerificationResult(
            valid=False,
            reason=f"Unsupported platform: {platform}"
        )


def verify_and_process_consumable(
    purchase_token: str,
    product_id: str,
    platform: str,
    package_name: Optional[str] = None,
) -> VerificationResult:
    """
    Verify a consumable purchase and mark it for consumption.
    
    This is a convenience wrapper that:
    1. Verifies the purchase
    2. Optionally consumes it (for Google Play)
    
    Note: Consumption should be done AFTER crediting the user to ensure
    they receive their credits even if consumption fails.
    
    Args:
        purchase_token: Purchase token
        product_id: Product ID (must be a consumable)
        platform: 'android' or 'ios'
        package_name: Android package name (optional)
    
    Returns:
        VerificationResult with validation status
    """
    if not is_consumable_product(product_id):
        return VerificationResult(
            valid=False,
            reason=f"Product {product_id} is not a consumable product"
        )
    
    return verify_purchase(
        purchase_token=purchase_token,
        product_id=product_id,
        platform=platform,
        package_name=package_name,
    )
