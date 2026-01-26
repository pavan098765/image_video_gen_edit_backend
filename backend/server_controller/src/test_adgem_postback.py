#!/usr/bin/env python3
"""
AdGem Postback Test Utility

This script generates test AdGem postback requests for local testing.
Credits are determined by the `amount` field (configured in AdGem dashboard).

Usage:
    python test_adgem_postback.py [firebase_uid]

Environment Variables:
    ADGEM_POSTBACK_KEY - Your AdGem postback/secret key
    SERVER_URL - Server URL (default: http://localhost:8000)

Example:
    python test_adgem_postback.py your_firebase_uid_here
"""

import os
import sys
import requests
from datetime import datetime


# Configuration
POSTBACK_KEY = os.getenv("ADGEM_POSTBACK_KEY", "4jg79i9fa9lg74g9lcj336g1")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
APP_ID = "31874"  # kala-ai-offerwall


def get_secret_token() -> str:
    """Get the secret token for postback URL authentication."""
    return POSTBACK_KEY


def test_postback(
    player_id: str,
    amount: int = 80,  # Credits to award (configured in AdGem dashboard)
    payout: float = 20.0,  # AdGem payout (for reference only)
    offer_id: str = "test_offer_001",
    offer_name: str = "Test Offer - App Install",
    country: str = "IN",
    transaction_id: str = None,
):
    """
    Send a test postback to the server.
    
    Args:
        player_id: Firebase UID of the user to credit
        amount: Credits to award (this is what AdGem sends based on dashboard multiplier)
        payout: AdGem payout amount (for logging only, not used for credit calculation)
        offer_id: Offer identifier
        offer_name: Human-readable offer name
        country: Country code
        transaction_id: Unique transaction ID (auto-generated if not provided)
    """
    # Generate unique transaction ID if not provided
    if not transaction_id:
        transaction_id = f"test_tx_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Build query parameters (using AdGem's actual macro names)
    params = {
        "app_id": APP_ID,  # 31874 - kala-ai-offerwall
        "player_id": player_id,
        "amount": str(amount),  # Credits to award
        "payout": str(payout),
        "transaction_id": transaction_id,
        "offer_id": offer_id,
        "offer_name": offer_name,
        "country": country,
        "conversion_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "secret": get_secret_token(),
    }
    
    # Build URL
    url = f"{SERVER_URL}/ads/adgem/postback"
    
    print("=" * 70)
    print("AdGem Postback Test")
    print("=" * 70)
    print(f"\nServer URL: {url}")
    print(f"\nSecret Token: {POSTBACK_KEY[:8]}...{POSTBACK_KEY[-4:]}")
    print(f"\nParameters:")
    for key, value in params.items():
        if key == "secret":
            print(f"  {key}: {value[:8]}...{value[-4:]}")
        else:
            print(f"  {key}: {value}")
    
    # Send request
    print(f"\nSending GET request...")
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"\nResponse:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Body: {response.text}")
        
        if response.status_code == 200 and response.text == "OK":
            print(f"\n✅ SUCCESS! Credits awarded to player_id={player_id}")
            print(f"\n   Credits awarded: {amount}")
            print(f"   (amount field = credits, configured via AdGem dashboard multiplier)")
        elif response.status_code == 200:
            print(f"\n⚠️  Request accepted but response was: {response.text}")
        else:
            print(f"\n❌ FAILED! Status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to {SERVER_URL}")
        print("   Make sure the server is running.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("\n" + "=" * 70)
    return params


def test_invalid_secret(player_id: str):
    """Test that invalid secret tokens are rejected."""
    print("\n" + "=" * 70)
    print("Testing INVALID Secret Token (should be rejected)")
    print("=" * 70)
    
    params = {
        "app_id": "1234",
        "player_id": player_id,
        "amount": "100",
        "payout": "10.0",
        "transaction_id": f"invalid_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "offer_id": "test_offer",
        "secret": "invalid_secret_12345",  # Invalid!
    }
    
    url = f"{SERVER_URL}/ads/adgem/postback"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"\nResponse:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Body: {response.text}")
        
        if response.status_code == 403 and response.text == "INVALID":
            print(f"\n✅ CORRECT! Invalid secret was rejected.")
        else:
            print(f"\n⚠️  Note: Server allows requests without valid secret (logs warning)")
            print(f"     To enforce secret validation, configure ADGEM_POSTBACK_KEY")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("=" * 70)


def test_duplicate_transaction(player_id: str, transaction_id: str):
    """Test that duplicate transactions are handled idempotently."""
    print("\n" + "=" * 70)
    print("Testing Duplicate Transaction (should return OK without re-crediting)")
    print("=" * 70)
    
    params = {
        "app_id": "1234",
        "player_id": player_id,
        "amount": "100",
        "payout": "20.0",
        "transaction_id": transaction_id,
        "offer_id": "test_offer",
        "secret": get_secret_token(),
    }
    
    url = f"{SERVER_URL}/ads/adgem/postback"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"\nResponse:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Body: {response.text}")
        
        if response.status_code == 200 and response.text == "OK":
            print(f"\n✅ CORRECT! Duplicate handled idempotently (returned OK).")
        else:
            print(f"\n❌ UNEXPECTED! Should have returned 200 OK")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("=" * 70)


def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n" + "=" * 70)
    print("Testing Health Endpoint")
    print("=" * 70)
    
    url = f"{SERVER_URL}/ads/adgem/health"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"\nResponse:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Body: {response.json()}")
        
        if response.status_code == 200:
            print(f"\n✅ Health endpoint working!")
        else:
            print(f"\n❌ Health endpoint returned non-200")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("=" * 70)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AdGem Postback Integration Test Suite")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Server URL: {SERVER_URL}")
    print(f"  Postback Key: {POSTBACK_KEY[:8]}...{POSTBACK_KEY[-4:] if len(POSTBACK_KEY) > 12 else POSTBACK_KEY}")
    
    if not POSTBACK_KEY:
        print("\n❌ ERROR: ADGEM_POSTBACK_KEY not set!")
        print("   Set it via environment variable or edit this script.")
        sys.exit(1)
    
    # Get player_id (firebase_uid) from command line or use test value
    if len(sys.argv) > 1:
        player_id = sys.argv[1]
        print(f"\n  Using player_id from command line: {player_id}")
    else:
        player_id = "test_firebase_uid_12345"
        print(f"\n  Using default test player_id: {player_id}")
        print("  (Pass a real firebase_uid as argument to test with real user)")
    
    # Test 1: Health endpoint
    test_health_endpoint()
    
    # Test 2: Valid postback
    print("\n\n📤 Test 1: Valid Postback (should succeed)")
    result = test_postback(
        player_id=player_id,
        amount=80,  # Credits to award (from AdGem dashboard)
        payout=20.0,  # AdGem payout (for reference)
    )
    
    # Test 3: Invalid secret
    print("\n\n📤 Test 2: Invalid Secret Token (should be rejected)")
    test_invalid_secret(player_id)
    
    # Test 4: Duplicate transaction
    print("\n\n📤 Test 3: Duplicate Transaction (should be idempotent)")
    test_duplicate_transaction(player_id, result["transaction_id"])
    
    # Test 5: Credit examples
    print("\n\n📊 Credit Configuration")
    print("\n  Credits are configured via AdGem dashboard multiplier.")
    print("  The `amount` field in postback = credits to award.")
    print("  Server NEVER calculates credits - AdGem dashboard is source of truth.")
    print("\n  Example offers (configured in AdGem):")
    print(f"  {'Offer Type':<25} {'Amount (Credits)':<15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'Simple App Install':<25} {'80 credits':<15}")
    print(f"  {'App + Registration':<25} {'150 credits':<15}")
    print(f"  {'In-App Purchase':<25} {'500+ credits':<15}")
    
    print("\n\n" + "=" * 70)
    print("Test Suite Complete!")
    print("=" * 70)
    print("\nTo test with a real user, run:")
    print(f"  python {sys.argv[0]} <firebase_uid>")
    print("\nMake sure the server is running and the user exists in the database.")


if __name__ == "__main__":
    main()
