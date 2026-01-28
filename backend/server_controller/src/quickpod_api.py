# quickpod_api.py
"""
QuickPod API Wrapper

Provides:
- Authentication and token management
- Pod lifecycle operations (create, start, stop, destroy)
- Machine listing and offer selection
- Exponential backoff for retries
- Better error handling (distinguish fatal vs transient)
- Logging of API responses
"""

import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import traceback
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from exceptions import (
    QuickPodAPIError,
    QuickPodFatalError,
    PodCreationError,
    PodDestroyError,
)

logger = logging.getLogger("quickpod")


# =============================================================================
# Offer Info Dataclass (for cost tracking)
# =============================================================================

@dataclass
class OfferInfo:
    """Information about a QuickPod offer including pricing."""
    offer_id: int
    machine_id: int
    hourly_cost: float  # $/hour
    gpu_type: str  # e.g., "NVIDIA GeForce RTX 4090"
    gpu_vram: int  # MB
    tflops: float
    memory: float  # GB RAM


# =============================================================================
# Configuration
# =============================================================================

QUICKPOD_BASE = os.getenv("QUICKPOD_BASE", "https://api.quickpod.org")
BASE_URL = "https://api.quickpod.org/update"
RENTABLE_URL = "https://api.quickpod.org/rentable"
MYPODS_URL = "https://api.quickpod.org/mypods"

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 30.0
RETRY_BACKOFF_FACTOR = 2.0

# Request timeouts
DEFAULT_TIMEOUT = 30
LOGIN_TIMEOUT = 15
POD_OPERATION_TIMEOUT = 60


class QuickPodErrorType(str, Enum):
    """Types of QuickPod errors"""
    TRANSIENT = "transient"  # Network issues, temporary failures
    FATAL = "fatal"  # Database errors, invalid requests
    AUTH = "auth"  # Authentication failures
    RATE_LIMIT = "rate_limit"  # Too many requests
    UNKNOWN = "unknown"


# =============================================================================
# Error Classification
# =============================================================================

def classify_error(error: Exception, response: Optional[requests.Response] = None) -> QuickPodErrorType:
    """
    Classify an error as transient or fatal.
    
    Args:
        error: The exception that occurred
        response: Optional response object
        
    Returns:
        QuickPodErrorType classification
    """
    error_msg = str(error).lower()
    
    # Check response status code
    if response is not None:
        status = response.status_code
        
        # Authentication errors
        if status == 401:
            return QuickPodErrorType.AUTH
        
        # Rate limiting
        if status == 429:
            return QuickPodErrorType.RATE_LIMIT
        
        # Fatal database/validation errors
        if status == 406:
            return QuickPodErrorType.FATAL
        
        # Server errors are usually transient
        if 500 <= status < 600:
            return QuickPodErrorType.TRANSIENT
        
        # Client errors other than auth are usually fatal
        if 400 <= status < 500:
            return QuickPodErrorType.FATAL
    
    # Check error message for known patterns
    if "database query failed" in error_msg or "406" in error_msg:
        return QuickPodErrorType.FATAL
    
    if "timeout" in error_msg or "connection" in error_msg:
        return QuickPodErrorType.TRANSIENT
    
    if isinstance(error, (Timeout, ConnectionError)):
        return QuickPodErrorType.TRANSIENT
    
    return QuickPodErrorType.UNKNOWN


# =============================================================================
# Retry Logic
# =============================================================================

def calculate_retry_delay(attempt: int) -> float:
    """Calculate delay for retry with exponential backoff"""
    delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
    return min(delay, MAX_RETRY_DELAY)


def should_retry(error_type: QuickPodErrorType, attempt: int) -> bool:
    """Determine if a request should be retried"""
    if attempt >= MAX_RETRIES:
        return False
    
    # Only retry transient and unknown errors
    return error_type in (QuickPodErrorType.TRANSIENT, QuickPodErrorType.UNKNOWN)


# =============================================================================
# QuickPod Manager
# =============================================================================

class QuickPodManager:
    """
    QuickPod API client with improved error handling and retry logic.
    
    For pod-specific operations, instantiate with pod_uuid.
    
    Token caching: Uses a shared class-level token cache keyed by email.
    All instances with the same email share the same token, avoiding
    redundant logins. Token auto-refreshes when expired (TTL: 6 hours).
    """
    
    # -------------------------------------------------------------------------
    # CLASS-LEVEL TOKEN CACHE (shared across all instances)
    # -------------------------------------------------------------------------
    # Format: {email: {"token": str, "acquired_at": datetime, "ttl": timedelta}}
    _token_cache: Dict[str, Dict[str, Any]] = {}
    _token_cache_lock = None  # Will use threading.Lock for thread safety
    
    @classmethod
    def _get_cached_token(cls, email: str) -> Optional[Tuple[str, datetime, timedelta]]:
        """Get cached token for email if still valid."""
        import threading
        if cls._token_cache_lock is None:
            cls._token_cache_lock = threading.Lock()
        
        with cls._token_cache_lock:
            cache_entry = cls._token_cache.get(email)
            if not cache_entry:
                return None
            
            token = cache_entry.get("token")
            acquired_at = cache_entry.get("acquired_at")
            ttl = cache_entry.get("ttl", timedelta(hours=6))
            
            if not token or not acquired_at:
                return None
            
            # Check if expired
            if (datetime.utcnow() - acquired_at) >= ttl:
                logger.debug(f"Cached token for {email} expired")
                return None
            
            return (token, acquired_at, ttl)
    
    @classmethod
    def _set_cached_token(cls, email: str, token: str, acquired_at: datetime, ttl: timedelta):
        """Cache token for email (thread-safe)."""
        import threading
        if cls._token_cache_lock is None:
            cls._token_cache_lock = threading.Lock()
        
        with cls._token_cache_lock:
            cls._token_cache[email] = {
                "token": token,
                "acquired_at": acquired_at,
                "ttl": ttl,
            }
            logger.debug(f"Cached token for {email} (TTL: {ttl})")
    
    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        pod_uuid: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.email = email or os.getenv("QUICKPOD_EMAIL")
        self.password = password or os.getenv("QUICKPOD_PASSWORD")
        self.pod_uuid = pod_uuid
        self.base = api_base or QUICKPOD_BASE
        
        # Token management - check shared cache first
        cached = self._get_cached_token(self.email) if self.email else None
        if cached:
            self.token, self.token_acquired_at, self.token_ttl = cached
            logger.debug(f"QuickPodManager using cached token for {self.email}")
        else:
            self.token: Optional[str] = None
            self.token_acquired_at: Optional[datetime] = None
            self.token_ttl = timedelta(hours=6)
        
        logger.info(f"QuickPodManager initialized for {self.email} with pod_uuid={pod_uuid}")
    
    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    
    def login(self) -> str:
        """
        Authenticate with QuickPod and acquire a token.
        
        Stores token in shared class-level cache for reuse by other instances.
        
        Returns:
            Authentication token
            
        Raises:
            QuickPodAPIError: If login fails
        """
        url = f"{BASE_URL}/auth/login"
        payload = {"email": self.email, "password": self.password}
        
        try:
            resp = requests.post(url, json=payload, timeout=LOGIN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            self.token = data.get("authToken") or data.get("token") or data.get("jwt")
            if not self.token:
                raise QuickPodAPIError("login", reason=f"No token in response: {data}")
            
            self.token_acquired_at = datetime.utcnow()
            
            # Parse expiry if provided (QuickPod tokens typically valid for ~24h)
            expires_at = data.get("expires_at") or data.get("expiry")
            if expires_at:
                try:
                    self.token_ttl = datetime.fromisoformat(expires_at) - datetime.utcnow()
                except Exception:
                    pass
            
            # IMPORTANT: Store in shared cache for other instances to reuse
            if self.email:
                self._set_cached_token(
                    self.email, 
                    self.token, 
                    self.token_acquired_at, 
                    self.token_ttl
                )
            
            logger.info("QuickPod login successful, token acquired and cached")
            return self.token
            
        except RequestException as e:
            logger.error(f"QuickPod login failed: {e}")
            raise QuickPodAPIError("login", reason=str(e), cause=e)
    
    def _is_token_valid(self) -> bool:
        """Check if current token is still valid (instance or cached)"""
        # First check instance token
        if self.token and self.token_acquired_at:
            if (datetime.utcnow() - self.token_acquired_at) < self.token_ttl:
                return True
        
        # Check shared cache (another instance might have refreshed)
        if self.email:
            cached = self._get_cached_token(self.email)
            if cached:
                self.token, self.token_acquired_at, self.token_ttl = cached
                logger.debug(f"Loaded token from shared cache for {self.email}")
                return True
        
        return False
    
    def _should_refresh_token(self) -> bool:
        """Check if token should be refreshed (< 10 minutes remaining)"""
        if not self.token or not self.token_acquired_at:
            return True
        
        elapsed = datetime.utcnow() - self.token_acquired_at
        remaining = self.token_ttl - elapsed
        
        # Refresh if less than 10 minutes remaining
        return remaining < timedelta(minutes=10)
    
    def _auth_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        """Get authorization headers, refreshing token if needed"""
        # Only refresh if explicitly requested OR token is invalid/expiring soon
        if force_refresh or not self._is_token_valid() or self._should_refresh_token():
            # Double-check we actually need to login
            # Check cache one more time as another thread might have refreshed
            if self.email:
                cached = self._get_cached_token(self.email)
                if cached and not force_refresh:
                    self.token, self.token_acquired_at, self.token_ttl = cached
                    elapsed = datetime.utcnow() - self.token_acquired_at
                    remaining = self.token_ttl - elapsed
                    if remaining >= timedelta(minutes=10):
                        # Cache has a fresh enough token, use it
                        return {"Authorization": f"Bearer {self.token}"}
            
            self.login()
        return {"Authorization": f"Bearer {self.token}"}
    
    # -------------------------------------------------------------------------
    # Request Helper
    # -------------------------------------------------------------------------
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        retry_on_401: bool = True,
        **kwargs,
    ) -> Any:
        """
        Make an API request with error handling and retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            headers: Additional headers
            params: Query parameters
            timeout: Request timeout
            retry_on_401: Whether to retry on 401 with refreshed token
            **kwargs: Additional arguments for requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            QuickPodAPIError: For transient errors
            QuickPodFatalError: For non-recoverable errors
        """
        url = f"{self.base.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                merged_headers = self._auth_headers() or {}
                if headers:
                    merged_headers.update(headers)
                
                # Add default pod_uuid param if set
                if params is None and self.pod_uuid:
                    params = {"pod_uuid": self.pod_uuid}
                
                resp = requests.request(
                    method,
                    url,
                    headers=merged_headers,
                    params=params,
                    timeout=timeout,
                    **kwargs
                )
                
                # Handle 401 with token refresh
                if resp.status_code == 401 and retry_on_401:
                    logger.warning("401 from QuickPod, refreshing token and retrying")
                    merged_headers = self._auth_headers(force_refresh=True)
                    resp = requests.request(
                        method,
                        url,
                        headers=merged_headers,
                        params=params,
                        timeout=timeout,
                        **kwargs
                    )
                
                # Log response preview
                raw_text = resp.text[:2048]
                
                # Try to parse JSON
                try:
                    data = resp.json()
                except Exception:
                    logger.error(
                        f"QuickPod non-JSON response (status={resp.status_code}) "
                        f"for {method} {endpoint}: {raw_text}"
                    )
                    resp.raise_for_status()
                    return {"raw": raw_text}
                
                # Check for API error response
                if isinstance(data, dict) and "error" in data:
                    error_type = classify_error(
                        Exception(data.get("error")),
                        resp
                    )
                    
                    if error_type == QuickPodErrorType.FATAL:
                        raise QuickPodFatalError(
                            endpoint,
                            reason=f"{data.get('error')} (full: {json.dumps(data)})"
                        )
                    
                    raise QuickPodAPIError(
                        endpoint,
                        reason=f"{data.get('error')} (full: {json.dumps(data)})"
                    )
                
                resp.raise_for_status()
                return data
                
            except (QuickPodFatalError,):
                # Don't retry fatal errors
                raise
                
            except Exception as e:
                error_type = classify_error(e, getattr(e, 'response', None))
                
                if error_type == QuickPodErrorType.FATAL:
                    raise QuickPodFatalError(endpoint, reason=str(e), cause=e)
                
                if should_retry(error_type, attempt):
                    delay = calculate_retry_delay(attempt)
                    logger.warning(
                        f"QuickPod request failed (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                
                raise QuickPodAPIError(endpoint, reason=str(e), cause=e)
    
    # -------------------------------------------------------------------------
    # Template Management
    # -------------------------------------------------------------------------
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Fetch available templates from QuickPod API"""
        return self._make_request("GET", "templates")
    
    def get_template_by_name(self, template_name: str) -> Optional[str]:
        """
        Get template UUID by name.
        
        Args:
            template_name: Template name to search for
            
        Returns:
            Template UUID if found, None otherwise
        """
        templates = self.get_templates()
        for template in templates:
            if template.get("template_name") == template_name:
                return template.get("template_uuid")
        return None
    
    # -------------------------------------------------------------------------
    # Pod Operations
    # -------------------------------------------------------------------------
    
    def start_pod(self) -> Dict[str, Any]:
        """Start a stopped pod"""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot start pod.")
        
        logger.info(f"Starting pod {self.pod_uuid}")
        return self._make_request(
            "GET",
            "update/startpod",
            params={"pod_uuid": self.pod_uuid},
            timeout=POD_OPERATION_TIMEOUT,
        )
    
    def stop_pod(self) -> Dict[str, Any]:
        """Stop a running pod"""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot stop pod.")
        
        logger.info(f"Stopping pod {self.pod_uuid}")
        data = self._make_request(
            "GET",
            "update/stoppod",
            params={"pod_uuid": self.pod_uuid},
            timeout=POD_OPERATION_TIMEOUT,
        )
        # Some QuickPod endpoints return {"status": "..."} without "error".
        # Treat non-success statuses as failures so the caller doesn't log false success.
        if isinstance(data, dict) and data.get("status") and str(data.get("status")).lower() != "success":
            raise PodDestroyError(self.pod_uuid, reason=f"Stop pod failed: {data}")
        return data
    
    def restart_pod(self) -> Dict[str, Any]:
        """Restart a pod"""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot restart pod.")
        
        logger.info(f"Restarting pod {self.pod_uuid}")
        return self._make_request(
            "GET",
            "update/restartpod",
            params={"pod_uuid": self.pod_uuid},
            timeout=POD_OPERATION_TIMEOUT,
        )
    
    def get_my_pods(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch list of all current pods from /mypods endpoint.
        
        Uses authentication like other QuickPod API endpoints.
        
        Used to verify pod destruction - if pod is not in this list or has
        destroyed=true, it's confirmed destroyed.
        
        Returns:
            List of pod objects, or None if API call failed
        """
        try:
            # Use _make_request which handles authentication automatically
            # Pass empty params={} to avoid default pod_uuid param being added
            data = self._make_request(
                "GET",
                "mypods",  # Will be prefixed with self.base (https://api.quickpod.org)
                params={},  # Explicitly empty - /mypods doesn't need pod_uuid
                timeout=DEFAULT_TIMEOUT,
            )
            if isinstance(data, list):
                return data
            # Some responses wrap the list in an object
            if isinstance(data, dict) and "pods" in data:
                return data["pods"]
            logger.warning(f"Unexpected mypods response format: {type(data)}")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch my_pods list: {e}")
            return None
    
    def is_pod_destroyed(self, pod_uuid: Optional[str] = None) -> bool:
        """
        Verify if a pod is actually destroyed by checking /mypods.
        
        A pod is considered destroyed if:
        - It doesn't exist in the /mypods response
        - OR it exists with destroyed=true
        - OR it exists with state='exited' and intended_state='stopped'
        
        Args:
            pod_uuid: Pod UUID to check (defaults to self.pod_uuid)
            
        Returns:
            True if pod is confirmed destroyed, False if still active or API failed
        """
        check_uuid = pod_uuid or self.pod_uuid
        if not check_uuid:
            return True  # No pod to check
        
        try:
            pods = self.get_my_pods()
            
            # CRITICAL: If API call failed (returned None), assume NOT destroyed to be safe
            # This prevents false positives where we think a pod is destroyed when really
            # we just couldn't check (e.g., auth failure, network error)
            if pods is None:
                logger.warning(f"Cannot verify pod {check_uuid} destruction - API call failed, assuming NOT destroyed")
                return False
            
            for pod in pods:
                # Match by altname (which contains pod_uuid) or docker_ID
                # Ensure all values are strings for comparison
                pod_id = str(pod.get("altname", "") or "")
                docker_id = str(pod.get("docker_ID", "") or "")
                check_uuid_str = str(check_uuid)
                if check_uuid_str in pod_id or docker_id == check_uuid_str:
                    # Found the pod - check if it's destroyed
                    if pod.get("destroyed", False):
                        logger.info(f"Pod {check_uuid} confirmed destroyed (destroyed=true)")
                        return True
                    state = str(pod.get("State", "")).lower()
                    intended = str(pod.get("intended_state", "")).lower()
                    if state == "exited" and intended in ("stopped", "destroyed"):
                        logger.info(f"Pod {check_uuid} confirmed destroyed (state=exited, intended={intended})")
                        return True
                    # Pod still exists and is active
                    logger.warning(f"Pod {check_uuid} still active: state={state}, intended={intended}")
                    return False
            
            # Pod not found in list = destroyed (only valid if API call succeeded)
            logger.info(f"Pod {check_uuid} confirmed destroyed (not in mypods list)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to verify pod {check_uuid} destruction: {e}")
            # On error, assume not destroyed to be safe
            return False
    
    def destroy_pod(self, verify: bool = True, max_verify_attempts: int = 3) -> Dict[str, Any]:
        """
        Destroy and release a pod with verification.
        
        Args:
            verify: If True, verify destruction via /mypods endpoint
            max_verify_attempts: Max retry attempts if verification fails
        
        Returns:
            API response
            
        Raises:
            PodDestroyError: If destruction fails or cannot be verified
        """
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot destroy pod.")
        
        logger.info(f"Destroying pod {self.pod_uuid}")
        last_error = None
        
        for attempt in range(max_verify_attempts):
            try:
                data = self._make_request(
                    "GET",
                    "update/destroypod",
                    params={"pod_uuid": self.pod_uuid},
                    timeout=POD_OPERATION_TIMEOUT,
                )
                # Some QuickPod endpoints return {"status": "..."} without "error".
                # Treat non-success statuses as failures so the controller doesn't log false success.
                if isinstance(data, dict) and data.get("status") and str(data.get("status")).lower() != "success":
                    raise PodDestroyError(self.pod_uuid, reason=f"Destroy pod failed: {data}")
                
                # Verify destruction if requested
                if verify:
                    # Brief delay before verification to allow QuickPod to process
                    time.sleep(1.0)
                    if self.is_pod_destroyed():
                        logger.info(f"Pod {self.pod_uuid} destruction verified via /mypods")
                        return data
                    else:
                        # Pod still exists - retry destroy
                        if attempt < max_verify_attempts - 1:
                            logger.warning(
                                f"Pod {self.pod_uuid} destroy returned success but pod still active. "
                                f"Retrying ({attempt + 1}/{max_verify_attempts})..."
                            )
                            time.sleep(2.0 * (attempt + 1))  # Backoff
                            continue
                        else:
                            raise PodDestroyError(
                                self.pod_uuid, 
                                reason=f"Destroy returned success but pod still active after {max_verify_attempts} attempts"
                            )
                else:
                    return data
                    
            except PodDestroyError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_verify_attempts - 1:
                    logger.warning(f"Destroy attempt {attempt + 1} failed: {e}, retrying...")
                    logger.warning(traceback.format_exc())
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise PodDestroyError(self.pod_uuid, reason=str(e), cause=e)
        
        # Should not reach here, but handle edge case
        raise PodDestroyError(self.pod_uuid, reason=f"Max attempts reached: {last_error}")
    
    def delete_pod(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for destroy_pod (backward compatibility)"""
        return self.destroy_pod(*args, **kwargs)
    
    def get_logs(self) -> Dict[str, Any]:
        """Fetch pod logs for debugging"""
        if not self.pod_uuid:
            raise RuntimeError("No pod_uuid set. Cannot fetch logs.")
        
        return self._make_request(
            "GET",
            "update/podlogs",
            params={"pod_uuid": self.pod_uuid},
        )
    
    # -------------------------------------------------------------------------
    # Machine & Offer Management
    # -------------------------------------------------------------------------
    
    def get_rentable_machines(self) -> List[Dict[str, Any]]:
        """
        Fetch list of available rentable machines.
        
        Returns:
            List of machine offers
        """
        try:
            resp = requests.get(RENTABLE_URL, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            offers = resp.json()
        except Exception as e:
            raise QuickPodAPIError("get_rentable_machines", reason=str(e), cause=e)

        # Debug dump to help inspect TFLOPS/GPU data returned by QuickPod
        dump_path = os.getenv("RENTABLE_DUMP_PATH", "/tmp/rentable_dump.json")
        try:
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "count": len(offers) if isinstance(offers, list) else 0,
                # Store the first 200 offers to avoid oversized files
                "offers": offers[:200] if isinstance(offers, list) else offers,
            }
            with open(dump_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.debug(f"Wrote rentable dump to {dump_path} (count={payload.get('count')})")
        except Exception as dump_err:
            logger.debug(f"Failed to write rentable dump: {dump_err}")

        return offers
        
    def get_best_offer(
        self,
        preferred_machine_id: Optional[int] = None,
        min_tflops: float = 0,
        allowed_gpu_names: Optional[List[str]] = None,
        min_gpu_count: int = 1,
        blacklisted_machines: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the best available machine offer.
        
        Selection priority:
        1. Preferred machine (if specified and available)
        2. MACHINE_ID env variable
        3. WARM_MACHINE_ID env variable
        4. Fallback: cheapest offer by GPU tier
        
        Args:
            preferred_machine_id: Preferred machine ID to use
            min_tflops: Minimum TFLOPS required
            allowed_gpu_names: List of allowed GPU names (e.g., ["5090", "4090"])
            min_gpu_count: Minimum GPU count required
            blacklisted_machines: List of machine IDs to exclude from selection
            
        Returns:
            Tuple of (offer_id, machine_id) or (None, None) if no match
        """
        blacklist_set = set(str(m) for m in (blacklisted_machines or []))
        try:
            offers = self.get_rentable_machines()
        except Exception as e:
            logger.error(f"Failed to fetch offers: {e}")
            return None, None
        
        if not offers:
            logger.warning("No offers available")
            return None, None
        
        # Helper functions
        def extract_offer_id(o: Dict) -> Optional[int]:
            return o.get("offer_id") or o.get("id")
        
        def extract_machine_id(o: Dict) -> Optional[int]:
            for key in ("machines_id", "machine_id", "machineId"):
                if key in o and o[key] is not None:
                    try:
                        return int(o[key])
                    except (ValueError, TypeError):
                        pass
            return None
        
        def is_multi_gpu_offer(o: Dict) -> bool:
            name = str(o.get("offer_name", "")).upper()
            if name.startswith(("2X", "3X", "4X")):
                return True
            gpu_indexes = str(o.get("gpu_indexes", "") or "")
            tokens = [t for t in gpu_indexes.split("|") if t.strip()]
            return len(tokens) > 1
        
        def extract_price(o: Dict) -> float:
            try:
                return float(o.get("hourly_cost", 1e9))
            except (ValueError, TypeError):
                return 1e9
        
        def has_valid_tflops(o: Dict) -> bool:
            try:
                return float(o.get("tflops", 0)) >= min_tflops
            except (ValueError, TypeError):
                return False
        
        def matches_gpu_name(o: Dict) -> bool:
            if not allowed_gpu_names:
                return True
            offer_name = str(o.get("offer_name", "")).upper()
            for gpu in allowed_gpu_names:
                if gpu.upper() in offer_name:
                    return True
            return False
        
        def is_blacklisted(o: Dict) -> bool:
            mid = extract_machine_id(o)
            if mid and str(mid) in blacklist_set:
                return True
            return False
        
        def has_sufficient_memory(o: Dict) -> bool:
            """Require at least 30GB RAM to handle model loading + inference."""
            try:
                return float(o.get("memory", 0)) > 30
            except (ValueError, TypeError):
                return False
        
        def is_valid_offer(o: Dict) -> bool:
            return (
                not is_multi_gpu_offer(o) and
                has_valid_tflops(o) and
                matches_gpu_name(o) and
                has_sufficient_memory(o) and  # Require >30GB RAM
                not is_blacklisted(o)  # Filter out blacklisted machines
            )
        
        # 1. Try preferred machine
        if preferred_machine_id:
            filtered = [
                o for o in offers
                if extract_machine_id(o) == int(preferred_machine_id)
                and is_valid_offer(o)
            ]
            if filtered:
                oid = extract_offer_id(filtered[0])
                mid = extract_machine_id(filtered[0])
                if oid:
                    logger.info(f"Using preferred machine {mid}, offer_id={oid}")
                    return oid, mid
            logger.warning(f"Preferred machine {preferred_machine_id} not available")
        
        # 2. Try env machine fallback
        for env_key in ("MACHINE_ID", "WARM_MACHINE_ID"):
            env_val = os.getenv(env_key)
            if env_val:
                try:
                    machine_id = int(env_val)
                    filtered = [
                        o for o in offers
                        if extract_machine_id(o) == machine_id
                        and is_valid_offer(o)
                    ]
                    if filtered:
                        oid = extract_offer_id(filtered[0])
                        mid = extract_machine_id(filtered[0])
                        logger.info(f"Using {env_key}={mid}, offer_id={oid}")
                        return oid, mid
                except ValueError:
                    pass
        
        # 3. Fallback by GPU tier
        gpu_tiers = ["5090", "4090", "4080TI", "4080", "4070"]
        for tier in gpu_tiers:
            tier_offers = [
                o for o in offers
                if tier in str(o.get("offer_name", "")).upper()
                and is_valid_offer(o)
            ]
            tier_offers.sort(key=extract_price)
            if tier_offers:
                cheapest = tier_offers[0]
                oid = extract_offer_id(cheapest)
                mid = extract_machine_id(cheapest)
                logger.info(
                    f"Using fallback: GPU={tier}, offer_id={oid}, "
                    f"price={extract_price(cheapest):.3f}, "
                    f"tflops={cheapest.get('tflops')}, machine_id={mid}"
                )
                return oid, mid
        
        logger.error("No suitable offers found")
        return None, None
    
    def get_best_offer_with_details(
        self,
        preferred_machine_id: Optional[int] = None,
        min_tflops: float = 0,
        allowed_gpu_names: Optional[List[str]] = None,
        min_gpu_count: int = 1,
        blacklisted_machines: Optional[List[str]] = None,
    ) -> Optional[OfferInfo]:
        """
        Find the best available machine offer with full details including pricing.
        
        Same logic as get_best_offer but returns OfferInfo with hourly_cost.
        
        Returns:
            OfferInfo with full details, or None if no match
        """
        blacklist_set = set(str(m) for m in (blacklisted_machines or []))
        try:
            offers = self.get_rentable_machines()
        except Exception as e:
            logger.error(f"Failed to fetch offers: {e}")
            return None
        
        if not offers:
            logger.warning("No offers available")
            return None
        
        # Helper functions (same as get_best_offer)
        def extract_offer_id(o: Dict) -> Optional[int]:
            return o.get("offer_id") or o.get("id")
        
        def extract_machine_id(o: Dict) -> Optional[int]:
            for key in ("machines_id", "machine_id", "machineId"):
                if key in o and o[key] is not None:
                    try:
                        return int(o[key])
                    except (ValueError, TypeError):
                        pass
            return None
        
        def is_multi_gpu_offer(o: Dict) -> bool:
            name = str(o.get("offer_name", "")).upper()
            if name.startswith(("2X", "3X", "4X")):
                return True
            gpu_indexes = str(o.get("gpu_indexes", "") or "")
            tokens = [t for t in gpu_indexes.split("|") if t.strip()]
            return len(tokens) > 1
        
        def extract_price(o: Dict) -> float:
            try:
                return float(o.get("hourly_cost", 1e9))
            except (ValueError, TypeError):
                return 1e9
        
        def has_valid_tflops(o: Dict) -> bool:
            try:
                return float(o.get("tflops", 0)) >= min_tflops
            except (ValueError, TypeError):
                return False
        
        def matches_gpu_name(o: Dict) -> bool:
            if not allowed_gpu_names:
                return True
            offer_name = str(o.get("offer_name", "")).upper()
            for gpu in allowed_gpu_names:
                if gpu.upper() in offer_name:
                    return True
            return False
        
        def is_blacklisted(o: Dict) -> bool:
            mid = extract_machine_id(o)
            if mid and str(mid) in blacklist_set:
                return True
            return False
        
        def has_sufficient_memory(o: Dict) -> bool:
            try:
                return float(o.get("memory", 0)) > 30
            except (ValueError, TypeError):
                return False
        
        def is_valid_offer(o: Dict) -> bool:
            return (
                not is_multi_gpu_offer(o) and
                has_valid_tflops(o) and
                matches_gpu_name(o) and
                has_sufficient_memory(o) and
                not is_blacklisted(o)
            )
        
        def to_offer_info(o: Dict) -> OfferInfo:
            return OfferInfo(
                offer_id=extract_offer_id(o) or 0,
                machine_id=extract_machine_id(o) or 0,
                hourly_cost=extract_price(o),
                gpu_type=o.get("gpu_type", o.get("offer_name", "Unknown")),
                gpu_vram=int(o.get("gpu_vram", 0)),
                tflops=float(o.get("tflops", 0)),
                memory=float(o.get("memory", 0)),
            )
        
        # 1. Try preferred machine
        if preferred_machine_id:
            filtered = [
                o for o in offers
                if extract_machine_id(o) == int(preferred_machine_id)
                and is_valid_offer(o)
            ]
            if filtered:
                offer = filtered[0]
                info = to_offer_info(offer)
                logger.info(f"Using preferred machine {info.machine_id}, offer_id={info.offer_id}, hourly=${info.hourly_cost:.3f}")
                return info
        
        # 2. Try env machine fallback
        for env_key in ("MACHINE_ID", "WARM_MACHINE_ID"):
            env_val = os.getenv(env_key)
            if env_val:
                try:
                    machine_id = int(env_val)
                    filtered = [
                        o for o in offers
                        if extract_machine_id(o) == machine_id
                        and is_valid_offer(o)
                    ]
                    if filtered:
                        info = to_offer_info(filtered[0])
                        logger.info(f"Using {env_key}={info.machine_id}, offer_id={info.offer_id}, hourly=${info.hourly_cost:.3f}")
                        return info
                except ValueError:
                    pass
        
        # 3. Fallback by GPU tier
        gpu_tiers = ["5090", "4090", "4080TI", "4080", "4070"]
        for tier in gpu_tiers:
            tier_offers = [
                o for o in offers
                if tier in str(o.get("offer_name", "")).upper()
                and is_valid_offer(o)
            ]
            tier_offers.sort(key=extract_price)
            if tier_offers:
                info = to_offer_info(tier_offers[0])
                logger.info(
                    f"Using fallback: GPU={tier}, offer_id={info.offer_id}, "
                    f"hourly=${info.hourly_cost:.3f}, gpu={info.gpu_type}"
                )
                return info
        
        logger.error("No suitable offers found")
        return None
    
    def create_pod(
        self,
        offer_id: int,
        template_uuid: str,
        disk_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Create a new pod.
        
        Args:
            offer_id: Offer ID from get_best_offer
            template_uuid: Template UUID to use
            disk_size: Disk size in GB
            
        Returns:
            API response including pod_uuid
            
        Raises:
            PodCreationError: If creation fails
        """
        body = {
            "offers_id": offer_id,
            "disk_size": str(disk_size),
            "template_uuid": template_uuid,
        }
        
        logger.info(
            f"Creating pod: offer={offer_id}, template={template_uuid}, disk={disk_size}GB"
        )
        
        try:
            data = self._make_request(
                "POST",
                "update/createpod",
                json=body,
                timeout=POD_OPERATION_TIMEOUT,
            )
            
            if data is None:
                raise PodCreationError("unknown", reason="No data returned from API")
            
            if data.get("status") != "success":
                raise PodCreationError("unknown", reason=f"API returned: {data}")
            
            self.pod_uuid = (
                data.get("pod_uuid") or 
                data.get("uuid") or 
                data.get("id")
            )
            
            logger.info(f"Pod created successfully: {self.pod_uuid}")
            logger.debug(f"Full response: {json.dumps(data, indent=2)}")
            
            return data
            
        except (PodCreationError, QuickPodFatalError):
            raise
        except Exception as e:
            raise PodCreationError("unknown", reason=str(e), cause=e)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("QuickPod API Test - Pod Lifecycle (Create & Destroy)")
    print("=" * 80)
    print()
    # Set your credentials here
    QUICKPOD_EMAIL = "pavanalur3@gmail.com"
    QUICKPOD_PASSWORD = "C@9rw6fxATDm8sM"
    
    # Initialize manager with email and password
    print("[1] Initializing QuickPodManager...")
    manager = QuickPodManager(email=QUICKPOD_EMAIL, password=QUICKPOD_PASSWORD)
    print(f"    Email: {manager.email}")
    print(f"    Base URL: {manager.base}")
    print()
    
    try:
        # Login
        print("[2] Logging in...")
        print(f"    → POST {BASE_URL}/auth/login")
        token = manager.login()
        print(f"    ✓ Login successful, token acquired")
        print()
        
        # Get templates
        print("[3] Fetching templates...")
        print(f"    → GET {manager.base}/templates")
        templates = manager.get_templates()
        print(f"    ✓ Found {len(templates)} templates")
        print(f"    Templates: {json.dumps(templates, indent=2)}")
        if templates:
            template = templates[0]
            template_uuid = template.get("template_uuid")
            template_name = template.get("template_name", "unknown")
            print(f"    Using template: {template_name} ({template_uuid})")
        print()
        
        # Get best offer
        print("[4] Getting best offer...")
        print(f"    → GET {RENTABLE_URL}")
        offer_id, machine_id = manager.get_best_offer()
        if offer_id and machine_id:
            print(f"    ✓ Best offer: offer_id={offer_id}, machine_id={machine_id}")
        else:
            print("    ✗ No suitable offers found")
            sys.exit(1)
        print()
        
        # Create pod
        print("[5] Creating pod...")
        print(f"    → POST {BASE_URL}/createpod")
        print(f"    Body: {{'offers_id': {offer_id}, 'disk_size': '50', 'template_uuid': '{template_uuid}'}}")
        pod_data = manager.create_pod(offer_id, template_uuid, disk_size=50)
        pod_uuid = manager.pod_uuid
        print(f"    ✓ Pod created successfully!")
        print(f"    Pod UUID: {pod_uuid}")
        print(f"    Response: {json.dumps(pod_data, indent=2)}")
        print()
        
        # Wait a moment
        print("[6] Waiting 45 seconds before destroying...")
        time.sleep(45)
        print()
        
        # Destroy pod
        print("[7] Destroying pod...")
        print(f"    → GET {BASE_URL}/destroypod?pod_uuid={pod_uuid}")
        print(f"    Pod UUID: {pod_uuid}")
        
        # Create a new manager instance for destroy (as per the actual usage pattern)
        destroy_manager = QuickPodManager(email=QUICKPOD_EMAIL, password=QUICKPOD_PASSWORD, pod_uuid=pod_uuid)
        destroy_manager.login()  # Ensure authenticated
        
        destroy_result = destroy_manager.destroy_pod(verify=True)
        print(f"    ✓ Destroy request completed")
        print(f"    Response: {json.dumps(destroy_result, indent=2)}")
        print()
        
        # Verification (if verify was True, this was already done, but show it anyway)
        print("[8] Verifying pod destruction...")
        print(f"    → GET {MYPODS_URL}")
        time.sleep(2)  # Brief wait for QuickPod to process
        is_destroyed = destroy_manager.is_pod_destroyed(pod_uuid)
        if is_destroyed:
            print(f"    ✓ Pod {pod_uuid} confirmed destroyed")
        else:
            print(f"    ✗ Pod {pod_uuid} still exists (or verification failed)")
        print()
        
        print("=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

