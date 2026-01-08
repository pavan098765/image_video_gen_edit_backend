# middleware.py
"""
Security & Rate Limiting Middleware

Provides:
- Aggressive security middleware (method blocking, UA detection, scanner detection)
- Path whitelist middleware
- CORS configuration
- Request logging
- Request timeout handling
- Worker IP whitelist exemptions
"""

import os
import re
import time
import asyncio
import logging
from typing import Set, Dict, List, Optional, Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from security import ip_ban_manager, security_config

logger = logging.getLogger("middleware")
security_logger = logging.getLogger("security")


# =============================================================================
# Configuration
# =============================================================================

# Blocked HTTP methods
BLOCKED_METHODS: Set[str] = {"PROPFIND", "CONNECT", "TRACE", "TRACK", "PRI"}

# Blocked user agents (lowercase)
BLOCKED_USER_AGENTS: List[str] = [
    "curl",
    "wget",
    "python-requests",
    "sqlmap",
    "nmap",
    "nikto",
    "masscan",
    "scrapy",
    "libwww-perl",
    "zgrab",
    "httpclient",
    "scanbot",
    "nessus",
    "dirbuster",
    "acunetix",
    "wpscan",
    "whatweb",
    "httprobe",
    "bot",
    "go-http-client",
]

# Suspicious path patterns
SUSPICIOUS_PATHS: List[str] = [
    r"\.\./",
    r"etc/passwd",
    r"wp-admin",
    r"phpmyadmin",
    r"\.env",
    r"\.git",
    r"/owa/",
    r"/solr/",
    r"/admin",
    r"/console",
    r"/status\.php",
    r"/login",
    r"/jasperserver",
    r"/owncloud",
    r"/zabbix",
    r"/WebInterface",
    r"/sitecore",
    r"/cgi-bin",
    r"/index\.jsp",
    r"/OA_HTML",
    r"/Telerik",
    r"cf_scripts",
]

# Suspicious headers
SUSPICIOUS_HEADERS: List[str] = ["X-Original-URL", "X-Rewrite-URL", "Forwarded"]

# Injection patterns in body
INJECTION_PATTERNS: List[bytes] = [
    b"<script",
    b"<?php",
    b"union select",
    b"drop table",
    b"--",
    b"alert(",
]

# Allowed path prefixes
ALLOWED_PATHS: List[str] = [
    "/health",
    "/submit",
    "/job_complete",
    "/job_status",
    "/register_worker",
    "/heartbeat",
    "/download",
    "/auth",  # Firebase auth endpoints
    "/ads",   # Ad network SSV callbacks (AdMob, etc.)
]

# Worker paths (exempt from some checks)
WORKER_PATHS: List[str] = ["/register_worker", "/heartbeat"]

# Trusted API paths (require auth - Bearer token or fingerprint)
TRUSTED_API_PATHS: List[str] = [
    "/submit",
    "/job_complete",
    "/job_status",
    "/auth",  # Firebase auth endpoints require Bearer token
]

# Public paths (no auth needed)
PUBLIC_PATHS: List[str] = [
    "/download", 
    "/health",
    "/ads/admob/ssv",  # AdMob SSV callback (called by Google, no auth)
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies"""
    # Check X-Forwarded-For header
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        # Take the first (leftmost) IP as the original client
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct connection
    return request.client.host if request.client else "unknown"


def is_path_allowed(path: str, allowed_prefixes: List[str]) -> bool:
    """Check if path matches any allowed prefix"""
    return any(path.startswith(prefix) for prefix in allowed_prefixes)


def is_suspicious_user_agent(user_agent: str) -> bool:
    """Check if user agent is in blocked list"""
    ua_lower = user_agent.lower()
    return any(blocked in ua_lower for blocked in BLOCKED_USER_AGENTS)


def is_suspicious_path(path: str) -> bool:
    """Check if path matches suspicious patterns"""
    for pattern in SUSPICIOUS_PATHS:
        if re.search(pattern, path, re.IGNORECASE):
            return True
    return False


def has_suspicious_headers(headers: Dict) -> Optional[str]:
    """Check for suspicious headers, return first found or None"""
    for header in SUSPICIOUS_HEADERS:
        if header in headers:
            return header
    return None


async def check_body_for_injection(body: bytes) -> Optional[bytes]:
    """Check request body for injection patterns, return pattern if found"""
    body_lower = body.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in body_lower:
            return pattern
    return None


# =============================================================================
# Security Middleware
# =============================================================================

class AggressiveSecurityMiddleware(BaseHTTPMiddleware):
    """
    Aggressive security middleware that blocks:
    - Dangerous HTTP methods
    - Known scanner user agents
    - Suspicious paths
    - Injection attempts
    - Rate limit violations
    """
    
    def __init__(
        self,
        app: ASGIApp,
        block_empty_ua: bool = True,
        max_body_size: int = 10_000_000,
    ):
        super().__init__(app)
        self.block_empty_ua = block_empty_ua
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip = get_client_ip(request)
        path = request.url.path.lower()
        ua = request.headers.get("user-agent", "")
        auth_header = request.headers.get("authorization", "")
        
        # === Check if IP is trusted worker ===
        if ip_ban_manager.is_trusted_worker_ip(ip):
            return await call_next(request)
        
        # === Check if IP is banned ===
        if ip_ban_manager.is_banned(ip):
            security_logger.warning(f"Blocked banned IP: {ip} path={path}")
            return Response("Access Denied", status_code=403)
        
        # === Rate limiting ===
        if not path.startswith("/job_status") and not ip_ban_manager.check_rate_limit(ip):
            await ip_ban_manager.ban_ip(
                ip,
                "Rate limit exceeded",
                duration=security_config.ban_duration,
            )
            return Response("Too Many Requests", status_code=429)
        
        # === Block dangerous HTTP methods ===
        if request.method.upper() in BLOCKED_METHODS:
            await ip_ban_manager.ban_ip(
                ip,
                f"Blocked method {request.method}",
                permanent=True,
            )
            return Response("Method Not Allowed", status_code=405)
        
        # === Worker paths - skip aggressive checks ===
        if is_path_allowed(path, WORKER_PATHS):
            try:
                return await call_next(request)
            except Exception as e:
                security_logger.exception(f"Worker path error from {ip}: {e}")
                return Response("Internal Server Error", status_code=500)
        
        # === Trusted API paths - require auth header ===
        if is_path_allowed(path, TRUSTED_API_PATHS):
            if not (auth_header.startswith("Bearer ") or request.headers.get("x-fingerprint")):
                await ip_ban_manager.increase_suspicion(ip, 3, "Trusted endpoint without auth")
                return Response("Forbidden", status_code=403)
            try:
                return await call_next(request)
            except Exception as e:
                security_logger.exception(f"API path error from {ip}: {e}")
                return Response("Internal Server Error", status_code=500)
        
        # === Public paths - minimal checks ===
        if is_path_allowed(path, PUBLIC_PATHS):
            try:
                return await call_next(request)
            except Exception as e:
                security_logger.exception(f"Public path error from {ip}: {e}")
                return Response("Internal Server Error", status_code=500)
        
        # === Aggressive checks for untrusted paths ===
        
        # Suspicious path detection
        if is_suspicious_path(path):
            await ip_ban_manager.increase_suspicion(ip, 2, f"Scanner path: {path}")
            await ip_ban_manager.ban_ip(ip, f"Scanner path: {path}", permanent=True)
            return Response("Forbidden", status_code=403)
        
        # Empty user agent
        if self.block_empty_ua and not ua.strip():
            await ip_ban_manager.ban_ip(ip, "Empty user-agent", permanent=True)
            return Response("Forbidden", status_code=403)
        
        # Blocked user agent
        if is_suspicious_user_agent(ua):
            await ip_ban_manager.ban_ip(ip, f"Bad user-agent: {ua}", permanent=True)
            return Response("Forbidden", status_code=403)
        
        # Suspicious headers
        suspicious_header = has_suspicious_headers(dict(request.headers))
        if suspicious_header:
            await ip_ban_manager.ban_ip(ip, f"Suspicious header: {suspicious_header}", permanent=True)
            return Response("Forbidden", status_code=403)
        
        # Body inspection for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                
                # Size check
                if len(body) > self.max_body_size:
                    await ip_ban_manager.ban_ip(ip, "Oversized body", permanent=False)
                    return Response("Payload Too Large", status_code=413)
                
                # Injection check
                injection_pattern = await check_body_for_injection(body)
                if injection_pattern:
                    await ip_ban_manager.ban_ip(
                        ip,
                        f"Injection pattern: {injection_pattern}",
                        permanent=True,
                    )
                    return Response("Forbidden", status_code=403)
            except Exception:
                pass
        
        # Path anomalies
        if len(path) > 200 or path.count("/") > 20:
            await ip_ban_manager.increase_suspicion(ip, 1, "Path anomaly")
        
        if re.search(r"[<>{}\"';]", path):
            await ip_ban_manager.increase_suspicion(ip, 1, "Path character anomaly")
        
        # Normal request flow
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            security_logger.exception(f"Unhandled exception from {ip} path={path}: {e}")
            return Response("Internal Server Error", status_code=500)


# =============================================================================
# Path Whitelist Middleware
# =============================================================================

class PathWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Rejects requests to paths not in the whitelist.
    
    Prevents scanners from probing unlisted endpoints.
    """
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        
        # Build allowed list
        allowed = list(ALLOWED_PATHS)
        
        # Allow docs/openapi in debug mode
        if self.debug:
            allowed.extend(["/docs", "/openapi.json", "/redoc"])
        
        # Check if path is allowed
        if not is_path_allowed(path, allowed):
            security_logger.debug(f"Rejected unknown path: {path}")
            return Response("Forbidden", status_code=403)
        
        return await call_next(request)


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs request details for debugging and audit.
    """
    
    def __init__(self, app: ASGIApp, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        ip = get_client_ip(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - "
            f"{duration:.3f}s - "
            f"IP: {ip}"
        )
        
        return response


# =============================================================================
# Request Timeout Middleware
# =============================================================================

class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforces request timeout.
    """
    
    def __init__(self, app: ASGIApp, timeout: int = 30):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {request.method} {request.url.path}")
            return Response("Request Timeout", status_code=408)


# =============================================================================
# CORS Configuration
# =============================================================================

def get_cors_middleware_config() -> Dict:
    """Get CORS middleware configuration from environment"""
    origins = os.getenv("CORS_ORIGINS", "*")
    
    return {
        "allow_origins": origins.split(",") if origins != "*" else ["*"],
        "allow_credentials": os.getenv("CORS_CREDENTIALS", "true").lower() == "true",
        "allow_methods": os.getenv("CORS_METHODS", "*").split(","),
        "allow_headers": os.getenv("CORS_HEADERS", "*").split(","),
        "expose_headers": os.getenv("CORS_EXPOSE_HEADERS", "").split(",") or [],
        "max_age": int(os.getenv("CORS_MAX_AGE", "600")),
    }


# =============================================================================
# Middleware Setup Helper
# =============================================================================

def setup_middleware(app, debug: bool = False):
    """
    Set up all middleware for the application.
    
    Args:
        app: FastAPI application
        debug: Whether debug mode is enabled
    """
    # CORS (must be first)
    cors_config = get_cors_middleware_config()
    app.add_middleware(CORSMiddleware, **cors_config)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware, log_body=debug)
    
    # Request timeout
    timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
    app.add_middleware(RequestTimeoutMiddleware, timeout=timeout)
    
    # Path whitelist
    app.add_middleware(PathWhitelistMiddleware, debug=debug)
    
    # Aggressive security (last, so it runs first)
    max_body = int(os.getenv("MAX_BODY_SIZE", "10000000"))
    app.add_middleware(
        AggressiveSecurityMiddleware,
        block_empty_ua=os.getenv("BLOCK_EMPTY_UA", "true").lower() == "true",
        max_body_size=max_body,
    )
    
    logger.info("Middleware configured")

