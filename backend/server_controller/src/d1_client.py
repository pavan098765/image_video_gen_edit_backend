# d1_client.py
"""
Cloudflare D1 Database Client

Provides HTTP-based access to Cloudflare D1 database.
Used when deployed on VM to use cloud database instead of local SQLite.

D1 API: https://developers.cloudflare.com/api/operations/cloudflare-d1-query-database

IMPORTANT: D1 uses synchronous HTTP calls via requests library.
To avoid blocking the asyncio event loop, wrap D1 operations in:
    await asyncio.to_thread(sync_d1_function)
"""

import os
import logging
import json
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager

import requests

# Thread pool for D1 operations to avoid blocking asyncio event loop
# CRITICAL: D1 HTTP calls can take 10-30 seconds and would block everything
_d1_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="d1_")

logger = logging.getLogger("d1_client")


class D1Error(Exception):
    """Base exception for D1 errors"""
    pass


class D1ConnectionError(D1Error):
    """Connection/network error"""
    pass


class D1QueryError(D1Error):
    """Query execution error"""
    pass


class D1Client:
    """
    Cloudflare D1 HTTP Client.
    
    Provides SQLAlchemy-like interface for D1 database operations.
    
    Usage:
        client = D1Client(
            account_id="your-account-id",
            database_id="your-database-id",
            api_token="your-api-token"
        )
        
        # Execute query
        results = client.execute("SELECT * FROM users WHERE id = ?", ("user-123",))
        
        # Execute many
        client.execute_many("INSERT INTO users (id, name) VALUES (?, ?)", [
            ("user-1", "Alice"),
            ("user-2", "Bob"),
        ])
    """
    
    BASE_URL = "https://api.cloudflare.com/client/v4"
    DEFAULT_TIMEOUT = 10  # Reduced from 30s to prevent hanging
    
    def __init__(
        self,
        account_id: Optional[str] = None,
        database_id: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        self.account_id = account_id or os.getenv("CF_ACCOUNT_ID")
        self.database_id = database_id or os.getenv("CF_D1_DATABASE_ID")
        self.api_token = api_token or os.getenv("CF_API_TOKEN")
        
        if not all([self.account_id, self.database_id, self.api_token]):
            raise D1Error(
                "D1 configuration incomplete. Required: CF_ACCOUNT_ID, CF_D1_DATABASE_ID, CF_API_TOKEN"
            )
        
        self._endpoint = (
            f"{self.BASE_URL}/accounts/{self.account_id}"
            f"/d1/database/{self.database_id}/query"
        )
        
        logger.info(f"D1Client initialized for database {self.database_id}")
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
    
    def _parse_datetime(self, value: Any) -> Any:
        """Parse datetime strings from D1 results"""
        if isinstance(value, str):
            # Try common datetime formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return value
    
    def _process_row(self, row: Dict[str, Any], parse_datetime: bool = True) -> Dict[str, Any]:
        """Process a result row, optionally parsing datetime fields"""
        if not parse_datetime:
            return row
        
        # Common datetime column names
        datetime_columns = {
            "created_at", "updated_at", "last_seen", "last_heartbeat",
            "last_active", "completed_at", "timeout_at", "started_at",
            "queued_at", "dispatched_at", "expires_at", "renewed_at",
            "started_at", "last_job_finished_at", "timestamp",
        }
        
        processed = {}
        for key, value in row.items():
            if key in datetime_columns and value is not None:
                processed[key] = self._parse_datetime(value)
            else:
                processed[key] = value
        return processed
    
    def _clean_params(self, params: Union[Tuple, List]) -> List:
        """
        Convert params to JSON-serializable values.
        
        Handles SQLAlchemy special types like Null, True_, False_, etc.
        """
        cleaned = []
        for p in params:
            # Get the class name for type checking
            class_name = p.__class__.__name__ if p is not None else "NoneType"
            
            # Handle None
            if p is None:
                cleaned.append(None)
            # Handle SQLAlchemy Null type
            elif class_name == 'Null':
                cleaned.append(None)
            # Handle SQLAlchemy True_/False_ (boolean expressions)
            elif class_name in ('True_', 'true'):
                cleaned.append(1)
            elif class_name in ('False_', 'false'):
                cleaned.append(0)
            # Handle Python booleans (must check before general types)
            elif isinstance(p, bool):
                cleaned.append(1 if p else 0)
            # Handle datetime
            elif isinstance(p, datetime):
                cleaned.append(p.strftime("%Y-%m-%d %H:%M:%S"))
            # Handle SQLAlchemy BindParameter (extract the value)
            elif class_name == 'BindParameter' and hasattr(p, 'value'):
                # Recursively clean the value
                cleaned.extend(self._clean_params([p.value]))
            # Handle other types that have a 'value' attribute (SQLAlchemy literals)
            elif hasattr(p, 'value') and not callable(getattr(p, 'value')):
                val = p.value
                if isinstance(val, bool):
                    cleaned.append(1 if val else 0)
                elif val is None:
                    cleaned.append(None)
                else:
                    cleaned.append(val)
            else:
                cleaned.append(p)
        return cleaned
    
    def execute(
        self,
        sql: str,
        params: Optional[Union[Tuple, List]] = None,
        parse_datetime: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query string with ? placeholders
            params: Query parameters (tuple or list)
            parse_datetime: Whether to parse datetime strings
            
        Returns:
            List of result dictionaries
            
        Raises:
            D1QueryError: If query fails
            D1ConnectionError: If network error
        """
        payload = {"sql": sql}
        if params:
            payload["params"] = self._clean_params(params)
        
        try:
            response = requests.post(
                self._endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                errors = data.get("errors", [])
                error_msg = errors[0].get("message") if errors else "Unknown error"
                raise D1QueryError(f"D1 query failed: {error_msg}")
            
            # D1 returns results in a specific format
            result = data.get("result", [])
            if not result:
                return []
            
            # Get the first result set (D1 supports batch queries)
            first_result = result[0] if isinstance(result, list) else result
            rows = first_result.get("results", [])
            
            # Process rows
            return [self._process_row(row, parse_datetime) for row in rows]
            
        except requests.exceptions.Timeout:
            raise D1ConnectionError("D1 request timed out")
        except requests.exceptions.ConnectionError as e:
            raise D1ConnectionError(f"D1 connection failed: {e}")
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    errors = error_data.get("errors", [])
                    error_msg = errors[0].get("message") if errors else str(e)
                except Exception:
                    error_msg = str(e)
                raise D1QueryError(f"D1 HTTP error: {error_msg}")
            raise D1QueryError(f"D1 HTTP error: {e}")
    
    def execute_write(
        self,
        sql: str,
        params: Optional[Union[Tuple, List]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a write query (INSERT, UPDATE, DELETE) and return metadata.
        
        Returns:
            Dict with 'changes' (rows affected) and 'last_row_id' (for INSERT)
        """
        payload = {"sql": sql}
        if params:
            payload["params"] = self._clean_params(params)
        
        try:
            response = requests.post(
                self._endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                errors = data.get("errors", [])
                error_msg = errors[0].get("message") if errors else "Unknown error"
                raise D1QueryError(f"D1 write failed: {error_msg}")
            
            result = data.get("result", [])
            if not result:
                return {"changes": 0, "last_row_id": None}
            
            first_result = result[0] if isinstance(result, list) else result
            meta = first_result.get("meta", {})
            
            return {
                "changes": meta.get("changes", 0),
                "last_row_id": meta.get("last_row_id"),
                "rows_read": meta.get("rows_read", 0),
                "rows_written": meta.get("rows_written", 0),
            }
            
        except requests.exceptions.Timeout:
            raise D1ConnectionError("D1 request timed out")
        except requests.exceptions.ConnectionError as e:
            raise D1ConnectionError(f"D1 connection failed: {e}")
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    errors = error_data.get("errors", [])
                    error_msg = errors[0].get("message") if errors else str(e)
                except Exception:
                    error_msg = str(e)
                raise D1QueryError(f"D1 HTTP error: {error_msg}")
            raise D1QueryError(f"D1 HTTP error: {e}")
    
    def execute_batch(
        self,
        statements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple SQL statements in a batch.
        
        Args:
            statements: List of {"sql": "...", "params": [...]} dicts
            
        Returns:
            List of result sets
        """
        # D1 batch API endpoint
        batch_endpoint = (
            f"{self.BASE_URL}/accounts/{self.account_id}"
            f"/d1/database/{self.database_id}/raw"
        )
        
        try:
            response = requests.post(
                batch_endpoint,
                headers=self._headers(),
                json={"sql": statements},
                timeout=self.DEFAULT_TIMEOUT * 2,  # Longer timeout for batch
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                errors = data.get("errors", [])
                error_msg = errors[0].get("message") if errors else "Unknown error"
                raise D1QueryError(f"D1 batch failed: {error_msg}")
            
            return data.get("result", [])
            
        except requests.exceptions.Timeout:
            raise D1ConnectionError("D1 batch request timed out")
        except requests.exceptions.ConnectionError as e:
            raise D1ConnectionError(f"D1 connection failed: {e}")


# =============================================================================
# D1 Session - SQLAlchemy-like interface
# =============================================================================

class D1Session:
    """
    SQLAlchemy-like session interface for D1.
    
    Provides familiar .query(), .add(), .commit() interface while using D1 backend.
    
    Note: This is a simplified implementation. Complex ORM features like
    lazy loading, relationship traversal, etc. are not fully supported.
    """
    
    def __init__(self, client: D1Client):
        self.client = client
        self._pending_writes: List[Tuple[str, List]] = []
        self._new_objects: List[Any] = []
        self._dirty_objects: List[Any] = []
        self._deleted_objects: List[Any] = []
    
    def __enter__(self) -> 'D1Session':
        """Context manager entry - return self"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - close session"""
        if exc_type is not None:
            # Exception occurred, rollback
            self.rollback()
        self.close()
        return False  # Don't suppress exceptions
    
    def execute(self, sql: str, params: Optional[Union[Tuple, List]] = None) -> List[Dict]:
        """Execute raw SQL and return results"""
        return self.client.execute(sql, params)
    
    def execute_write(self, sql: str, params: Optional[Union[Tuple, List]] = None) -> Dict:
        """Execute write SQL and return metadata"""
        return self.client.execute_write(sql, params)
    
    def add(self, obj: Any):
        """Add object to session (will be inserted on commit)"""
        self._new_objects.append(obj)
    
    def delete(self, obj: Any):
        """Mark object for deletion"""
        self._deleted_objects.append(obj)
    
    def commit(self):
        """Commit all pending changes"""
        # Note: For true transactional support, you'd need to use D1's batch API
        # This is a simplified implementation
        
        for obj in self._new_objects:
            self._insert_object(obj)
        
        for obj in self._deleted_objects:
            self._delete_object(obj)
        
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._deleted_objects.clear()
    
    def rollback(self):
        """Rollback pending changes (just clears pending lists)"""
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._deleted_objects.clear()
    
    def refresh(self, obj: Any):
        """
        Refresh object from database (SQLAlchemy compatibility).
        
        For D1, we re-query the object by its primary key and update attributes.
        Handles JSON deserialization for MutableDict/MutableList columns.
        """
        try:
            table = self._get_table_name(obj)
            pk_col, pk_val = self._get_primary_key(obj)
            
            sql = f"SELECT * FROM {table} WHERE {pk_col} = ?"
            rows = self.client.execute(sql, [pk_val])
            
            if rows:
                row = rows[0]
                # Update object attributes from DB
                for key, value in row.items():
                    if hasattr(obj, key):
                        # Handle JSON columns - D1 returns them as strings
                        # but SQLAlchemy MutableDict/MutableList expect dict/list
                        if isinstance(value, str) and value:
                            # Check if the column is a JSON type
                            col_type = None
                            if hasattr(obj.__class__, "__table__"):
                                col = obj.__class__.__table__.columns.get(key)
                                if col is not None:
                                    col_type = str(col.type)
                            
                            # Deserialize if it's a JSON column or looks like JSON
                            if col_type and "JSON" in col_type.upper():
                                try:
                                    value = json.loads(value)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            elif value.startswith(("{", "[")):
                                try:
                                    value = json.loads(value)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                        
                        try:
                            setattr(obj, key, value)
                        except (ValueError, TypeError) as e:
                            # If setattr fails, log but continue
                            logger.debug(f"D1Session.refresh: couldn't set {key}: {e}")
        except Exception as e:
            logger.warning(f"D1Session.refresh failed: {e}")
            # Silently fail - the object already has its values from creation
    
    def close(self):
        """Close session (no-op for HTTP-based D1)"""
        self.rollback()
    
    def _get_table_name(self, obj: Any) -> str:
        """Get table name from object"""
        if hasattr(obj, "__tablename__"):
            return obj.__tablename__
        return obj.__class__.__name__.lower() + "s"
    
    def _get_primary_key(self, obj: Any) -> Tuple[str, Any]:
        """Get primary key column and value from object"""
        # Try common primary key names
        for pk_name in ["id", "job_id", "worker_id"]:
            if hasattr(obj, pk_name):
                return pk_name, getattr(obj, pk_name)
        raise ValueError(f"Cannot determine primary key for {obj}")
    
    def _object_to_columns(self, obj: Any) -> Dict[str, Any]:
        """Convert object to column dict, excluding SQLAlchemy relationships"""
        columns = {}
        
        # Get actual column names from SQLAlchemy model if available
        column_names = set()
        if hasattr(obj.__class__, "__table__"):
            column_names = {c.name for c in obj.__class__.__table__.columns}
        
        for key in dir(obj):
            if key.startswith("_"):
                continue
            if key in ("metadata", "registry", "to_dict", "pod_uuid"):
                continue
            
            # If we have column info, only include actual columns
            if column_names and key not in column_names:
                continue
            
            value = getattr(obj, key)
            if callable(value):
                continue
            
            # Skip SQLAlchemy relationship collections (InstrumentedList, etc.)
            value_type = type(value).__name__
            if value_type in ("InstrumentedList", "AppenderQuery", "Query"):
                continue
            
            # Skip None values for non-nullable fields might cause issues,
            # but include them for optional fields
            
            # Handle datetime
            if isinstance(value, datetime):
                columns[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, (dict, list)):
                columns[key] = json.dumps(value)
            elif isinstance(value, bool):
                columns[key] = 1 if value else 0
            else:
                columns[key] = value
        return columns
    
    def _insert_object(self, obj: Any):
        """Insert object into database"""
        table = self._get_table_name(obj)
        columns = self._object_to_columns(obj)
        
        col_names = list(columns.keys())
        placeholders = ", ".join(["?" for _ in col_names])
        col_str = ", ".join(col_names)
        
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
        params = list(columns.values())
        
        self.client.execute_write(sql, params)
    
    def _delete_object(self, obj: Any):
        """Delete object from database"""
        table = self._get_table_name(obj)
        pk_col, pk_val = self._get_primary_key(obj)
        
        sql = f"DELETE FROM {table} WHERE {pk_col} = ?"
        self.client.execute_write(sql, [pk_val])
    
    def query(self, model_class: type) -> 'D1Query':
        """Start a query for a model class"""
        return D1Query(self, model_class)


class D1Query:
    """
    Query builder for D1 (simplified SQLAlchemy-like interface).
    """
    
    def __init__(self, session: D1Session, model_class: type):
        self.session = session
        self.model_class = model_class
        self.table_name = getattr(model_class, "__tablename__", model_class.__name__.lower() + "s")
        
        self._filters: List[Tuple[str, Any]] = []
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
    
    def filter(self, *args, **kwargs) -> 'D1Query':
        """
        Add filter conditions.
        
        Supports:
        - Keyword args: filter(name="John") -> WHERE name = ?
        - SQLAlchemy-style: filter(User.name == "John") -> WHERE name = ?
        """
        # Handle SQLAlchemy-style binary expressions (e.g., User.name == "value")
        for arg in args:
            # Check if it's a SQLAlchemy BinaryExpression
            if hasattr(arg, 'left') and hasattr(arg, 'right') and hasattr(arg, 'operator'):
                # Extract column name from left side
                col_name = None
                if hasattr(arg.left, 'key'):
                    col_name = arg.left.key
                elif hasattr(arg.left, 'name'):
                    col_name = arg.left.name
                
                # Extract value from right side
                value = None
                if hasattr(arg.right, 'value'):
                    value = arg.right.value
                else:
                    value = arg.right
                
                # Determine operator
                op_name = arg.operator.__name__ if hasattr(arg.operator, '__name__') else str(arg.operator)
                
                if col_name is not None:
                    if 'eq' in op_name or op_name == '==':
                        self._filters.append((f"{col_name} = ?", value))
                    elif 'ne' in op_name or op_name == '!=':
                        self._filters.append((f"{col_name} != ?", value))
                    elif 'lt' in op_name and 'le' not in op_name:
                        self._filters.append((f"{col_name} < ?", value))
                    elif 'le' in op_name:
                        self._filters.append((f"{col_name} <= ?", value))
                    elif 'gt' in op_name and 'ge' not in op_name:
                        self._filters.append((f"{col_name} > ?", value))
                    elif 'ge' in op_name:
                        self._filters.append((f"{col_name} >= ?", value))
                    else:
                        # Default to equality
                        self._filters.append((f"{col_name} = ?", value))
        
        # Simple equality filters via kwargs
        for col, val in kwargs.items():
            self._filters.append((f"{col} = ?", val))
        return self
    
    def filter_by(self, **kwargs) -> 'D1Query':
        """Add filter conditions (alias for filter)"""
        return self.filter(**kwargs)
    
    def order_by(self, column: str, desc: bool = False) -> 'D1Query':
        """Add ORDER BY clause"""
        direction = "DESC" if desc else "ASC"
        self._order_by = f"{column} {direction}"
        return self
    
    def limit(self, n: int) -> 'D1Query':
        """Add LIMIT clause"""
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'D1Query':
        """Add OFFSET clause"""
        self._offset = n
        return self
    
    def _build_query(self, select: str = "*") -> Tuple[str, List]:
        """Build SQL query string and params"""
        sql = f"SELECT {select} FROM {self.table_name}"
        params = []
        
        if self._filters:
            conditions = []
            for condition, value in self._filters:
                conditions.append(condition)
                params.append(value)
            sql += " WHERE " + " AND ".join(conditions)
        
        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
        
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        
        if self._offset is not None:
            sql += f" OFFSET {self._offset}"
        
        return sql, params
    
    def _dict_to_model(self, row: Dict[str, Any]) -> Any:
        """
        Convert result dict to model instance.
        
        Handles JSON field deserialization for SQLAlchemy MutableDict/MutableList columns.
        """
        obj = self.model_class()
        
        # Get JSON column names from model if available
        json_columns = set()
        if hasattr(self.model_class, "__table__"):
            for col in self.model_class.__table__.columns:
                # Check if column type is JSON
                col_type_name = col.type.__class__.__name__
                if col_type_name in ("JSON", "MutableDict", "MutableList"):
                    json_columns.add(col.name)
        
        # Common JSON column names (fallback)
        common_json_cols = {"params", "info", "resource_metrics", "resource_limits", 
                           "error_log", "details", "meta", "job_id_counts"}
        json_columns.update(common_json_cols)
        
        for key, value in row.items():
            if hasattr(obj, key):
                # Deserialize JSON strings for JSON columns
                if key in json_columns and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not valid JSON, keep as string
                        pass
                
                # Handle None for nullable columns
                try:
                    setattr(obj, key, value)
                except (ValueError, TypeError) as e:
                    # Log warning but don't fail - use None for problematic values
                    logger.warning(f"Could not set {key}={type(value).__name__} on {self.model_class.__name__}: {e}")
                    try:
                        setattr(obj, key, None)
                    except Exception:
                        pass
        return obj
    
    def all(self) -> List[Any]:
        """Execute query and return all results as model instances"""
        sql, params = self._build_query()
        rows = self.session.client.execute(sql, params)
        return [self._dict_to_model(row) for row in rows]
    
    def first(self) -> Optional[Any]:
        """Execute query and return first result"""
        self._limit = 1
        results = self.all()
        return results[0] if results else None
    
    def one(self) -> Any:
        """Execute query and return exactly one result (raises if not found)"""
        result = self.first()
        if result is None:
            raise D1QueryError(f"No result found for {self.model_class.__name__}")
        return result
    
    def one_or_none(self) -> Optional[Any]:
        """Execute query and return one result or None"""
        return self.first()
    
    def count(self) -> int:
        """Execute COUNT query"""
        sql, params = self._build_query(select="COUNT(*) as count")
        rows = self.session.client.execute(sql, params)
        return rows[0]["count"] if rows else 0
    
    def delete(self) -> int:
        """Execute DELETE query and return rows affected"""
        sql = f"DELETE FROM {self.table_name}"
        params = []
        
        if self._filters:
            conditions = []
            for condition, value in self._filters:
                conditions.append(condition)
                params.append(value)
            sql += " WHERE " + " AND ".join(conditions)
        
        result = self.session.client.execute_write(sql, params)
        return result.get("changes", 0)


# =============================================================================
# D1 Session Factory
# =============================================================================

class D1SessionFactory:
    """
    Factory for creating D1 sessions.
    
    Mimics SQLAlchemy's sessionmaker.
    """
    
    def __init__(self, client: D1Client):
        self.client = client
    
    def __call__(self) -> D1Session:
        """Create new session"""
        return D1Session(self.client)


# =============================================================================
# Global D1 Instance
# =============================================================================

_d1_client: Optional[D1Client] = None
_d1_session_factory: Optional[D1SessionFactory] = None


def init_d1(
    account_id: Optional[str] = None,
    database_id: Optional[str] = None,
    api_token: Optional[str] = None,
) -> D1Client:
    """Initialize global D1 client"""
    global _d1_client, _d1_session_factory
    
    _d1_client = D1Client(account_id, database_id, api_token)
    _d1_session_factory = D1SessionFactory(_d1_client)
    
    logger.info("D1 client initialized successfully")
    return _d1_client


def get_d1_client() -> D1Client:
    """Get global D1 client"""
    if _d1_client is None:
        raise D1Error("D1 client not initialized. Call init_d1() first.")
    return _d1_client


def get_d1_session() -> D1Session:
    """Get new D1 session"""
    if _d1_session_factory is None:
        raise D1Error("D1 session factory not initialized. Call init_d1() first.")
    return _d1_session_factory()


@contextmanager
def d1_session_scope():
    """Context manager for D1 session with auto-cleanup"""
    session = get_d1_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_d1_enabled() -> bool:
    """Check if D1 is enabled (all required env vars set)"""
    return all([
        os.getenv("CF_ACCOUNT_ID"),
        os.getenv("CF_D1_DATABASE_ID"),
        os.getenv("CF_API_TOKEN"),
    ])


# =============================================================================
# ASYNC UTILITIES FOR NON-BLOCKING D1 OPERATIONS
# =============================================================================
# Use these in async contexts to prevent D1 from blocking the event loop


async def run_d1_sync(func, *args, **kwargs):
    """
    Run a synchronous D1 operation in a thread pool to avoid blocking asyncio.
    
    CRITICAL: D1 HTTP calls can take 10-30 seconds and would block the entire
    event loop, causing heartbeat timeouts and other failures.
    
    Usage:
        # Instead of:
        with SessionLocal() as db:
            user = db.query(User).filter(...).first()
        
        # Use:
        def _get_user():
            with SessionLocal() as db:
                return db.query(User).filter(...).first()
        user = await run_d1_sync(_get_user)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_d1_thread_pool, lambda: func(*args, **kwargs))
