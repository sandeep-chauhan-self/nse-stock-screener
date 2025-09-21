"""
NSE Stock Screener - Secrets Management System
Secure handling of API keys, database credentials, and sensitive configuration.
"""
import os
import json
import logging
from typing import Optional, Dict[str, Any], Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
import warnings
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
logger = logging.getLogger(__name__)
@dataclass
class SecretConfig:
    """Configuration for secrets management."""

    # Environment variable prefix
    env_prefix: str = "NSE_SCREENER"

    # Vault integration (future)
    vault_enabled: bool = False
    vault_url: str = ""
    vault_token: str = ""

    # Local encryption (for development)
    encryption_enabled: bool = False
    master_key_env: str = "NSE_SCREENER_MASTER_KEY"

    # Required secrets
    required_secrets: List[str] = field(default_factory=lambda: [
        "API_KEY_NSE",
        "API_KEY_YAHOO",
        "DB_PASSWORD",
        "DB_USERNAME",
        "DB_HOST",
        "REDIS_PASSWORD"
    ])

    # Optional secrets with defaults
    optional_secrets: Dict[str, str] = field(default_factory=lambda: {
        "DB_PORT": "5432",
        "DB_NAME": "nse_screener",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "LOG_LEVEL": "INFO",
        "PROMETHEUS_PORT": "8000"
    })
class SecretValidationError(Exception):
    """Raised when secret validation fails."""
    pass
class SecretsManager:
    """
    Centralized secrets management for NSE Stock Screener.
    Features:
    - Environment variable loading with validation
    - Optional encryption for local development
    - Audit logging for secret access
    - Future: HashiCorp Vault integration
    """
    def __init__(self, config: Optional[SecretConfig] = None) -> None:
        self.config = config or SecretConfig()
        self._secrets_cache: Dict[str, str] = {}
        self._cipher = None

        # Initialize encryption if enabled
        if self.config.encryption_enabled:
            self._init_encryption()

        # Load secrets
        self._load_secrets()
        logger.info("SecretsManager initialized", extra={
            "vault_enabled": self.config.vault_enabled,
            "encryption_enabled": self.config.encryption_enabled,
            "required_secrets_count": len(self.config.required_secrets)
        })
    def _init_encryption(self):
        """Initialize encryption for local secret storage."""
        master_key = os.getenv(self.config.master_key_env)
        if not master_key:
            raise SecretValidationError(
                f"Master key not found in environment variable: {self.config.master_key_env}"
            )

        # Derive encryption key from master key
        password = master_key.encode()
        salt = b'nse_screener_salt_2025'
  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._cipher = Fernet(key)
        logger.info("Encryption initialized for secrets management")
    def _load_secrets(self):
        """Load secrets from environment variables."""

        # Load required secrets
        missing_secrets = []
        for secret_name in self.config.required_secrets:
            env_var = f"{self.config.env_prefix}_{secret_name}"
            value = os.getenv(env_var)
            if not value:
                missing_secrets.append(env_var)
            else:
                self._secrets_cache[secret_name] = value
                logger.debug(f"Loaded required secret: {secret_name}")
        if missing_secrets:
            error_msg = f"Missing required environment variables: {', '.join(missing_secrets)}"
            logger.error(error_msg)
            raise SecretValidationError(error_msg)

        # Load optional secrets with defaults
        for secret_name, default_value in self.config.optional_secrets.items():
            env_var = f"{self.config.env_prefix}_{secret_name}"
            value = os.getenv(env_var, default_value)
            self._secrets_cache[secret_name] = value
            logger.debug(f"Loaded optional secret: {secret_name}")
        logger.info(f"Successfully loaded {len(self._secrets_cache)} secrets")
    def get_secret(self, name: str, masked_logging: bool = True) -> str:
        """
        Get a secret value by name.
        Args:
            name: Secret name (without prefix)
            masked_logging: Whether to mask the secret in logs
        Returns:
            Secret value
        Raises:
            SecretValidationError: If secret not found
        """
        if name not in self._secrets_cache:
            error_msg = f"Secret not found: {name}"
            logger.error(error_msg)
            raise SecretValidationError(error_msg)
        secret_value = self._secrets_cache[name]

        # Audit log secret access
        log_value = self._mask_secret(secret_value) if masked_logging else secret_value
        logger.info(f"Secret accessed: {name}", extra={
            "secret_name": name,
            "value_masked": masked_logging,
            "value_length": len(secret_value)
        })
        return secret_value
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration as a dictionary."""
        return {
            "host": self.get_secret("DB_HOST"),
            "port": self.get_secret("DB_PORT"),
            "database": self.get_secret("DB_NAME"),
            "username": self.get_secret("DB_USERNAME"),
            "password": self.get_secret("DB_PASSWORD")
        }
    def get_redis_config(self) -> Dict[str, str]:
        """Get Redis configuration as a dictionary."""
        return {
            "host": self.get_secret("REDIS_HOST"),
            "port": int(self.get_secret("REDIS_PORT")),
            "password": self.get_secret("REDIS_PASSWORD")
        }
    def get_api_keys(self) -> Dict[str, str]:
        """Get all API keys as a dictionary."""
        return {
            "nse": self.get_secret("API_KEY_NSE"),
            "yahoo": self.get_secret("API_KEY_YAHOO")
        }
    def validate_secrets(self) -> bool:
        """
        Validate all secrets are properly configured.
        Returns:
            True if all secrets are valid
        Raises:
            SecretValidationError: If validation fails
        """
        validation_errors = []

        # Validate database config
        try:
            db_config = self.get_database_config()
            if not all(db_config.values()):
                validation_errors.append("Database configuration incomplete")
        except Exception as e:
            validation_errors.append(f"Database validation failed: {e}")

        # Validate API keys
        try:
            api_keys = self.get_api_keys()
            for service, key in api_keys.items():
                if not key or len(key) < 10:
  # Basic length check
                    validation_errors.append(f"Invalid API key for {service}")
        except Exception as e:
            validation_errors.append(f"API key validation failed: {e}")
        if validation_errors:
            error_msg = "Secret validation failed: " + "; ".join(validation_errors)
            logger.error(error_msg)
            raise SecretValidationError(error_msg)
        logger.info("All secrets validated successfully")
        return True
    def encrypt_secret(self, plaintext: str) -> str:
        """
        Encrypt a secret for local storage.
        Args:
            plaintext: Secret to encrypt
        Returns:
            Base64-encoded encrypted secret
        """
        if not self.config.encryption_enabled or not self._cipher:
            raise SecretValidationError("Encryption not enabled")
        encrypted = self._cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    def decrypt_secret(self, encrypted_text: str) -> str:
        """
        Decrypt a secret from storage.
        Args:
            encrypted_text: Base64-encoded encrypted secret
        Returns:
            Decrypted secret
        """
        if not self.config.encryption_enabled or not self._cipher:
            raise SecretValidationError("Encryption not enabled")
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
        decrypted = self._cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
    def _mask_secret(self, secret: str) -> str:
        """Mask a secret for logging purposes."""
        if len(secret) <= 4:
            return "****"
        return secret[:2] + "*" * (len(secret) - 4) + secret[-2:]
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on secrets management.
        Returns:
            Health status dictionary
        """
        try:
            self.validate_secrets()
            return {
                "status": "healthy",
                "secrets_loaded": len(self._secrets_cache),
                "encryption_enabled": self.config.encryption_enabled,
                "vault_enabled": self.config.vault_enabled
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "secrets_loaded": len(self._secrets_cache)
            }
    def list_secret_names(self) -> List[str]:
        """List[str] all available secret names (for debugging)."""
        return List[str](self._secrets_cache.keys())

# Singleton instance for global access
_secrets_manager: Optional[SecretsManager] = None
def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
def init_secrets_manager(config: Optional[SecretConfig] = None) -> SecretsManager:
    """Initialize the global secrets manager with custom config."""
    global _secrets_manager
    _secrets_manager = SecretsManager(config)
    return _secrets_manager

# Convenience functions for common use cases
def get_database_url() -> str:
    """Get database connection URL."""
    secrets = get_secrets_manager()
    config = secrets.get_database_config()
    return f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
def get_redis_url() -> str:
    """Get Redis connection URL."""
    secrets = get_secrets_manager()
    config = secrets.get_redis_config()
    if config['password']:
        return f"redis://:{config['password']}@{config['host']}:{config['port']}/0"
    else:
        return f"redis://{config['host']}:{config['port']}/0"
def get_api_key(service: str) -> str:
    """Get API key for a specific service."""
    secrets = get_secrets_manager()
    api_keys = secrets.get_api_keys()
    if service not in api_keys:
        raise SecretValidationError(f"API key not found for service: {service}")
    return api_keys[service]
if __name__ == "__main__":

    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    try:

        # Initialize secrets manager
        secrets = SecretsManager()

        # Validate all secrets
        secrets.validate_secrets()

        # Get specific configurations
        db_config = secrets.get_database_config()
        print(f"Database config loaded: {List[str](db_config.keys())}")

        # Health check
        health = secrets.health_check()
        print(f"Health status: {health['status']}")
    except SecretValidationError as e:
        print(f"Secrets validation failed: {e}")
        print("\nRequired environment variables:")
        config = SecretConfig()
        for secret in config.required_secrets:
            print(f"  {config.env_prefix}_{secret}")
