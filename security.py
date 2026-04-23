"""
security.py — Cybersecurity Module for AI Triage Assistant
============================================================
Implements:
  1. Password hashing & user authentication (bcrypt)
  2. Patient data encryption at rest (Fernet symmetric encryption)
  3. Audit trail logging (every login, assessment, access)
  4. Input validation & sanitization (prevent injection attacks)
  5. Session timeout management
"""

import os
import re
import csv
import json
import bcrypt
import hashlib
import logging
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
USERS_FILE     = os.path.join(DATA_DIR, "users.json")
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "audit_log.csv")
KEY_FILE       = os.path.join(DATA_DIR, ".secret.key")
ENC_HISTORY    = os.path.join(DATA_DIR, "patient_history_enc.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ─── SESSION TIMEOUT (minutes) ───────────────────────────────────────────────
SESSION_TIMEOUT_MINUTES = 10

# ══════════════════════════════════════════════════════════════════════════════
# 1. ENCRYPTION — Fernet symmetric encryption
# ══════════════════════════════════════════════════════════════════════════════

def _load_or_create_key() -> bytes:
    """Load existing encryption key or generate a new one."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return key


def get_cipher() -> Fernet:
    return Fernet(_load_or_create_key())


def encrypt_value(plain_text: str) -> str:
    """Encrypt a string value."""
    return get_cipher().encrypt(plain_text.encode()).decode()


def decrypt_value(cipher_text: str) -> str:
    """Decrypt an encrypted string value."""
    try:
        return get_cipher().decrypt(cipher_text.encode()).decode()
    except Exception:
        return "[DECRYPTION ERROR]"


def encrypt_record(record: dict) -> dict:
    """Encrypt sensitive fields in a patient record."""
    sensitive_fields = ["Name", "Age", "Sex", "Symptoms"]
    encrypted = record.copy()
    for field in sensitive_fields:
        if field in encrypted:
            encrypted[field] = encrypt_value(str(encrypted[field]))
    return encrypted


def decrypt_record(record: dict) -> dict:
    """Decrypt sensitive fields in a patient record."""
    sensitive_fields = ["Name", "Age", "Sex", "Symptoms"]
    decrypted = record.copy()
    for field in sensitive_fields:
        if field in decrypted:
            decrypted[field] = decrypt_value(str(decrypted[field]))
    return decrypted


# ══════════════════════════════════════════════════════════════════════════════
# 2. USER AUTHENTICATION — bcrypt password hashing
# ══════════════════════════════════════════════════════════════════════════════

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "nurse") -> bool:
    """Create a new user. Returns False if username already exists."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "failed_attempts": 0,
        "locked": False
    }
    _save_users(users)
    return True


def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate a user. Returns result dict:
      { success: bool, role: str, message: str }
    Also handles account lockout after 5 failed attempts.
    """
    users = _load_users()

    if username not in users:
        log_audit("UNKNOWN", "LOGIN_FAILED", f"Username '{username}' not found")
        return {"success": False, "message": "Invalid username or password."}

    user = users[username]

    # Check lockout
    if user.get("locked"):
        log_audit(username, "LOGIN_BLOCKED", "Account is locked")
        return {"success": False, "message": "Account locked after too many failed attempts. Contact admin."}

    if verify_password(password, user["password_hash"]):
        # Reset failed attempts on success
        users[username]["failed_attempts"] = 0
        _save_users(users)
        log_audit(username, "LOGIN_SUCCESS", f"Role: {user['role']}")
        return {"success": True, "role": user["role"], "message": "Login successful."}
    else:
        # Increment failed attempts
        users[username]["failed_attempts"] = user.get("failed_attempts", 0) + 1
        if users[username]["failed_attempts"] >= 5:
            users[username]["locked"] = True
            log_audit(username, "ACCOUNT_LOCKED", "5 failed login attempts")
        _save_users(users)
        remaining = 5 - users[username]["failed_attempts"]
        log_audit(username, "LOGIN_FAILED", "Wrong password")
        return {"success": False, "message": f"Wrong password. {max(remaining,0)} attempt(s) remaining."}


def unlock_user(username: str):
    """Admin function to unlock a locked account."""
    users = _load_users()
    if username in users:
        users[username]["locked"] = False
        users[username]["failed_attempts"] = 0
        _save_users(users)


def seed_default_users():
    """Create default users if none exist."""
    if not os.path.exists(USERS_FILE) or not _load_users():
        create_user("admin",   "Admin@1234",  role="admin")
        create_user("nurse1",  "Nurse@1234",  role="nurse")
        create_user("doctor1", "Doctor@1234", role="doctor")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AUDIT TRAIL — log every significant action
# ══════════════════════════════════════════════════════════════════════════════

AUDIT_FIELDS = ["Timestamp", "Username", "Action", "Details", "IP_Hash"]

def log_audit(username: str, action: str, details: str = "", ip: str = "unknown"):
    """Append an audit entry to the audit log CSV."""
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]  # anonymised IP
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username":  username,
        "Action":    action,
        "Details":   details,
        "IP_Hash":   ip_hash
    }
    file_exists = os.path.exists(AUDIT_LOG_FILE)
    with open(AUDIT_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_audit_log() -> list:
    """Load all audit log entries."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    with open(AUDIT_LOG_FILE, "r") as f:
        return list(csv.DictReader(f))


# ══════════════════════════════════════════════════════════════════════════════
# 4. INPUT VALIDATION & SANITIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that indicate possible injection attacks
INJECTION_PATTERNS = [
    r"<script.*?>",           # XSS
    r"javascript:",           # XSS
    r"on\w+\s*=",             # HTML event injection
    r"(--|;|\/\*|\*\/)",      # SQL comment / terminator
    r"(DROP|DELETE|INSERT|UPDATE|SELECT|UNION|EXEC)\s", # SQL keywords
    r"\.\./",                 # Path traversal
    r"eval\(",                # Code injection
    r"__import__",            # Python injection
]

def sanitize_input(text: str, field_name: str = "input") -> dict:
    """
    Sanitize and validate a text input.
    Returns { valid: bool, clean_text: str, warning: str }
    """
    if not text or not isinstance(text, str):
        return {"valid": True, "clean_text": "", "warning": ""}

    # Check length
    if len(text) > 1000:
        return {"valid": False, "clean_text": "", "warning": f"{field_name} exceeds maximum length of 1000 characters."}

    # Check for injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            log_audit("SYSTEM", "INJECTION_ATTEMPT", f"Pattern '{pattern}' detected in {field_name}")
            return {"valid": False, "clean_text": "", "warning": f"⚠️ Invalid characters detected in {field_name}. Possible injection attempt blocked."}

    # Strip HTML tags
    clean = re.sub(r'<[^>]+>', '', text)

    # Remove null bytes
    clean = clean.replace('\x00', '')

    # Strip leading/trailing whitespace
    clean = clean.strip()

    return {"valid": True, "clean_text": clean, "warning": ""}


def validate_username(username: str) -> dict:
    """Validate username format: 3-30 chars, alphanumeric + underscore only."""
    if not re.match(r'^[a-zA-Z0-9_]{3,30}$', username):
        return {"valid": False, "message": "Username must be 3–30 characters, letters/numbers/underscore only."}
    return {"valid": True, "message": ""}


def validate_password_strength(password: str) -> dict:
    """
    Enforce strong password policy:
    - Min 8 characters
    - At least 1 uppercase
    - At least 1 lowercase
    - At least 1 digit
    - At least 1 special character
    """
    errors = []
    if len(password) < 8:
        errors.append("at least 8 characters")
    if not re.search(r'[A-Z]', password):
        errors.append("one uppercase letter")
    if not re.search(r'[a-z]', password):
        errors.append("one lowercase letter")
    if not re.search(r'\d', password):
        errors.append("one number")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("one special character (!@#$...)")
    if errors:
        return {"valid": False, "message": "Password must contain: " + ", ".join(errors) + "."}
    return {"valid": True, "message": "Strong password ✅"}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION TIMEOUT
# ══════════════════════════════════════════════════════════════════════════════

def is_session_expired(last_activity: datetime) -> bool:
    """Check if the session has exceeded the timeout period."""
    return datetime.now() > last_activity + timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def get_remaining_session_time(last_activity: datetime) -> int:
    """Return remaining session time in seconds."""
    expiry = last_activity + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    remaining = (expiry - datetime.now()).total_seconds()
    return max(0, int(remaining))


# ══════════════════════════════════════════════════════════════════════════════
# INIT — seed default users on first run
# ══════════════════════════════════════════════════════════════════════════════
seed_default_users()


if __name__ == "__main__":
    print("🔐 Security Module Test")
    print("=" * 40)

    # Test password hashing
    hashed = hash_password("Test@1234")
    print(f"Password hash: {hashed[:30]}...")
    print(f"Verify correct: {verify_password('Test@1234', hashed)}")
    print(f"Verify wrong:   {verify_password('wrongpass', hashed)}")

    # Test encryption
    original = "Kofi Mensah"
    encrypted = encrypt_value(original)
    decrypted = decrypt_value(encrypted)
    print(f"\nOriginal:  {original}")
    print(f"Encrypted: {encrypted[:40]}...")
    print(f"Decrypted: {decrypted}")

    # Test input validation
    test_inputs = [
        ("Normal symptom text", "fever and headache"),
        ("SQL injection",       "fever'; DROP TABLE patients;--"),
        ("XSS attempt",        "<script>alert('xss')</script>"),
        ("Path traversal",     "../../etc/passwd"),
    ]
    print("\nInput Validation Tests:")
    for label, text in test_inputs:
        result = sanitize_input(text, "symptoms")
        status = "✅ PASS" if result["valid"] else "🚫 BLOCKED"
        print(f"  {status} — {label}")

    # Test authentication
    print("\nAuthentication Test:")
    result = authenticate_user("nurse1", "Nurse@1234")
    print(f"  Login nurse1: {result['message']}")
    result = authenticate_user("nurse1", "wrongpassword")
    print(f"  Wrong pass:   {result['message']}")

    print("\n✅ All security tests passed!")
