import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

APP_ROOT = Path(__file__).resolve().parents[1]
CHAT_STORE_PATH = APP_ROOT / "data" / "chat_sessions.json"
CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_store() -> Dict:
    return {"sessions": []}


def load_store() -> Dict:
    if not CHAT_STORE_PATH.exists():
        return _default_store()
    try:
        data = json.loads(CHAT_STORE_PATH.read_text(encoding="utf-8"))
        if "sessions" not in data or not isinstance(data["sessions"], list):
            return _default_store()
        return data
    except (json.JSONDecodeError, OSError):
        return _default_store()


def save_store(data: Dict) -> None:
    CHAT_STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_sessions() -> List[Dict]:
    data = load_store()
    sessions = data.get("sessions", [])
    return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)


def create_session(title: str = "New Chat") -> str:
    data = load_store()
    session_id = str(uuid.uuid4())
    now = _now_iso()
    data["sessions"].append(
        {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "memory_summary": "",
            "messages": [],
        }
    )
    save_store(data)
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    data = load_store()
    for session in data.get("sessions", []):
        if session.get("id") == session_id:
            return session
    return None


def add_message(session_id: str, role: str, content: str) -> None:
    data = load_store()
    for session in data.get("sessions", []):
        if session.get("id") == session_id:
            session.setdefault("messages", []).append({"role": role, "content": content, "ts": _now_iso()})
            session["updated_at"] = _now_iso()
            if role == "user" and session.get("title", "New Chat") == "New Chat":
                session["title"] = content[:48] + ("..." if len(content) > 48 else "")
            save_store(data)
            return


def delete_session(session_id: str) -> None:
    data = load_store()
    before = len(data.get("sessions", []))
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("id") != session_id]
    if len(data["sessions"]) != before:
        save_store(data)


def get_recent_messages(session_id: str, limit: int = 12) -> List[Dict]:
    session = get_session(session_id)
    if not session:
        return []
    messages = session.get("messages", [])
    if not isinstance(messages, list):
        return []
    return messages[-limit:]


def get_memory_summary(session_id: str) -> str:
    session = get_session(session_id)
    if not session:
        return ""
    summary = session.get("memory_summary", "")
    return summary if isinstance(summary, str) else ""


def set_memory_summary(session_id: str, summary: str) -> None:
    data = load_store()
    for session in data.get("sessions", []):
        if session.get("id") == session_id:
            session["memory_summary"] = summary
            session["updated_at"] = _now_iso()
            save_store(data)
            return
