from typing import Dict, List, Tuple
from threading import Lock

class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, List[Tuple[str,  str]]] = {}
        self.lock = Lock()
    
    def add_message(self, session_id:str, question: str, answer:  str):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append((question, answer))
    
    def get_session(self, session_id: str) -> List[Tuple[str, str]]:
        with self.lock:
            return list(self.sessions.get(session_id, []))
    
    def clear_session(self, session_id: str):
        with self.lock:
            self.sessions.pop(session_id, None)