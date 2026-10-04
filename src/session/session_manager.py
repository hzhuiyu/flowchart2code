"""
Session Manager
Manages the state of the whole processing flow: tracks which agent runs in
the current round and stores intermediate results.
"""

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class AgentType(Enum):
    """Agent type enumeration"""
    FLOWCHART_EXTRACTOR = "agent1_flowchart_extractor"
    PSEUDOCODE_GENERATOR = "agent2_pseudocode_to_code"
    VALIDATOR = "agent3_validation"


class SessionStatus(Enum):
    """Session status enumeration"""
    INIT = "init"
    FLOWCHART_PROCESSING = "flowchart_processing"
    PROBLEM_EXTRACTING = "problem_extracting"
    PSEUDOCODE_GENERATING = "pseudocode_generating"
    CODE_GENERATING = "code_generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionManager:
    """Session manager"""

    def __init__(self, session_dir: str = "session"):
        """
        Initialize the session manager.

        Args:
            session_dir: directory where sessions are stored
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_sessions = {}  # in-memory session cache

    def create_session(self, image_path: str, dataset: str = None) -> str:
        """
        Create a new session.

        Args:
            image_path: flowchart image path
            dataset: dataset name

        Returns:
            session id
        """
        session_id = str(uuid.uuid4())

        session_data = {
            'session_id': session_id,
            'status': SessionStatus.INIT.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'current_agent': AgentType.FLOWCHART_EXTRACTOR.value,
            'input': {
                'image_path': image_path,
                'dataset': dataset
            },
            'output': {
                'flowchart_data': None,
                'problem_info': None,
                'pseudocode': None,
                'generated_code': None,
                'validation_result': None
            },
            'error': None,
            'logs': []
        }

        self.current_sessions[session_id] = session_data
        self._save_session(session_id)

        self._add_log(session_id, f"Session created, image path: {image_path}")

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.

        Args:
            session_id: session id

        Returns:
            session data, or None if it does not exist
        """
        # Check the in-memory cache first
        if session_id in self.current_sessions:
            return self.current_sessions[session_id]

        # Load from file
        session_data = self._load_session(session_id)
        if session_data:
            self.current_sessions[session_id] = session_data

        return session_data

    def update_session_status(self, session_id: str, status: SessionStatus,
                            current_agent: AgentType = None, error: str = None) -> bool:
        """
        Update the session status.

        Args:
            session_id: session id
            status: new status
            current_agent: current agent
            error: error message

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['status'] = status.value
        session['updated_at'] = datetime.now().isoformat()

        if current_agent:
            session['current_agent'] = current_agent.value

        if error:
            session['error'] = error
            self._add_log(session_id, f"Error: {error}")
        else:
            self._add_log(session_id, f"Status updated to: {status.value}")

        self._save_session(session_id)
        return True

    def set_flowchart_data(self, session_id: str, flowchart_data: Dict[str, Any]) -> bool:
        """
        Set the flowchart data.

        Args:
            session_id: session id
            flowchart_data: flowchart data

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['output']['flowchart_data'] = flowchart_data
        session['updated_at'] = datetime.now().isoformat()

        self._add_log(session_id, f"Flowchart data set, containing {len(flowchart_data.get('nodes', []))} nodes")

        self._save_session(session_id)
        return True

    def set_problem_info(self, session_id: str, problem_info: Dict[str, Any]) -> bool:
        """
        Set the problem information.

        Args:
            session_id: session id
            problem_info: problem information

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['output']['problem_info'] = problem_info
        session['updated_at'] = datetime.now().isoformat()

        self._add_log(session_id, f"Problem info set: {problem_info.get('title', 'Unknown')}")

        self._save_session(session_id)
        return True

    def set_pseudocode(self, session_id: str, pseudocode: str) -> bool:
        """
        Set the pseudocode.

        Args:
            session_id: session id
            pseudocode: pseudocode

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['output']['pseudocode'] = pseudocode
        session['updated_at'] = datetime.now().isoformat()

        self._add_log(session_id, f"Pseudocode generated, length: {len(pseudocode)} chars")

        self._save_session(session_id)
        return True

    def set_generated_code(self, session_id: str, code: str) -> bool:
        """
        Set the generated code.

        Args:
            session_id: session id
            code: generated code

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['output']['generated_code'] = code
        session['updated_at'] = datetime.now().isoformat()

        self._add_log(session_id, f"Code generated, length: {len(code)} chars")

        self._save_session(session_id)
        return True

    def set_validation_result(self, session_id: str, validation_result: Dict[str, Any]) -> bool:
        """
        Set the validation result.

        Args:
            session_id: session id
            validation_result: validation result

        Returns:
            whether the update succeeded
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session['output']['validation_result'] = validation_result
        session['updated_at'] = datetime.now().isoformat()

        self._add_log(session_id, f"Validation result set")

        self._save_session(session_id)
        return True

    def get_next_agent(self, session_id: str) -> Optional[AgentType]:
        """
        Get the next agent that should run.

        Args:
            session_id: session id

        Returns:
            the next agent type, or None when the flow is finished
        """
        session = self.get_session(session_id)
        if not session:
            return None

        current_agent = session['current_agent']
        status = session['status']

        # Determine the next step from the current state and agent
        if status == SessionStatus.COMPLETED.value or status == SessionStatus.FAILED.value:
            return None

        if current_agent == AgentType.FLOWCHART_EXTRACTOR.value:
            return AgentType.FLOWCHART_EXTRACTOR
        elif current_agent == AgentType.PSEUDOCODE_GENERATOR.value:
            return AgentType.PSEUDOCODE_GENERATOR
        elif current_agent == AgentType.VALIDATOR.value:
            return AgentType.VALIDATOR

        return None

    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """
        Get session progress information.

        Args:
            session_id: session id

        Returns:
            progress information
        """
        session = self.get_session(session_id)
        if not session:
            return {'error': 'session not found'}

        output = session['output']
        total_steps = 4  # 4 major steps in total
        completed_steps = 0

        if output.get('flowchart_data'):
            completed_steps += 1
        if output.get('problem_info'):
            completed_steps += 1
        if output.get('pseudocode'):
            completed_steps += 1
        if output.get('generated_code'):
            completed_steps += 1

        progress = {
            'session_id': session_id,
            'status': session['status'],
            'current_agent': session['current_agent'],
            'progress_percentage': (completed_steps / total_steps) * 100,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'created_at': session['created_at'],
            'updated_at': session['updated_at'],
            'error': session.get('error')
        }

        return progress

    def list_sessions(self, status: SessionStatus = None) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Args:
            status: filter by status; None lists all sessions

        Returns:
            list of sessions
        """
        sessions = []

        # Scan the session directory
        for session_file in self.session_dir.glob('*.json'):
            session_id = session_file.stem
            session_data = self._load_session(session_id)
            if session_data:
                if status is None or session_data['status'] == status.value:
                    sessions.append({
                        'session_id': session_id,
                        'status': session_data['status'],
                        'current_agent': session_data['current_agent'],
                        'created_at': session_data['created_at'],
                        'updated_at': session_data['updated_at']
                    })

        # Sort by creation time
        sessions.sort(key=lambda x: x['created_at'], reverse=True)

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: session id

        Returns:
            whether the deletion succeeded
        """
        try:
            # Remove from the in-memory cache
            if session_id in self.current_sessions:
                del self.current_sessions[session_id]

            # Remove the file
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            return True

        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    def _add_log(self, session_id: str, message: str):
        """Append a log entry."""
        session = self.get_session(session_id)
        if session:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            session['logs'].append(log_entry)

            # Cap the number of log entries
            if len(session['logs']) > 100:
                session['logs'] = session['logs'][-100:]

    def _save_session(self, session_id: str):
        """Save the session to a file."""
        session = self.current_sessions.get(session_id)
        if session:
            session_file = self.session_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, ensure_ascii=False, indent=2)

    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session from a file."""
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session file: {e}")
            return None


def test_session_manager():
    """Test the session manager."""
    manager = SessionManager()

    # Create a test session
    session_id = manager.create_session("test_image.png", "Algorithm")
    print(f"Session created: {session_id}")

    # Update the status
    manager.update_session_status(session_id, SessionStatus.FLOWCHART_PROCESSING)

    # Set the flowchart data
    flowchart_data = {
        "nodes": [{"id": 1, "type": "start", "label": "Start"}],
        "edges": []
    }
    manager.set_flowchart_data(session_id, flowchart_data)

    # Get progress
    progress = manager.get_session_progress(session_id)
    print(f"Session progress: {progress['progress_percentage']:.1f}%")

    # List all sessions
    sessions = manager.list_sessions()
    print(f"Total sessions: {len(sessions)}")


if __name__ == "__main__":
    test_session_manager()
