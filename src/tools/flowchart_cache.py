"""
Flowchart Cache Tool
Used to check and read extracted flowchart information
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FlowchartCache:
    """Flowchart cache tool"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize flowchart cache tool

        Args:
            output_dir: Output directory path
        """
        # Ensure absolute path is used
        if output_dir:
            self.output_dir = str(Path(output_dir).resolve())
        else:
            self.output_dir = str((Path(__file__).parent.parent.parent / "output").resolve())
        self._cache = {}  # In-memory cache to avoid repeated file reads
    
    def _get_nodes_file_path(self, dataset: str) -> str:
        """Get nodes file path (absolute path)"""
        return str(Path(self.output_dir).resolve() / dataset / "nodes.jsonl")
    
    def is_extracted(self, task_id: str, dataset: str, allow_empty: bool = False) -> bool:
        """
        Check if the flowchart information for the specified task has been extracted
        
        Args:
            task_id: Task ID
            dataset: Dataset name
            allow_empty: Whether to allow empty results (default False, empty results are considered not extracted)
            
        Returns:
            True if extracted, False otherwise
        """
        flowchart_data = self.get_flowchart_data(task_id, dataset)
        if flowchart_data is None:
            return False
        if allow_empty:
            return True
        return len(flowchart_data.get("nodes", [])) > 0
    
    def get_flowchart_data(self, task_id: str, dataset: str) -> Optional[Dict[str, Any]]:
        """
        Get flowchart data for the specified task
        
        Args:
            task_id: Task ID
            dataset: Dataset name
            
        Returns:
            Flowchart data, or None if not found
        """
        # Check in-memory cache
        cache_key = f"{dataset}:{task_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Read from file
        nodes_file = self._get_nodes_file_path(dataset)
        if not os.path.exists(nodes_file):
            return None
        
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line.strip())
                        if task_id in entry:
                            flowchart_data = entry[task_id]
                            # Store in in-memory cache
                            self._cache[cache_key] = flowchart_data
                            return flowchart_data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading nodes file: {e}")
        
        return None
    
    def save_flowchart_data(self, task_id: str, dataset: str,
                           flowchart_data: Dict[str, Any]) -> None:
        """
        Save flowchart data to cache file

        Args:
            task_id: Task ID
            dataset: Dataset name
            flowchart_data: Flowchart data
        """
        # Create output directory (using absolute path)
        output_dir = Path(self.output_dir).resolve() / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get nodes file path
        nodes_file = self._get_nodes_file_path(dataset)

        # Check if data for this task already exists
        existing_data = self.get_flowchart_data(task_id, dataset)

        if existing_data:
            # If exists, update the record
            self._update_entry(nodes_file, task_id, flowchart_data)
        else:
            # If not exists, append new record
            node_entry = {task_id: flowchart_data}
            with open(nodes_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(node_entry, ensure_ascii=False) + '\n')

        # Update in-memory cache
        cache_key = f"{dataset}:{task_id}"
        self._cache[cache_key] = flowchart_data
    
    def _update_entry(self, nodes_file: str, task_id: str, 
                     flowchart_data: Dict[str, Any]) -> None:
        """
        Update an existing record
        
        Args:
            nodes_file: Nodes file path
            task_id: Task ID
            flowchart_data: Flowchart data
        """
        # Read all records
        entries = []
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    if task_id in entry:
                        # Update this record
                        entries.append({task_id: flowchart_data})
                    else:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        # Rewrite file
        with open(nodes_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def list_extracted_tasks(self, dataset: str) -> List[str]:
        """
        List all extracted task IDs
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of task IDs
        """
        nodes_file = self._get_nodes_file_path(dataset)
        if not os.path.exists(nodes_file):
            return []
        
        task_ids = []
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line.strip())
                        task_ids.extend(entry.keys())
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading nodes file: {e}")
        
        return task_ids
    
    def clear_cache(self, dataset: str = None) -> None:
        """
        Clear cache
        
        Args:
            dataset: Dataset name, if None clear all cache
        """
        if dataset:
            # Clear cache for specified dataset
            nodes_file = self._get_nodes_file_path(dataset)
            if os.path.exists(nodes_file):
                os.remove(nodes_file)
            # Clear in-memory cache for this dataset
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{dataset}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            # Clear all cache
            self._cache.clear()
            output_path = Path(self.output_dir)
            for dataset_dir in output_path.iterdir():
                if dataset_dir.is_dir():
                    nodes_file = dataset_dir / "nodes.jsonl"
                    if nodes_file.exists():
                        nodes_file.unlink()
