"""
ILR Interpreter
Directly interprets and executes ILR (Intermediate Logic Representation),
used to verify the correctness of ILR logic generated from flowcharts
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Node types"""
    START = "start"
    PROCESS = "process"
    DECISION = "decision"
    END = "end"
    CALL = "call"      # Function call node
    RETURN = "return"  # Function return node


@dataclass
class ILRNode:
    """ILR node data class"""
    id: int
    type: NodeType
    label: str
    bbox: Optional[List[int]] = None
    # Process node
    action: Optional[str] = None
    # Decision node
    logic: Optional[str] = None
    true_next: Optional[int] = None
    false_next: Optional[int] = None
    # Start/Process node
    next: Optional[int] = None
    # End node
    output: Optional[str] = None
    # Call node
    function_name: Optional[str] = None
    arguments: Optional[List[str]] = None
    return_var: Optional[str] = None
    # Return node
    value: Optional[str] = None


@dataclass
class ILRFunction:
    """ILR function definition"""
    name: str
    parameters: List[str]
    entry_node: int
    nodes: Dict[int, ILRNode]


class ILRParseError(Exception):
    """ILR parse error"""
    pass


class ILRExecutionError(Exception):
    """ILR execution error"""
    pass


class ILRInterpreter:
    """ILR Interpreter - directly executes ILR logic, supports function calls and recursion"""

    # Allowed built-in functions whitelist
    ALLOWED_BUILTINS = {
        'abs': abs,
        'len': len,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sum': sum,
        'min': min,
        'max': max,
        'sorted': sorted,
        'reversed': reversed,
        'round': round,
        'pow': pow,
        'divmod': divmod,
        'ord': ord,
        'chr': chr,
        'isinstance': isinstance,
        'type': type,
        'all': all,
        'any': any,
        'iter': iter,
    }

    def __init__(self, max_iterations: int = 10000, max_recursion_depth: int = 1000):
        """
        Initialize ILR interpreter

        Args:
            max_iterations: Maximum iteration count to prevent infinite loops
            max_recursion_depth: Maximum recursion depth to prevent infinite recursion
        """
        self.max_iterations = max_iterations
        self.max_recursion_depth = max_recursion_depth
        self.nodes: Dict[int, ILRNode] = {}
        self.functions: Dict[str, ILRFunction] = {}  # Function name -> ILRFunction
        self.problem_id: Optional[str] = None

    def parse(self, ilr_json: Union[str, Dict]) -> None:
        """
        Parse ILR JSON

        Args:
            ilr_json: ILR JSON string or dictionary

        Raises:
            ILRParseError: Raised when parsing fails
        """
        # If it's a string, first parse to dictionary
        if isinstance(ilr_json, str):
            # Clean possible markdown code block markers
            ilr_json = self._clean_json_string(ilr_json)
            try:
                ilr_dict = json.loads(ilr_json)
            except json.JSONDecodeError as e:
                raise ILRParseError(f"JSON parsing failed: {e}")
        else:
            ilr_dict = ilr_json

        # Validate basic structure
        if not isinstance(ilr_dict, dict):
            raise ILRParseError("ILR must be a JSON object")

        if 'nodes' not in ilr_dict:
            raise ILRParseError("ILR missing 'nodes' field")

        self.problem_id = ilr_dict.get('problem_id', 'unknown')

        # Parse main function nodes
        self.nodes = {}
        for node_data in ilr_dict['nodes']:
            node = self._parse_node(node_data)
            self.nodes[node.id] = node

        # Parse functions array (if exists)
        self.functions = {}
        if 'functions' in ilr_dict:
            for func_data in ilr_dict['functions']:
                func = self._parse_function(func_data)
                self.functions[func.name] = func

        # Validate ILR structure
        self._validate_structure()

    def _clean_json_string(self, json_str: str) -> str:
        """Clean markdown markers from JSON string"""
        # Remove markdown code block markers
        json_str = re.sub(r'```json\s*\n?', '', json_str)
        json_str = re.sub(r'```\s*\n?', '', json_str)
        return json_str.strip()

    def _parse_function(self, func_data: Dict) -> ILRFunction:
        """Parse function definition"""
        try:
            name = func_data['name']
            parameters = func_data.get('parameters', [])
            entry_node = func_data.get('entry_node', 1)

            # Parse function nodes
            nodes = {}
            for node_data in func_data.get('nodes', []):
                node = self._parse_node(node_data)
                nodes[node.id] = node

            return ILRFunction(
                name=name,
                parameters=parameters,
                entry_node=entry_node,
                nodes=nodes
            )
        except KeyError as e:
            raise ILRParseError(f"Function definition missing required field: {e}")

    def _parse_node(self, node_data: Dict) -> ILRNode:
        """Parse a single node"""
        try:
            node_id = node_data['id']
            node_type_str = node_data['type']

            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                raise ILRParseError(f"Unknown node type: {node_type_str}")

            node = ILRNode(
                id=node_id,
                type=node_type,
                label=node_data.get('label', ''),
                bbox=node_data.get('bbox'),
            )

            # Set specific fields based on type
            if node_type == NodeType.PROCESS:
                node.action = node_data.get('action')
                node.next = node_data.get('next')
            elif node_type == NodeType.DECISION:
                node.logic = node_data.get('logic')
                node.true_next = node_data.get('true_next')
                node.false_next = node_data.get('false_next')
            elif node_type == NodeType.START:
                node.next = node_data.get('next')
            elif node_type == NodeType.END:
                node.output = node_data.get('output')
            elif node_type == NodeType.CALL:
                node.function_name = node_data.get('function_name')
                node.arguments = node_data.get('arguments', [])
                node.return_var = node_data.get('return_var')
                node.next = node_data.get('next')
            elif node_type == NodeType.RETURN:
                node.value = node_data.get('value')

            return node

        except KeyError as e:
            raise ILRParseError(f"Node missing required field: {e}")

    def _validate_structure(self) -> None:
        """Validate ILR structure integrity"""
        # Validate main function nodes
        self._validate_nodes(self.nodes, "main")

        # Validate each function's nodes
        for func_name, func in self.functions.items():
            self._validate_function_nodes(func)

    def _validate_nodes(self, nodes: Dict[int, ILRNode], context: str) -> None:
        """Validate node set integrity"""
        # Check for unique start node
        start_nodes = [n for n in nodes.values() if n.type == NodeType.START]
        if len(start_nodes) == 0:
            raise ILRParseError(f"{context} missing start node")
        if len(start_nodes) > 1:
            raise ILRParseError(f"{context} has multiple start nodes: {[n.id for n in start_nodes]}")

        # Check all jump targets exist
        for node in nodes.values():
            if node.next is not None and node.next not in nodes:
                raise ILRParseError(f"{context} node {node.id} next points to non-existent node {node.next}")
            if node.true_next is not None and node.true_next not in nodes:
                raise ILRParseError(f"{context} node {node.id} true_next points to non-existent node {node.true_next}")
            if node.false_next is not None and node.false_next not in nodes:
                raise ILRParseError(f"{context} node {node.id} false_next points to non-existent node {node.false_next}")

        # Check for termination nodes (end or return)
        end_nodes = [n for n in nodes.values() if n.type in (NodeType.END, NodeType.RETURN)]
        if len(end_nodes) == 0:
            raise ILRParseError(f"{context} missing termination node (end or return)")

    def _validate_function_nodes(self, func: ILRFunction) -> None:
        """Validate function node integrity"""
        # Check if entry node exists
        if func.entry_node not in func.nodes:
            # Try to find start node
            start_nodes = [n for n in func.nodes.values() if n.type == NodeType.START]
            if start_nodes:
                func.entry_node = start_nodes[0].id
            else:
                raise ILRParseError(f"Function {func.name} entry node {func.entry_node} does not exist")

        self._validate_nodes(func.nodes, f"Function {func.name}")

    def _extract_variables_from_code(self, code: str) -> set:
        """
        Extract variable names that may be used in code

        Args:
            code: Python code string

        Returns:
            Set of variable names
        """
        if not code:
            return set()

        # Match ctx['var'] or ctx["var"] pattern
        pattern = r"ctx\[['\"]([\w]+)['\"]\]"
        return set(re.findall(pattern, code))

    def _initialize_variables(self, nodes: Dict[int, ILRNode], ctx: Dict[str, Any]) -> None:
        """
        Scan all nodes, set default values for uninitialized variables

        Args:
            nodes: Node dictionary
            ctx: Execution context (will be modified)
        """
        all_vars = set()

        # Scan all nodes to extract variables
        for node in nodes.values():
            if node.type == NodeType.PROCESS and node.action:
                all_vars.update(self._extract_variables_from_code(node.action))
            elif node.type == NodeType.DECISION and node.logic:
                all_vars.update(self._extract_variables_from_code(node.logic))
            elif node.type == NodeType.END and node.output:
                all_vars.update(self._extract_variables_from_code(node.output))
            elif node.type == NodeType.RETURN and node.value:
                all_vars.update(self._extract_variables_from_code(node.value))

        # Set default values for uninitialized variables
        for var in all_vars:
            if var not in ctx:
                ctx[var] = None  # Use None as default value

    def execute(self, inputs: Dict[str, Any] = None) -> Any:
        """
        Execute ILR

        Args:
            inputs: Input parameter dictionary, will be placed into ctx

        Returns:
            Execution result

        Raises:
            ILRExecutionError: Raised when execution fails
        """
        if not self.nodes:
            raise ILRExecutionError("No ILR to execute, please call parse() method first")

        # Initialize context
        ctx = dict(inputs) if inputs else {}

        # [Fix] Automatically unpack args parameter to arg0, arg1, arg2, etc.
        # This allows ILR to access parameters via ctx['arg0'], ctx['arg1'], etc.
        if 'args' in ctx and isinstance(ctx['args'], (list, tuple)):
            for i, arg in enumerate(ctx['args']):
                ctx[f'arg{i}'] = arg

        # Execute main function
        return self._execute_nodes(self.nodes, ctx, recursion_depth=0)

    def _execute_nodes(self, nodes: Dict[int, ILRNode], ctx: Dict[str, Any],
                       recursion_depth: int = 0) -> Any:
        """
        Execute a set of nodes (supports function calls)

        Args:
            nodes: Node dictionary
            ctx: Execution context
            recursion_depth: Current recursion depth

        Returns:
            Execution result
        """
        if recursion_depth > self.max_recursion_depth:
            raise ILRExecutionError(f"Recursion depth exceeded maximum limit ({self.max_recursion_depth})")

        # [New] Initialize all variables that may be used
        self._initialize_variables(nodes, ctx)

        # Create safe execution environment
        safe_globals = {
            '__builtins__': self.ALLOWED_BUILTINS,
            'ctx': ctx,
        }
        safe_locals = {'ctx': ctx}

        # Find start node
        start_node = next((n for n in nodes.values() if n.type == NodeType.START), None)
        if start_node is None:
            raise ILRExecutionError("Start node not found")

        current_node_id = start_node.next

        iteration = 0
        execution_trace = []  # Execution trace for debugging

        while current_node_id is not None:
            if iteration >= self.max_iterations:
                raise ILRExecutionError(f"Execution exceeded maximum iteration count ({self.max_iterations})，possible infinite loop")

            iteration += 1
            current_node = nodes.get(current_node_id)

            if current_node is None:
                raise ILRExecutionError(f"Node ID not found: {current_node_id}")

            execution_trace.append({
                'iteration': iteration,
                'node_id': current_node_id,
                'type': current_node.type.value,
                'label': current_node.label,
            })

            if current_node.type == NodeType.PROCESS:
                # Execute process node's action
                if current_node.action:
                    try:
                        exec(current_node.action, safe_globals, safe_locals)
                        # Sync ctx
                        ctx = safe_locals['ctx']
                        safe_globals['ctx'] = ctx
                    except Exception as e:
                        raise ILRExecutionError(
                            f"Failed to execute action of node {current_node_id}: {current_node.action}\nError: {e}"
                        )
                current_node_id = current_node.next

            elif current_node.type == NodeType.DECISION:
                # Execute decision node's logic
                if current_node.logic:
                    try:
                        result = eval(current_node.logic, safe_globals, safe_locals)
                    except Exception as e:
                        # Provide more detailed error information
                        ctx_info = {k: f"{type(v).__name__}({v})" if not isinstance(v, (list, dict)) else f"{type(v).__name__}(len={len(v)})"
                                   for k, v in ctx.items()}
                        raise ILRExecutionError(
                            f"Failed to execute logic of node {current_node_id}: {current_node.logic}\n"
                            f"Error: {e}\n"
                            f"Current ctx variables: {ctx_info}"
                        )

                    if result:
                        current_node_id = current_node.true_next
                    else:
                        current_node_id = current_node.false_next
                else:
                    raise ILRExecutionError(f"Decision node {current_node_id} missing logic field")

            elif current_node.type == NodeType.CALL:
                # Execute function call
                func_name = current_node.function_name
                if not func_name:
                    raise ILRExecutionError(f"Call node {current_node_id} missing function_name field")

                # Find function
                func = self.functions.get(func_name)
                if func is None:
                    raise ILRExecutionError(f"Function not found: {func_name}")

                # Calculate parameter values
                arg_values = []
                for arg_expr in (current_node.arguments or []):
                    try:
                        arg_value = eval(arg_expr, safe_globals, safe_locals)
                        arg_values.append(arg_value)
                    except Exception as e:
                        raise ILRExecutionError(
                            f"Failed to compute function parameter: {arg_expr}\nError: {e}"
                        )

                # Create new context, pass in parameters
                new_ctx = {}
                for i, param_name in enumerate(func.parameters):
                    if i < len(arg_values):
                        new_ctx[param_name] = arg_values[i]

                # Recursively execute function
                return_value = self._execute_nodes(func.nodes, new_ctx, recursion_depth + 1)

                # Store return value in current context
                if current_node.return_var:
                    ctx[current_node.return_var] = return_value
                    safe_locals['ctx'] = ctx
                    safe_globals['ctx'] = ctx

                current_node_id = current_node.next

            elif current_node.type == NodeType.RETURN:
                # Execute return node, return value to caller
                if current_node.value:
                    try:
                        result = eval(current_node.value, safe_globals, safe_locals)
                        return result
                    except Exception as e:
                        raise ILRExecutionError(
                            f"Failed to execute value of node {current_node_id}: {current_node.value}\nError: {e}"
                        )
                else:
                    return None

            elif current_node.type == NodeType.END:
                # Execute end node's output
                if current_node.output:
                    try:
                        result = eval(current_node.output, safe_globals, safe_locals)
                        return result
                    except Exception as e:
                        raise ILRExecutionError(
                            f"Failed to execute output of node {current_node_id}: {current_node.output}\nError: {e}"
                        )
                else:
                    # No output, return None
                    return None

            else:
                raise ILRExecutionError(f"Unknown node type: {current_node.type}")

        raise ILRExecutionError("Execution flow did not reach termination node")

    def _execute_nodes_with_trace(self, nodes: Dict[int, ILRNode], ctx: Dict[str, Any],
                                   recursion_depth: int = 0) -> Tuple[Any, List[Dict]]:
        """
        Execute a set of nodes and record detailed trace information

        Args:
            nodes: Node dictionary
            ctx: Execution context
            recursion_depth: Current recursion depth

        Returns:
            (Execution result, trace information list)
        """
        if recursion_depth > self.max_recursion_depth:
            raise ILRExecutionError(f"Recursion depth exceeded maximum limit ({self.max_recursion_depth})")

        # Initialize all variables that may be used
        self._initialize_variables(nodes, ctx)

        # Create safe execution environment
        safe_globals = {
            '__builtins__': self.ALLOWED_BUILTINS,
            'ctx': ctx,
        }
        safe_locals = {'ctx': ctx}

        # Find start node
        start_node = next((n for n in nodes.values() if n.type == NodeType.START), None)
        if start_node is None:
            raise ILRExecutionError("Start node not found")

        current_node_id = start_node.next

        iteration = 0
        execution_trace = []  # Detailed execution trace

        # Record initial state
        execution_trace.append({
            'step': 0,
            'node_id': start_node.id,
            'node_type': 'start',
            'node_label': start_node.label,
            'action': None,
            'condition': None,
            'condition_result': None,
            'ctx_before': dict(ctx),
            'ctx_after': dict(ctx),
            'ctx_changes': {},
            'next_node': current_node_id,
            'error': None
        })

        while current_node_id is not None:
            if iteration >= self.max_iterations:
                raise ILRExecutionError(f"Execution exceeded maximum iteration count ({self.max_iterations})，possible infinite loop")

            iteration += 1
            current_node = nodes.get(current_node_id)

            if current_node is None:
                raise ILRExecutionError(f"Node ID not found: {current_node_id}")

            # Record context before execution
            ctx_before = dict(ctx)
            next_node = None
            error = None
            action = None
            condition = None
            condition_result = None

            try:
                if current_node.type == NodeType.PROCESS:
                    # Execute process node's action
                    action = current_node.action
                    if current_node.action:
                        try:
                            exec(current_node.action, safe_globals, safe_locals)
                            # Sync ctx
                            ctx = safe_locals['ctx']
                            safe_globals['ctx'] = ctx
                        except Exception as e:
                            error = f"Failed to execute action: {str(e)}"
                            raise ILRExecutionError(
                                f"Failed to execute action of node {current_node_id}: {current_node.action}\nError: {e}"
                            )
                    next_node = current_node.next

                elif current_node.type == NodeType.DECISION:
                    # Execute decision node's logic
                    condition = current_node.logic
                    if current_node.logic:
                        try:
                            result = eval(current_node.logic, safe_globals, safe_locals)
                            condition_result = bool(result)
                        except Exception as e:
                            # Provide more detailed error information
                            ctx_info = {k: f"{type(v).__name__}({v})" if not isinstance(v, (list, dict)) else f"{type(v).__name__}(len={len(v)})"
                                       for k, v in ctx.items()}
                            error = f"Failed to execute logic: {str(e)}, ctx variables: {ctx_info}"
                            raise ILRExecutionError(
                                f"Failed to execute logic of node {current_node_id}: {current_node.logic}\n"
                                f"Error: {e}\n"
                                f"Current ctx variables: {ctx_info}"
                            )

                        if result:
                            next_node = current_node.true_next
                        else:
                            next_node = current_node.false_next
                    else:
                        error = "Decision node  missing logic field"
                        raise ILRExecutionError(f"Decision node {current_node_id} missing logic field")

                elif current_node.type == NodeType.CALL:
                    # Execute function call
                    func_name = current_node.function_name
                    if not func_name:
                        error = "Call node  missing function_name field"
                        raise ILRExecutionError(f"Call node {current_node_id} missing function_name field")

                    # Find function
                    func = self.functions.get(func_name)
                    if func is None:
                        error = f"Function not found: {func_name}"
                        raise ILRExecutionError(f"Function not found: {func_name}")

                    # Calculate parameter values
                    arg_values = []
                    for arg_expr in (current_node.arguments or []):
                        try:
                            arg_value = eval(arg_expr, safe_globals, safe_locals)
                            arg_values.append(arg_value)
                        except Exception as e:
                            error = f"Failed to compute function parameter: {str(e)}"
                            raise ILRExecutionError(
                                f"Failed to compute function parameter: {arg_expr}\nError: {e}"
                            )

                    # Create new context, pass in parameters
                    new_ctx = {}
                    for i, param_name in enumerate(func.parameters):
                        if i < len(arg_values):
                            new_ctx[param_name] = arg_values[i]

                    # Recursively execute function (don't record sub-function trace)
                    return_value = self._execute_nodes(func.nodes, new_ctx, recursion_depth + 1)

                    # Store return value in current context
                    if current_node.return_var:
                        ctx[current_node.return_var] = return_value
                        safe_locals['ctx'] = ctx
                        safe_globals['ctx'] = ctx

                    action = f"call {func_name}({', '.join(map(str, arg_values))}) -> {return_value}"
                    next_node = current_node.next

                elif current_node.type == NodeType.RETURN:
                    # Execute return node, return value to caller
                    if current_node.value:
                        try:
                            result = eval(current_node.value, safe_globals, safe_locals)
                            # Record trace
                            ctx_after = dict(ctx)
                            ctx_changes = {k: (ctx_before.get(k), ctx_after.get(k))
                                         for k in set(list(ctx_before.keys()) + list(ctx_after.keys()))
                                         if ctx_before.get(k) != ctx_after.get(k)}

                            execution_trace.append({
                                'step': iteration,
                                'node_id': current_node_id,
                                'node_type': 'return',
                                'node_label': current_node.label,
                                'action': f"return {current_node.value}",
                                'condition': None,
                                'condition_result': None,
                                'ctx_before': ctx_before,
                                'ctx_after': ctx_after,
                                'ctx_changes': ctx_changes,
                                'return_value': result,
                                'next_node': None,
                                'error': None
                            })
                            return result, execution_trace
                        except Exception as e:
                            error = f"Failed to execute return: {str(e)}"
                            raise ILRExecutionError(
                                f"Failed to execute value of node {current_node_id}: {current_node.value}\nError: {e}"
                            )
                    else:
                        execution_trace.append({
                            'step': iteration,
                            'node_id': current_node_id,
                            'node_type': 'return',
                            'node_label': current_node.label,
                            'action': 'return None',
                            'condition': None,
                            'condition_result': None,
                            'ctx_before': ctx_before,
                            'ctx_after': dict(ctx),
                            'ctx_changes': {},
                            'return_value': None,
                            'next_node': None,
                            'error': None
                        })
                        return None, execution_trace

                elif current_node.type == NodeType.END:
                    # Execute end node's output
                    if current_node.output:
                        try:
                            result = eval(current_node.output, safe_globals, safe_locals)
                            # Record trace
                            ctx_after = dict(ctx)
                            ctx_changes = {k: (ctx_before.get(k), ctx_after.get(k))
                                         for k in set(list(ctx_before.keys()) + list(ctx_after.keys()))
                                         if ctx_before.get(k) != ctx_after.get(k)}

                            execution_trace.append({
                                'step': iteration,
                                'node_id': current_node_id,
                                'node_type': 'end',
                                'node_label': current_node.label,
                                'action': f"output {current_node.output}",
                                'condition': None,
                                'condition_result': None,
                                'ctx_before': ctx_before,
                                'ctx_after': ctx_after,
                                'ctx_changes': ctx_changes,
                                'return_value': result,
                                'next_node': None,
                                'error': None
                            })
                            return result, execution_trace
                        except Exception as e:
                            error = f"Failed to execute output: {str(e)}"
                            raise ILRExecutionError(
                                f"Failed to execute output of node {current_node_id}: {current_node.output}\nError: {e}"
                            )
                    else:
                        # No output, return None
                        execution_trace.append({
                            'step': iteration,
                            'node_id': current_node_id,
                            'node_type': 'end',
                            'node_label': current_node.label,
                            'action': None,
                            'condition': None,
                            'condition_result': None,
                            'ctx_before': ctx_before,
                            'ctx_after': dict(ctx),
                            'ctx_changes': {},
                            'return_value': None,
                            'next_node': None,
                            'error': None
                        })
                        return None, execution_trace

                else:
                    error = f"Unknown node type: {current_node.type}"
                    raise ILRExecutionError(f"Unknown node type: {current_node.type}")

            except ILRExecutionError:
                # Record error and re-raise
                ctx_after = dict(ctx)
                ctx_changes = {k: (ctx_before.get(k), ctx_after.get(k))
                             for k in set(list(ctx_before.keys()) + list(ctx_after.keys()))
                             if ctx_before.get(k) != ctx_after.get(k)}

                execution_trace.append({
                    'step': iteration,
                    'node_id': current_node_id,
                    'node_type': current_node.type.value,
                    'node_label': current_node.label,
                    'action': action,
                    'condition': condition,
                    'condition_result': condition_result,
                    'ctx_before': ctx_before,
                    'ctx_after': ctx_after,
                    'ctx_changes': ctx_changes,
                    'next_node': next_node,
                    'error': error
                })
                raise

            # Record context after execution
            ctx_after = dict(ctx)
            ctx_changes = {k: (ctx_before.get(k), ctx_after.get(k))
                         for k in set(list(ctx_before.keys()) + list(ctx_after.keys()))
                         if ctx_before.get(k) != ctx_after.get(k)}

            execution_trace.append({
                'step': iteration,
                'node_id': current_node_id,
                'node_type': current_node.type.value,
                'node_label': current_node.label,
                'action': action,
                'condition': condition,
                'condition_result': condition_result,
                'ctx_before': ctx_before,
                'ctx_after': ctx_after,
                'ctx_changes': ctx_changes,
                'next_node': next_node,
                'error': error
            })

            current_node_id = next_node

        raise ILRExecutionError("Execution flow did not reach termination node")

    def execute_with_trace(self, inputs: Dict[str, Any] = None) -> Tuple[Any, List[Dict]]:
        """
        Execute ILR and return detailed trace information

        Args:
            inputs: Input parameter dictionary, will be placed into ctx

        Returns:
            (Execution result, trace information list)
        """
        if not self.nodes:
            raise ILRExecutionError("No ILR to execute, please call parse() method first")

        # Initialize context
        ctx = dict(inputs) if inputs else {}

        # Execute main function and record trace
        return self._execute_nodes_with_trace(self.nodes, ctx, recursion_depth=0)


    def to_python_code(self, function_name: str = "solution",
                       params: List[str] = None) -> str:
        """
        Convert ILR to executable Python function code

        Args:
            function_name: Generated function name
            params: Function parameter list

        Returns:
            Python code string
        """
        if not self.nodes:
            return f"def {function_name}():\n    pass"

        params = params or []
        code_lines = []

        # First generate all helper functions
        for func_name, func in self.functions.items():
            func_code = self._generate_function_code(func)
            code_lines.extend(func_code)
            code_lines.append("")  # Blank line separator

        # Collect all variables used in ctx
        ctx_vars = self._collect_ctx_variables()

        # Detect loop head nodes (for main function)
        self._loop_heads = self._detect_loops_for_nodes(self.nodes)

        # Generate main function signature
        params_str = ', '.join(params) if params else ''
        code_lines.append(f"def {function_name}({params_str}):")

        # Initialize ctx
        code_lines.append("    ctx = {}")

        # Put parameters into ctx
        for param in params:
            code_lines.append(f"    ctx['{param}'] = {param}")

        code_lines.append("")

        # Find start node, begin conversion
        start_node = next(n for n in self.nodes.values() if n.type == NodeType.START)

        # Use graph traversal to generate code
        visited = set()
        generated_code = self._generate_node_code(start_node.next, self.nodes, visited, indent=1)
        code_lines.extend(generated_code)

        return '\n'.join(code_lines)

    def _generate_function_code(self, func: ILRFunction) -> List[str]:
        """Generate Python code for helper function"""
        code_lines = []

        # Generate function signature
        params_str = ', '.join(func.parameters) if func.parameters else ''
        code_lines.append(f"def {func.name}({params_str}):")

        # Initialize ctx
        code_lines.append("    ctx = {}")

        # Put parameters into ctx
        for param in func.parameters:
            code_lines.append(f"    ctx['{param}'] = {param}")

        code_lines.append("")

        # Detect loop head nodes (for this function)
        self._loop_heads = self._detect_loops_for_nodes(func.nodes)

        # Find start node
        start_node = next((n for n in func.nodes.values() if n.type == NodeType.START), None)
        if start_node is None:
            code_lines.append("    pass")
            return code_lines

        # Use graph traversal to generate code
        visited = set()
        generated_code = self._generate_node_code(start_node.next, func.nodes, visited, indent=1)
        if generated_code:
            code_lines.extend(generated_code)
        else:
            code_lines.append("    pass")

        return code_lines

    def _detect_loops_for_nodes(self, nodes: Dict[int, ILRNode]) -> Dict[int, Dict]:
        """Detect loop heads in the specified node set"""
        loop_heads = {}

        for node_id, node in nodes.items():
            if node.type != NodeType.DECISION:
                continue

            true_loops_back = self._path_leads_back_in_nodes(node.true_next, node_id, set(), nodes)
            false_loops_back = self._path_leads_back_in_nodes(node.false_next, node_id, set(), nodes)

            if true_loops_back and not false_loops_back:
                loop_heads[node_id] = {
                    'loop_branch': 'true',
                    'exit_branch': 'false'
                }
            elif false_loops_back and not true_loops_back:
                loop_heads[node_id] = {
                    'loop_branch': 'false',
                    'exit_branch': 'true'
                }

        return loop_heads

    def _path_leads_back_in_nodes(self, start_id: Optional[int], target_id: int,
                                   visited: set, nodes: Dict[int, ILRNode]) -> bool:
        """Check if target node can be reached back in the specified node set"""
        if start_id is None:
            return False

        if start_id == target_id:
            return True

        if start_id in visited:
            return False

        visited.add(start_id)

        node = nodes.get(start_id)
        if node is None:
            return False

        if node.type == NodeType.PROCESS:
            return self._path_leads_back_in_nodes(node.next, target_id, visited, nodes)
        elif node.type == NodeType.DECISION:
            return (self._path_leads_back_in_nodes(node.true_next, target_id, visited.copy(), nodes) or
                    self._path_leads_back_in_nodes(node.false_next, target_id, visited.copy(), nodes))
        elif node.type == NodeType.CALL:
            return self._path_leads_back_in_nodes(node.next, target_id, visited, nodes)
        elif node.type in (NodeType.END, NodeType.RETURN):
            return False
        elif node.type == NodeType.START:
            return self._path_leads_back_in_nodes(node.next, target_id, visited, nodes)

        return False

    def _detect_loops(self) -> Dict[int, Dict]:
        """
        Detect which decision nodes are loop heads

        Use back-edge detection to identify loop heads.
        A node is a loop head if and only if:
        1. It is a decision type
        2. One of its branches can eventually loop back to itself, while the other cannot

        Returns:
            Dictionary with loop head node IDs as keys and loop info as values
        """
        loop_heads = {}

        # Analyze each decision node
        for node_id, node in self.nodes.items():
            if node.type != NodeType.DECISION:
                continue

            # Check which branch can loop back to the current node
            true_loops_back = self._path_leads_back(node.true_next, node_id, set())
            false_loops_back = self._path_leads_back(node.false_next, node_id, set())

            # When only one branch loops back to itself, it's a loop head
            if true_loops_back and not false_loops_back:
                loop_heads[node_id] = {
                    'loop_branch': 'true',
                    'exit_branch': 'false'
                }
            elif false_loops_back and not true_loops_back:
                loop_heads[node_id] = {
                    'loop_branch': 'false',
                    'exit_branch': 'true'
                }
            # When both branches loop back to itself, not considered a loop head (may be a conditional node inside loop body)
            # Skip this case

        return loop_heads

    def _shortest_path_back(self, start_id: Optional[int], target_id: int,
                            max_depth: int = 100) -> int:
        """
        Calculate shortest path length from start_id back to target_id

        Use BFS to find shortest path
        """
        if start_id is None:
            return float('inf')

        from collections import deque
        queue = deque([(start_id, 0)])
        visited = set()

        while queue:
            current_id, depth = queue.popleft()

            if depth > max_depth:
                return float('inf')

            if current_id == target_id:
                return depth

            if current_id in visited:
                continue

            visited.add(current_id)

            node = self.nodes.get(current_id)
            if node is None:
                continue

            # Add successor nodes
            if node.type == NodeType.PROCESS:
                if node.next is not None:
                    queue.append((node.next, depth + 1))
            elif node.type == NodeType.DECISION:
                if node.true_next is not None:
                    queue.append((node.true_next, depth + 1))
                if node.false_next is not None:
                    queue.append((node.false_next, depth + 1))
            elif node.type == NodeType.START:
                if node.next is not None:
                    queue.append((node.next, depth + 1))
            # END nodes have no successors

        return float('inf')

    def _path_reaches_end(self, start_id: Optional[int], avoid_id: int, visited: set) -> bool:
        """
        Check if end node can be reached from start_id (without going through avoid_id)
        """
        if start_id is None or start_id == avoid_id:
            return False

        if start_id in visited:
            return False

        visited.add(start_id)

        node = self.nodes.get(start_id)
        if node is None:
            return False

        if node.type == NodeType.END:
            return True

        if node.type == NodeType.PROCESS:
            return self._path_reaches_end(node.next, avoid_id, visited)
        elif node.type == NodeType.DECISION:
            return (self._path_reaches_end(node.true_next, avoid_id, visited.copy()) or
                    self._path_reaches_end(node.false_next, avoid_id, visited.copy()))
        elif node.type == NodeType.START:
            return self._path_reaches_end(node.next, avoid_id, visited)

        return False

    def _path_leads_back(self, start_id: Optional[int], target_id: int,
                         visited: set) -> bool:
        """
        Check if start_id can reach back to target_id (back-edge detection)

        Args:
            start_id: Starting node ID
            target_id: Target node ID (the node to check if reachable)
            visited: Set of visited nodes

        Returns:
            Whether a path exists from start_id to target_id
        """
        if start_id is None:
            return False

        if start_id == target_id:
            return True

        if start_id in visited:
            return False

        visited.add(start_id)

        node = self.nodes.get(start_id)
        if node is None:
            return False

        # Check successor nodes based on node type
        if node.type == NodeType.PROCESS:
            return self._path_leads_back(node.next, target_id, visited)
        elif node.type == NodeType.DECISION:
            return (self._path_leads_back(node.true_next, target_id, visited.copy()) or
                    self._path_leads_back(node.false_next, target_id, visited.copy()))
        elif node.type == NodeType.END:
            return False
        elif node.type == NodeType.START:
            return self._path_leads_back(node.next, target_id, visited)

        return False

    def _collect_ctx_variables(self) -> set:
        """Collect all ctx variables used in ILR"""
        variables = set()
        pattern = r"ctx\[['\"]([\w]+)['\"]\]"

        for node in self.nodes.values():
            for field in [node.action, node.logic, node.output, node.value]:
                if field:
                    matches = re.findall(pattern, field)
                    variables.update(matches)

        # Also collect variables in functions
        for func in self.functions.values():
            for node in func.nodes.values():
                for field in [node.action, node.logic, node.output, node.value]:
                    if field:
                        matches = re.findall(pattern, field)
                        variables.update(matches)

        return variables

    def _generate_node_code(self, node_id: Optional[int], nodes: Dict[int, ILRNode],
                            visited: set, indent: int = 0,
                            loop_stack: List[int] = None) -> List[str]:
        """
        Recursively generate Python code for nodes

        Args:
            node_id: Current node ID
            nodes: Node dictionary
            visited: Set of visited nodes
            indent: Indentation level
            loop_stack: Current nested loop head ID stack
        """
        if node_id is None:
            return []

        if loop_stack is None:
            loop_stack = []

        # If returned to any loop head, stop generation (loop continues automatically)
        if node_id in loop_stack:
            return []

        # Normal visited check
        if node_id in visited:
            return []

        node = nodes.get(node_id)
        if node is None:
            return []

        visited.add(node_id)
        ind = "    " * indent
        lines = []

        # Add node label as comment
        if node.label:
            lines.append(f"{ind}# {node.label}")

        if node.type == NodeType.PROCESS:
            if node.action:
                lines.append(f"{ind}{node.action}")
            lines.extend(self._generate_node_code(
                node.next, nodes, visited, indent, loop_stack))

        elif node.type == NodeType.DECISION:
            # Check if it's a loop head
            loop_info = getattr(self, '_loop_heads', {}).get(node_id)

            if loop_info:
                # This is a loop head node, generate while loop
                new_loop_stack = loop_stack + [node_id]

                if loop_info['loop_branch'] == 'true':
                    # Continue loop when condition is true
                    lines.append(f"{ind}while {node.logic}:")
                    # Generate loop body (true branch)
                    loop_body = self._generate_node_code(
                        node.true_next, nodes, visited.copy(), indent + 1, new_loop_stack)
                    if loop_body:
                        lines.extend(loop_body)
                    else:
                        lines.append(f"{ind}    pass")
                    # Continue generating code after loop (false branch, executed after loop exits)
                    exit_code = self._generate_node_code(
                        node.false_next, nodes, visited, indent, loop_stack)
                    lines.extend(exit_code)
                else:
                    # Continue loop when condition is false (need to negate condition)
                    lines.append(f"{ind}while not ({node.logic}):")
                    # Generate loop body (false branch)
                    loop_body = self._generate_node_code(
                        node.false_next, nodes, visited.copy(), indent + 1, new_loop_stack)
                    if loop_body:
                        lines.extend(loop_body)
                    else:
                        lines.append(f"{ind}    pass")
                    # Continue generating code after loop (true branch, executed after loop exits)
                    exit_code = self._generate_node_code(
                        node.true_next, nodes, visited, indent, loop_stack)
                    lines.extend(exit_code)
            else:
                # Normal conditional branch, generate if-else
                if node.logic:
                    lines.append(f"{ind}if {node.logic}:")
                    # True branch
                    true_code = self._generate_node_code(
                        node.true_next, nodes, visited.copy(), indent + 1, loop_stack)
                    if true_code:
                        lines.extend(true_code)
                    else:
                        lines.append(f"{ind}    pass")
                    # False branch
                    lines.append(f"{ind}else:")
                    false_code = self._generate_node_code(
                        node.false_next, nodes, visited.copy(), indent + 1, loop_stack)
                    if false_code:
                        lines.extend(false_code)
                    else:
                        lines.append(f"{ind}    pass")

        elif node.type == NodeType.CALL:
            # Generate function call code
            func_name = node.function_name
            args_str = ', '.join(node.arguments or [])
            if node.return_var:
                lines.append(f"{ind}ctx['{node.return_var}'] = {func_name}({args_str})")
            else:
                lines.append(f"{ind}{func_name}({args_str})")
            lines.extend(self._generate_node_code(
                node.next, nodes, visited, indent, loop_stack))

        elif node.type == NodeType.RETURN:
            # Generate return statement
            if node.value:
                lines.append(f"{ind}return {node.value}")
            else:
                lines.append(f"{ind}return None")

        elif node.type == NodeType.END:
            if node.output:
                lines.append(f"{ind}return {node.output}")
            else:
                lines.append(f"{ind}return None")

        return lines


class ILRDiagnostic:
    """ILR diagnostic tool - compares ILR interpretation execution and generated code execution results"""

    def __init__(self):
        self.interpreter = ILRInterpreter()
        self.tester = ILRTester()

    def diagnose(self, ilr_json: Union[str, Dict],
                 generated_code: str,
                 problem_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose issue source: ILR logic issue vs code generation issue

        Args:
            ilr_json: ILR JSON
            generated_code: Generated Python code
            problem_info: Problem info, containing test cases

        Returns:
            Diagnosis results
        """
        from code_tester import CodeTester

        test_cases = problem_info.get('test_cases', [])
        entry_point = problem_info.get('entry_point')

        # 1. Test ILR interpretation execution
        ilr_result = self.tester.test_ilr(ilr_json, test_cases, entry_point)

        # 2. Test generated code
        code_tester = CodeTester()
        code_result = code_tester.test_generated_code(generated_code, problem_info)

        # 3. Comparative analysis
        diagnosis = self._analyze_results(ilr_result, code_result)

        return {
            'ilr_test': ilr_result,
            'code_test': code_result,
            'diagnosis': diagnosis
        }

    def _analyze_results(self, ilr_result: Dict, code_result: Dict) -> Dict:
        """Analyze test results, determine issue source"""
        diagnosis = {
            'issue_type': None,
            'details': [],
            'recommendations': []
        }

        ilr_passed = ilr_result.get('passed', 0)
        ilr_total = ilr_result.get('total', 0)
        code_passed = code_result.get('passed', 0)
        code_total = code_result.get('total', 0)

        ilr_success = ilr_passed == ilr_total and ilr_total > 0
        code_success = code_passed == code_total and code_total > 0

        if ilr_success and code_success:
            diagnosis['issue_type'] = 'NO_ISSUE'
            diagnosis['details'].append('ILR and generated code both passed all tests')

        elif ilr_success and not code_success:
            diagnosis['issue_type'] = 'CODE_GENERATION_ERROR'
            diagnosis['details'].append(f'ILR passed all tests ({ilr_passed}/{ilr_total})')
            diagnosis['details'].append(f'Generated code failed ({code_passed}/{code_total})')
            diagnosis['recommendations'].append(
                'The issue is in the code generation phase - model did not correctly translate ILR to code'
            )
            diagnosis['recommendations'].append(
                'Consider improving the code generation prompt'
            )

        elif not ilr_success and not code_success:
            diagnosis['issue_type'] = 'ILR_LOGIC_ERROR'
            diagnosis['details'].append(f'ILR failed tests ({ilr_passed}/{ilr_total})')
            diagnosis['details'].append(f'Generated code also failed ({code_passed}/{code_total})')
            diagnosis['recommendations'].append(
                'The issue is in the ILR generation phase - the flowchart logic was not correctly extracted'
            )
            diagnosis['recommendations'].append(
                'Consider improving the vision model or ILR extraction prompt'
            )

        elif not ilr_success and code_success:
            diagnosis['issue_type'] = 'CODE_GENERATOR_FIXED'
            diagnosis['details'].append(f'ILR failed tests ({ilr_passed}/{ilr_total})')
            diagnosis['details'].append(f'Generated code passed all tests ({code_passed}/{code_total})')
            diagnosis['recommendations'].append(
                'The code generator corrected the logic errors in the ILR'
            )
            diagnosis['recommendations'].append(
                'The ILR extraction could still be improved for consistency'
            )

        else:
            diagnosis['issue_type'] = 'UNKNOWN'
            diagnosis['details'].append('Unable to determine issue source')

        return diagnosis

    def print_report(self, result: Dict) -> None:
        """Print diagnostic report"""
        print("=" * 60)
        print("ILR Diagnostic Report")
        print("=" * 60)

        ilr_test = result.get('ilr_test', {})
        code_test = result.get('code_test', {})
        diagnosis = result.get('diagnosis', {})

        print(f"\n[ILR Interpretation Test]")
        print(f"  Passed: {ilr_test.get('passed', 0)}/{ilr_test.get('total', 0)}")
        print(f"  Pass Rate: {ilr_test.get('pass_rate', 0):.1f}%")
        if ilr_test.get('error'):
            print(f"  Error: {ilr_test['error']}")

        print(f"\n[Generated Code Test]")
        print(f"  Passed: {code_test.get('passed', 0)}/{code_test.get('total', 0)}")
        print(f"  Pass Rate: {code_test.get('pass_rate', 0):.1f}%")
        if code_test.get('error'):
            print(f"  Error: {code_test['error']}")

        print(f"\n[Diagnosis]")
        print(f"  Issue Type: {diagnosis.get('issue_type', 'UNKNOWN')}")

        if diagnosis.get('details'):
            print(f"\n  Details:")
            for detail in diagnosis['details']:
                print(f"    - {detail}")

        if diagnosis.get('recommendations'):
            print(f"\n  Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"    - {rec}")

        print("\n" + "=" * 60)


class ILRTester:
    """ILR tester - uses test cases to verify ILR logic"""

    def __init__(self):
        self.interpreter = ILRInterpreter()

    def test_ilr(self, ilr_json: Union[str, Dict],
                 test_cases: List[Dict[str, Any]],
                 entry_point: str = None) -> Dict[str, Any]:
        """
        Test if ILR can correctly execute all test cases

        Args:
            ilr_json: ILR JSON
            test_cases: Test case list, each case contains input and output
            entry_point: Entry point function name (used to parse input parameters)

        Returns:
            Test result dictionary
        """
        results = []
        passed_count = 0
        total = len(test_cases)

        try:
            self.interpreter.parse(ilr_json)
        except ILRParseError as e:
            return {
                'success': False,
                'passed': 0,
                'total': total,
                'pass_rate': 0,
                'error': f"ILR parsing failed: {e}",
                'results': []
            }

        for i, test_case in enumerate(test_cases, 1):
            result = self._run_single_test(test_case, i, entry_point)
            results.append(result)
            if result['passed']:
                passed_count += 1

        return {
            'success': True,
            'passed': passed_count,
            'total': total,
            'pass_rate': passed_count / total * 100 if total > 0 else 0,
            'error': None,
            'results': results
        }

    def test_lcb(self, ilr_json: Union[str, Dict],
                 public_test_cases: List[Dict[str, Any]],
                 entry_point: str = "solve") -> Dict[str, Any]:
        """Test ILR with parameterized LCB solve(...) cases without executing stdin."""
        normalized_cases = []
        for case in public_test_cases or []:
            if not isinstance(case, dict):
                continue
            normalized_cases.append({
                "input": case.get("input", ""),
                "output": case.get("output"),
            })
        return self.test_ilr(ilr_json, normalized_cases, entry_point)
    def _run_single_test(self, test_case: Dict, test_num: int,
                         entry_point: str = None) -> Dict[str, Any]:
        """Run a single test"""
        input_data = test_case.get('input', {})
        expected_output = test_case.get('output')

        try:
            # Parse input
            ctx_inputs = self._parse_input(input_data, entry_point)

            # Execute ILR
            actual_output = self.interpreter.execute(ctx_inputs)

            # Compare output
            passed = self._compare_output(actual_output, expected_output)

            return {
                'test_num': test_num,
                'input': input_data,
                'expected_output': expected_output,
                'actual_output': actual_output,
                'passed': passed,
                'error': None if passed else f"Output mismatch: expected {expected_output}, actual {actual_output}"
            }

        except ILRExecutionError as e:
            return {
                'test_num': test_num,
                'input': input_data,
                'expected_output': expected_output,
                'actual_output': None,
                'passed': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'test_num': test_num,
                'input': input_data,
                'expected_output': expected_output,
                'actual_output': None,
                'passed': False,
                'error': f"Execution error: {e}"
            }

    def _parse_input(self, input_data: Any, entry_point: str = None) -> Dict[str, Any]:
        """Parse input data into ctx dictionary"""
        if isinstance(input_data, dict):
            # If it's a dictionary, check for args key
            if 'args' in input_data:
                # HumanEval format
                args = input_data['args']
                if isinstance(args, list):
                    # If it's a list, create positional args ctx
                    return {f'arg{i}': v for i, v in enumerate(args)}
                elif isinstance(args, tuple):
                    # If it's a tuple, create positional args ctx
                    return {f'arg{i}': v for i, v in enumerate(args)}
                elif isinstance(args, dict):
                    return args
                elif isinstance(args, str):
                    # If it's a string, try to parse as Python expression (supports operators like 2**31)
                    try:
                        import ast
                        # First try ast.literal_eval
                        try:
                            # Try to parse as tuple (multiple arguments)
                            parsed_args = ast.literal_eval(f'({args},)')
                            if isinstance(parsed_args, tuple):
                                return {f'arg{i}': v for i, v in enumerate(parsed_args)}
                        except:
                            pass
                        # If that fails, try eval to handle expressions (like 2**31)
                        try:
                            parsed_args = eval(f'({args},)', {'__builtins__': {}}, {})
                            if isinstance(parsed_args, tuple):
                                return {f'arg{i}': v for i, v in enumerate(parsed_args)}
                        except:
                            pass
                    except:
                        pass
                    # If all fail, return original string
                    return {'arg0': args}
                else:
                    # Other types, treat as single argument
                    return {'arg0': args}
            return input_data

        elif isinstance(input_data, str):
            # Try to parse string expression
            # For example: "function_name(arg1, arg2)" or direct value
            try:
                # Check if it's a function call format
                match = re.match(r'(\w+)\((.*)\)', input_data.strip())
                if match:
                    func_name = match.group(1)
                    args_str = match.group(2)

                    if args_str.strip():
                        # Parse arguments
                        try:
                            # Use ast.literal_eval to safely parse arguments
                            import ast
                            # Wrap as tuple to handle multiple arguments
                            args = ast.literal_eval(f"({args_str},)")
                            return {f'arg{i}': v for i, v in enumerate(args)}
                        except:
                            # Try using eval to handle expressions (like 2**31)
                            try:
                                args = eval(f"({args_str},)", {'__builtins__': {}}, {})
                                return {f'arg{i}': v for i, v in enumerate(args)}
                            except:
                                return {'input': input_data}
                    else:
                        return {}
                else:
                    # Not a function call, try to parse as literal value
                    import ast
                    try:
                        value = ast.literal_eval(input_data)
                        return {'input': value}
                    except:
                        # Try eval to handle expressions
                        try:
                            value = eval(input_data, {'__builtins__': {}}, {})
                            return {'input': value}
                        except:
                            return {'input': input_data}
            except:
                return {'input': input_data}

        elif isinstance(input_data, (list, tuple)):
            return {f'arg{i}': v for i, v in enumerate(input_data)}

        else:
            return {'input': input_data}

    def _compare_output(self, actual: Any, expected: Any) -> bool:
        """Compare actual output with expected output"""
        # Type conversion attempt
        if isinstance(expected, str):
            try:
                import ast
                expected = ast.literal_eval(expected)
            except:
                pass

        # Direct comparison
        if actual == expected:
            return True

        # Numeric comparison (handles floating point precision)
        try:
            if abs(float(actual) - float(expected)) < 1e-9:
                return True
        except:
            pass

        # String comparison
        if str(actual).strip() == str(expected).strip():
            return True

        return False


def generate_executable_code_from_ilr(ilr_json: Union[str, Dict],
                                       function_name: str = "solution",
                                       params: List[str] = None) -> str:
    """
    Convenience function: generate executable Python code from ILR

    Args:
        ilr_json: ILR JSON
        function_name: Function name
        params: Parameter list

    Returns:
        Python code string
    """
    interpreter = ILRInterpreter()
    interpreter.parse(ilr_json)
    return interpreter.to_python_code(function_name, params)


# ============ Test code ============

def test_ilr_interpreter():
    """Test ILR interpreter"""

    # Test ILR: Count vowels
    test_ilr = '''
{
  "problem_id": "count_vowels",
  "nodes": [
    {
      "id": 1,
      "type": "start",
      "label": "Start",
      "next": 2
    },
    {
      "id": 2,
      "type": "process",
      "label": "Initialize count and index",
      "action": "ctx['count'] = 0; ctx['i'] = 0",
      "next": 3
    },
    {
      "id": 3,
      "type": "decision",
      "label": "More characters?",
      "logic": "ctx['i'] < len(ctx['word'])",
      "true_next": 4,
      "false_next": 7
    },
    {
      "id": 4,
      "type": "decision",
      "label": "Is vowel?",
      "logic": "ctx['word'][ctx['i']].lower() in 'aeiou'",
      "true_next": 5,
      "false_next": 6
    },
    {
      "id": 5,
      "type": "process",
      "label": "Increment count",
      "action": "ctx['count'] += 1",
      "next": 6
    },
    {
      "id": 6,
      "type": "process",
      "label": "Increment index",
      "action": "ctx['i'] += 1",
      "next": 3
    },
    {
      "id": 7,
      "type": "end",
      "label": "Return count",
      "output": "ctx['count']"
    }
  ]
}
'''

    print("=" * 60)
    print("ILR Interpreter Test")
    print("=" * 60)

    # Create interpreter
    interpreter = ILRInterpreter()

    # Parse ILR
    print("\n1. Parsing ILR...")
    try:
        interpreter.parse(test_ilr)
        print(f"   Parse successful! Problem ID: {interpreter.problem_id}")
        print(f"   Node count: {len(interpreter.nodes)}")
    except ILRParseError as e:
        print(f"   Parse failed: {e}")
        return

    # Execute tests
    print("\n2. Running tests...")
    test_cases = [
        {'input': {'word': 'hello'}, 'expected': 2},
        {'input': {'word': 'world'}, 'expected': 1},
        {'input': {'word': 'aeiou'}, 'expected': 5},
        {'input': {'word': 'xyz'}, 'expected': 0},
        {'input': {'word': 'HELLO'}, 'expected': 2},
    ]

    for i, case in enumerate(test_cases, 1):
        result = interpreter.execute(case['input'])
        status = "PASS" if result == case['expected'] else "FAIL"
        print(f"   Test{i}: {case['input']['word']} -> {result} (expected: {case['expected']}) [{status}]")

    # Generate Python code
    print("\n3. Generating Python code...")
    python_code = interpreter.to_python_code("count_vowels", ["word"])
    print(python_code)

    # Verify generated code
    print("\n4. Verifying generated code...")
    try:
        exec_globals = {}
        exec(python_code, exec_globals)
        count_vowels = exec_globals['count_vowels']

        for case in test_cases:
            result = count_vowels(case['input']['word'])
            status = "PASS" if result == case['expected'] else "FAIL"
            print(f"   {case['input']['word']} -> {result} [{status}]")
    except Exception as e:
        print(f"   Code execution failed: {e}")


def test_ilr_tester():
    """Test ILR tester"""

    # Simple addition ILR
    add_ilr = '''
{
  "problem_id": "add_numbers",
  "nodes": [
    {"id": 1, "type": "start", "label": "Start", "next": 2},
    {"id": 2, "type": "process", "label": "Add", "action": "ctx['result'] = ctx['a'] + ctx['b']", "next": 3},
    {"id": 3, "type": "end", "label": "Return", "output": "ctx['result']"}
  ]
}
'''

    test_cases = [
        {'input': {'a': 2, 'b': 3}, 'output': 5},
        {'input': {'a': -1, 'b': 1}, 'output': 0},
        {'input': {'a': 0, 'b': 0}, 'output': 0},
        {'input': {'a': 100, 'b': 200}, 'output': 300},
    ]

    print("\n" + "=" * 60)
    print("ILR Tester Test")
    print("=" * 60)

    tester = ILRTester()
    result = tester.test_ilr(add_ilr, test_cases)

    print(f"\nTest Results:")
    print(f"  Success: {result['success']}")
    print(f"  Passed: {result['passed']}/{result['total']}")
    print(f"  Pass Rate: {result['pass_rate']:.1f}%")

    if result['error']:
        print(f"  Error: {result['error']}")

    print("\nDetailed Results:")
    for r in result['results']:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  Test{r['test_num']}: [{status}]")
        if r['error']:
            print(f"    Error: {r['error']}")


def test_complex_ilr():
    """Test complex ILR (with multi-level conditional branching)"""

    # ILR for determining number sign
    sign_ilr = '''
{
  "problem_id": "number_sign",
  "nodes": [
    {"id": 1, "type": "start", "label": "Start", "next": 2},
    {"id": 2, "type": "decision", "label": "Is positive?", "logic": "ctx['n'] > 0", "true_next": 3, "false_next": 4},
    {"id": 3, "type": "end", "label": "Return positive", "output": "'positive'"},
    {"id": 4, "type": "decision", "label": "Is negative?", "logic": "ctx['n'] < 0", "true_next": 5, "false_next": 6},
    {"id": 5, "type": "end", "label": "Return negative", "output": "'negative'"},
    {"id": 6, "type": "end", "label": "Return zero", "output": "'zero'"}
  ]
}
'''

    print("\n" + "=" * 60)
    print("Complex ILR Test (Multi-conditional Branching)")
    print("=" * 60)

    interpreter = ILRInterpreter()
    interpreter.parse(sign_ilr)

    test_cases = [
        ({'n': 5}, 'positive'),
        ({'n': -3}, 'negative'),
        ({'n': 0}, 'zero'),
    ]

    for inputs, expected in test_cases:
        result = interpreter.execute(inputs)
        status = "PASS" if result == expected else "FAIL"
        print(f"  n={inputs['n']} -> {result} (expected: {expected}) [{status}]")

    # Generate code
    print("\nGenerated Python code:")
    print(interpreter.to_python_code("get_sign", ["n"]))


def test_recursive_ilr():
    """Test recursive ILR (with function calls and recursion)"""

    # Recursive factorial ILR
    factorial_ilr = '''
{
  "problem_id": "factorial_recursive",
  "functions": [{
    "name": "factorial",
    "parameters": ["n"],
    "entry_node": 1,
    "nodes": [
      {"id": 1, "type": "start", "label": "Start", "bbox": [100, 450, 150, 550], "next": 2},
      {"id": 2, "type": "decision", "label": "n <= 1?", "bbox": [200, 400, 300, 600], "logic": "ctx['n'] <= 1", "true_next": 3, "false_next": 4},
      {"id": 3, "type": "return", "label": "Return 1", "bbox": [220, 650, 280, 750], "value": "1"},
      {"id": 4, "type": "call", "label": "Recursive call", "bbox": [350, 400, 450, 600], "function_name": "factorial", "arguments": ["ctx['n'] - 1"], "return_var": "sub", "next": 5},
      {"id": 5, "type": "return", "label": "Return n*sub", "bbox": [500, 400, 550, 600], "value": "ctx['n'] * ctx['sub']"}
    ]
  }],
  "nodes": [
    {"id": 1, "type": "start", "label": "Start", "bbox": [50, 50, 100, 150], "next": 2},
    {"id": 2, "type": "process", "label": "Get input", "bbox": [150, 50, 200, 150], "action": "ctx['input'] = ctx['n']", "next": 3},
    {"id": 3, "type": "call", "label": "Call factorial", "bbox": [250, 50, 300, 150], "function_name": "factorial", "arguments": ["ctx['input']"], "return_var": "result", "next": 4},
    {"id": 4, "type": "end", "label": "End", "bbox": [350, 50, 400, 150], "output": "ctx['result']"}
  ]
}
'''

    print("\n" + "=" * 60)
    print("Recursive ILR Test (Factorial Function)")
    print("=" * 60)

    interpreter = ILRInterpreter()
    interpreter.parse(factorial_ilr)

    print(f"\nParse successful!")
    print(f"Problem ID: {interpreter.problem_id}")
    print(f"Main function node count: {len(interpreter.nodes)}")
    print(f"Helper function count: {len(interpreter.functions)}")

    test_cases = [
        ({'n': 1}, 1),
        ({'n': 2}, 2),
        ({'n': 3}, 6),
        ({'n': 5}, 120),
        ({'n': 7}, 5040),
    ]

    print("\n1. Testing ILR interpretation execution...")
    for inputs, expected in test_cases:
        result = interpreter.execute(inputs)
        status = "PASS" if result == expected else "FAIL"
        print(f"  factorial({inputs['n']}) -> {result} (expected: {expected}) [{status}]")

    # Generate code
    print("\n2. Generated Python code:")
    python_code = interpreter.to_python_code("main", ["n"])
    print(python_code)

    # Verify generated code
    print("\n3. Verifying generated code...")
    try:
        exec_globals = {}
        exec(python_code, exec_globals)
        main_func = exec_globals['main']

        for inputs, expected in test_cases:
            result = main_func(inputs['n'])
            status = "PASS" if result == expected else "FAIL"
            print(f"  factorial({inputs['n']}) -> {result} [{status}]")
    except Exception as e:
        print(f"  Code execution failed: {e}")
        import traceback
        traceback.print_exc()


def test_helper_function_ilr():
    """Test ILR with helper functions"""

    # Sum ILR with helper function
    sum_ilr = '''
{
  "problem_id": "sum_with_helper",
  "functions": [{
    "name": "compute_sum",
    "parameters": ["arr"],
    "entry_node": 1,
    "nodes": [
      {"id": 1, "type": "start", "label": "Start", "bbox": [100, 100, 150, 200], "next": 2},
      {"id": 2, "type": "process", "label": "Init sum", "bbox": [200, 100, 250, 200], "action": "ctx['total'] = 0; ctx['i'] = 0", "next": 3},
      {"id": 3, "type": "decision", "label": "i < len(arr)?", "bbox": [300, 100, 400, 200], "logic": "ctx['i'] < len(ctx['arr'])", "true_next": 4, "false_next": 6},
      {"id": 4, "type": "process", "label": "Add element", "bbox": [350, 250, 400, 350], "action": "ctx['total'] += ctx['arr'][ctx['i']]", "next": 5},
      {"id": 5, "type": "process", "label": "Increment i", "bbox": [350, 400, 400, 500], "action": "ctx['i'] += 1", "next": 3},
      {"id": 6, "type": "return", "label": "Return total", "bbox": [500, 100, 550, 200], "value": "ctx['total']"}
    ]
  }],
  "nodes": [
    {"id": 1, "type": "start", "label": "Start", "bbox": [50, 50, 100, 150], "next": 2},
    {"id": 2, "type": "call", "label": "Call sum", "bbox": [150, 50, 200, 150], "function_name": "compute_sum", "arguments": ["ctx['data']"], "return_var": "sum_result", "next": 3},
    {"id": 3, "type": "end", "label": "End", "bbox": [250, 50, 300, 150], "output": "ctx['sum_result']"}
  ]
}
'''

    print("\n" + "=" * 60)
    print("Helper Function ILR Test (Sum Function)")
    print("=" * 60)

    interpreter = ILRInterpreter()
    interpreter.parse(sum_ilr)

    print(f"\nParse successful!")
    print(f"Problem ID: {interpreter.problem_id}")
    print(f"Main function node count: {len(interpreter.nodes)}")
    print(f"Helper function count: {len(interpreter.functions)}")

    test_cases = [
        ({'data': [1, 2, 3]}, 6),
        ({'data': [10, 20, 30, 40]}, 100),
        ({'data': []}, 0),
        ({'data': [5]}, 5),
    ]

    print("\n1. Testing ILR interpretation execution...")
    for inputs, expected in test_cases:
        result = interpreter.execute(inputs)
        status = "PASS" if result == expected else "FAIL"
        print(f"  sum({inputs['data']}) -> {result} (expected: {expected}) [{status}]")

    # Generate code
    print("\n2. Generated Python code:")
    python_code = interpreter.to_python_code("main", ["data"])
    print(python_code)

    # Verify generated code
    print("\n3. Verifying generated code...")
    try:
        exec_globals = {}
        exec(python_code, exec_globals)
        main_func = exec_globals['main']

        for inputs, expected in test_cases:
            result = main_func(inputs['data'])
            status = "PASS" if result == expected else "FAIL"
            print(f"  sum({inputs['data']}) -> {result} [{status}]")
    except Exception as e:
        print(f"  Code execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ilr_interpreter()
    test_ilr_tester()
    test_complex_ilr()
    test_recursive_ilr()
    test_helper_function_ilr()


def test_ilr_with_trace():
    """Test ILR interpreter trace functionality"""
    count_ilr = '''
{
  "problem_id": "count_to_n",
  "nodes": [
    {"id": 1, "type": "start", "label": "Start", "next": 2},
    {"id": 2, "type": "process", "label": "Initialize", "action": "ctx['n']=ctx['arg0']; ctx['i']=0; ctx['sum']=0", "next": 3},
    {"id": 3, "type": "decision", "label": "i < n?", "logic": "ctx['i'] < ctx['n']", "true_next": 4, "false_next": 6},
    {"id": 4, "type": "process", "label": "Add i to sum", "action": "ctx['sum'] += ctx['i']", "next": 5},
    {"id": 5, "type": "process", "label": "Increment i", "action": "ctx['i'] += 1", "next": 3},
    {"id": 6, "type": "end", "label": "Return sum", "output": "ctx['sum']"}
  ]
}
'''
    print("=" * 70)
    print("ILR Interpreter Trace Functionality Test")
    print("=" * 70)
    interpreter = ILRInterpreter()
    interpreter.parse(count_ilr)
    result, trace = interpreter.execute_with_trace({'arg0': 5})
    print(f"\nExecution result: {result}")
    print(f"\nExecution step count: {len(trace)}")
    print("\nDetailed execution trace:")
    print("-" * 70)
    for step_info in trace:
        print(f"\nStep {step_info['step']}: Node {step_info['node_id']} ({step_info['node_type']})")
        print(f"  Label: {step_info['node_label']}")
        if step_info['action']:
            print(f"  Action: {step_info['action']}")
        if step_info['condition'] is not None:
            print(f"  Condition: {step_info['condition']}")
            print(f"  Result: {step_info['condition_result']}")
        if step_info['ctx_changes']:
            print(f"  Variable Changes:")
            for var, (old_val, new_val) in step_info['ctx_changes'].items():
                print(f"    {var}: {old_val} -> {new_val}")
        if step_info.get('return_value') is not None:
            print(f"  Return Value: {step_info['return_value']}")
        if step_info['error']:
            print(f"  ERROR: {step_info['error']}")
        print(f"  Next Node: {step_info['next_node']}")
    print("\n" + "=" * 70)
