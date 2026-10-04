"""
Main Program - Enhanced Version
Integrates LangGraph multi-agent workflow, supports single flowchart processing and batch processing
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read jsonl file"""
    results = []
    if not os.path.exists(file_path):
        return results
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except:
                continue
    return results


def write_jsonl_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Write jsonl file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl_file(file_path: str, item: Dict[str, Any]) -> None:
    """Append one record to jsonl file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def str2bool(value):
    """Parse command line boolean argument, supports true/false."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")

from agents.langgraph_agent import LangGraphFlowchartAgent
from tools.problem_extractor import ProblemExtractor
from utils.intermediate_representation import (
    get_intermediate_representation_display_name,
    normalize_intermediate_representation_type,
)


class FlowchartProcessor:
    """Flowchart Processor Main Class - Enhanced version, supports LangGraph multi-agent workflow and batch processing"""

    def __init__(self, data_root: str = None,
                 vision_model_2_config: Dict[str, Any] = None,
                 language_model_1_config: Dict[str, Any] = None,
                 output_dir: str = None,
                 enable_evaluation: bool = True,
                 use_reflection: bool = False,
                 reflection_model_config: Dict[str, Any] = None,
                 max_iterations: int = 3,
                 intermediate_representation_type: str = "ilr",
                 prompt_variant: str = "default",
                 include_problem_text_description: bool = True):
        """
        Initialize processor

        Args:
            data_root: Data root directory
            vision_model_2_config: Vision model 2 configuration (for ILR generation)
            language_model_1_config: Language model 1 configuration (for code generation)
            output_dir: Output directory, if None uses default output directory
            enable_evaluation: Whether to enable evaluation
            use_reflection: Whether to use reflection functionality
            reflection_model_config: Reflection model configuration (only needed when use_reflection=True)
            max_iterations: Maximum number of iterations (default 3, only effective when use_reflection=True)
        """
        from pathlib import Path

        if data_root is None:
            data_root = str(Path(__file__).parent.parent / "data")

        self.data_root = data_root
        self.output_dir = output_dir or str(Path(__file__).parent.parent / "output")
        self.problem_extractor = ProblemExtractor(data_root)
        self.use_reflection = use_reflection
        self.intermediate_representation_type = normalize_intermediate_representation_type(
            intermediate_representation_type
        )
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.include_problem_text_description = include_problem_text_description

        # Choose different LangGraph agents based on whether reflection is used
        if use_reflection:
            if reflection_model_config is None:
                raise ValueError("Must provide reflection_model_config when using reflection functionality")

            # Initialize LangGraph multi-agent with reflection
            try:
                from agents.langgraph_agent_with_rethink import LangGraphFlowchartAgentWithRethink
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Reflection mode requires agents.langgraph_agent_with_rethink, which is not available."
                ) from exc
            self.langgraph_agent = LangGraphFlowchartAgentWithRethink(
                data_root=data_root,
                vision_model_2_config=vision_model_2_config,
                language_model_1_config=language_model_1_config,
                reflection_model_config=reflection_model_config,
                max_iterations=max_iterations,
                output_dir=self.output_dir,
                enable_evaluation=enable_evaluation,
                intermediate_representation_type=self.intermediate_representation_type,
                prompt_variant=self.prompt_variant,
                include_problem_text_description=self.include_problem_text_description,
            )
        else:
            # Initialize regular LangGraph multi-agent
            self.langgraph_agent = LangGraphFlowchartAgent(
                data_root=data_root,
                vision_model_2_config=vision_model_2_config,
                language_model_1_config=language_model_1_config,
                output_dir=self.output_dir,
                enable_evaluation=enable_evaluation,
                intermediate_representation_type=self.intermediate_representation_type,
                prompt_variant=self.prompt_variant,
                include_problem_text_description=self.include_problem_text_description,
            )

    def process_flowchart(self, image_path: str, dataset: str = None, use_langgraph: bool = True, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Process single flowchart, return result (format consistent with src)

        Args:
            image_path: Flowchart image path
            dataset: Dataset name
            use_langgraph: Whether to use LangGraph multi-agent workflow (default True, currently only supports LangGraph)
            save_to_file: Whether to save result to file (default True)

        Returns:
            Processing result, format consistent with src:
            {
                "task_id": "xxx",
                "response": "Original LLM response",
                "completion": "Extracted code",
                "usage": "Token usage",
                "model": "Model name"
            }
        """
        # Use LangGraph multi-agent complete workflow (includes code generation and testing)
        result = self.langgraph_agent.process_flowchart(image_path, dataset)

        # Save result to file
        if save_to_file:
            self._save_result_to_file(result, dataset)

        return result

    def _save_result_to_file(self, result: Dict[str, Any], dataset: str = None) -> None:
        """Save result to jsonl file"""
        if not dataset:
            dataset = "Algorithm"

        # Get model name (using langgraph_agent's get_output_model_name method)
        model_name = self.langgraph_agent.get_output_model_name()

        # Create output directory
        output_dir = os.path.join(self.output_dir, dataset, model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save to samples.jsonl file
        save_path = os.path.join(output_dir, "samples.jsonl")
        append_jsonl_file(save_path, result)

    def process_dataset_batch(self, dataset: str, limit: int = None, use_langgraph: bool = True,
                           enable_evaluation: bool = True) -> Dict[str, Any]:
        """
        Batch process entire dataset

        Args:
            dataset: Dataset name (Algorithm, HumanEval-V, MATH)
            limit: Limit number of images to process, None means process all images
            use_langgraph: Whether to use LangGraph workflow
            enable_evaluation: Whether to enable evaluation

        Returns:
            Batch processing result statistics
        """
        # Get all image paths from dataset
        image_paths = self._get_dataset_images(dataset)

        if limit and limit > 0:
            image_paths = image_paths[:limit]

        total_images = len(image_paths)
        if total_images == 0:
            return {
                'dataset': dataset,
                'total': 0,
                'processed': 0,
                'success': 0,
                'failed': 0,
                'results': []
            }

        print(f"Starting batch processing of dataset '{dataset}', total {total_images} images...")

        # Check processed results (consistent with src)
        model_name = self.langgraph_agent.get_output_model_name()
        output_dir = os.path.join(self.output_dir, dataset, model_name)
        save_path = os.path.join(output_dir, "samples.jsonl")

        print("-" * 20, "Check existing results", "-" * 20)
        try:
            done_data = read_jsonl_file(save_path)
            done_ids = [x["task_id"] for x in done_data]
            print(f"Skipping {len(done_ids)} tasks")
            image_paths = [img for img in image_paths if Path(img).stem not in done_ids]
            print(f"Remaining {len(image_paths)} tasks")
            assert len(image_paths) + len(done_ids) == total_images, "Skipping data and remaining tasks do not match the original data size"
        except Exception as e:
            print(e)
            print("No existing results!")
        print("-" * 20, "Starting", "-" * 20)

        results = []
        success_count = 0
        failed_count = 0

        # Use tqdm to show progress bar
        with tqdm(total=len(image_paths), desc=f"Processing {dataset}", unit="images") as pbar:
            for image_path in image_paths:
                pbar.set_postfix_str(f"Current: {Path(image_path).name}")

                try:
                    start_time = time.time()

                    # Process current image (result will be automatically saved to file)
                    result = self.process_flowchart(image_path, dataset, use_langgraph, save_to_file=True)

                    processing_time = time.time() - start_time

                    # Parse response to get detailed information
                    response_data = json.loads(result.get("response", "{}"))
                    status = response_data.get("status", "unknown")
                    problem_title = response_data.get("problem_info", {}).get("title", "Unknown") if response_data.get("problem_info") else "Unknown"

                    # If there are test results, get test results
                    test_result = response_data.get("test_result")

                    result_summary = {
                        'image_path': image_path,
                        'image_name': Path(image_path).name,
                        'task_id': result.get("task_id"),
                        'status': status,
                        'problem_title': problem_title,
                        'processing_time': processing_time,
                        'test_result': test_result,
                        'error': response_data.get("error")
                    }

                    results.append(result_summary)

                    if status == 'completed':
                        success_count += 1
                        if test_result:
                            pass_rate = test_result.get('pass_rate', 0)
                            pbar.write(f"  ✓ {Path(image_path).name}: Success (Pass rate: {pass_rate:.1f}%)")
                        else:
                            pbar.write(f"  ✓ {Path(image_path).name}: Success")
                    else:
                        failed_count += 1
                        pbar.write(f"  ✗ {Path(image_path).name}: Failed - {response_data.get('error', 'Unknown error')}")

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    pbar.write(f"  ✗ {Path(image_path).name}: Exception - {error_msg}")

                    results.append({
                        'image_path': image_path,
                        'image_name': Path(image_path).name,
                        'task_id': Path(image_path).stem,
                        'status': 'failed',
                        'problem_title': 'Unknown',
                        'processing_time': 0,
                        'test_result': None,
                        'error': error_msg
                    })

                    # Update progress bar
                    pbar.update(1)

        # Summary statistics
        summary = {
            'dataset': dataset,
            'total': total_images,
            'processed': len(results),
            'success': success_count,
            'failed': failed_count,
            'success_rate': success_count / total_images * 100 if total_images > 0 else 0,
            'results': results
        }

        pbar.write(f"\nBatch processing complete!")
        pbar.write(f"Dataset: {dataset}")
        pbar.write(f"Total images: {total_images}")
        pbar.write(f"Success: {success_count}")
        pbar.write(f"Failed: {failed_count}")
        pbar.write(f"Success rate: {summary['success_rate']:.1f}%")
        pbar.write(f"Results saved to: {save_path}")

        # Execute evaluation (if enabled)
        if enable_evaluation and self.langgraph_agent.enable_evaluation:
            pbar.write(f"\nStarting evaluation...")
            try:
                evaluation_results = self.langgraph_agent.evaluate_samples(
                    sample_file=save_path,
                    n_workers=4,
                    timeout=3.0
                )
                summary['evaluation'] = evaluation_results
            except Exception as e:
                pbar.write(f"Evaluation failed: {e}")
                summary['evaluation_error'] = str(e)

        return summary    
    def _get_dataset_images(self, dataset: str) -> List[str]:
        """
        Get all image paths from dataset

        Args:
            dataset: Dataset name

        Returns:
            List of image paths
        """
        import re

        def natural_sort_key(filename: str) -> tuple:
            """
            Natural sorting key function, converts numbers in filename to integers for comparison

            Args:
                filename: Filename

            Returns:
                Sorting key tuple
            """
            # Split string into numeric and non-numeric parts
            parts = re.split(r'(\d+)', filename)
            # Convert numeric parts to integers, keep non-numeric parts as-is
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        dataset_dir = Path(self.data_root) / dataset
        images_dir = dataset_dir / "images"

        if not images_dir.exists():
            raise ValueError(f"Images directory for dataset '{dataset}' does not exist: {images_dir}")

        # Support common image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(images_dir.glob(f"*{ext}"))
            image_paths.extend(images_dir.glob(f"*{ext.upper()}"))

        # Convert to string paths and deduplicate (Windows filesystem is case-insensitive and may cause duplicates)
        image_paths = list(set(str(path) for path in image_paths))

        # Use natural sorting
        image_paths = sorted(image_paths, key=natural_sort_key)

        return image_paths

    def get_session_result(self, session_id: str):
        """Get session result"""
        return self.session_manager.get_session(session_id)

    def get_session_progress(self, session_id: str):
        """Get session progress"""
        return self.session_manager.get_session_progress(session_id)

    def list_available_problems(self, dataset: str = None):
        """List available problems"""
        return self.problem_extractor.list_available_problems(dataset)

    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        data_root = Path(self.data_root)
        datasets = []

        if data_root.exists():
            for item in data_root.iterdir():
                if item.is_dir() and not item.name.startswith('.') and not item.name == 'images':
                    # Check if there's an images subdirectory
                    images_dir = item / "images"
                    if images_dir.exists():
                        datasets.append(item.name)

        return datasets


def load_config(config_path: str, configs_dir: Path) -> Dict[str, Any]:
    """Load configuration file, supports model name or complete path"""
    # If only model name (like qwen, glm, etc.), automatically add suffix
    if not config_path.endswith('.json'):
        # Try multiple possible config filenames
        possible_names = [
            f"{config_path}_config.json",
            f"{config_path}_api_key_config.json",
        ]
        for name in possible_names:
            test_path = configs_dir / name
            if test_path.exists():
                config_path = test_path
                break
        else:
            # None exist, use first as default
            config_path = configs_dir / possible_names[0]

    # If relative path, relative to configs directory
    if not Path(config_path).exists():
        config_path = configs_dir / config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_stream_settings(config: Dict[str, Any], role: str) -> None:
    """Inject streaming output options into model configuration"""
    config["stream"] = True
    config.setdefault("stream_label", f"{role}:{config.get('model', 'unknown-model')}")


def main():
    """Main function - supports command line arguments"""
    # Get absolute path of src directory
    src_dir = Path(__file__).parent
    project_root = src_dir.parent

    parser = argparse.ArgumentParser(description='Flowchart Code Generation System (LangGraph Multi-Agent)')
    parser.add_argument('mode', nargs='?', default='single',
                       choices=['single', 'batch', 'list', 'test'],
                       help='Run mode: single-single processing, batch-batch processing, list-list datasets, test-test components')
    parser.add_argument('--dataset', '-d', default='Algorithm',
                       help='Dataset name (Algorithm, HumanEval-V, MATH, LiveCodeBench)')
    parser.add_argument('--image', '-i',
                       default=str(project_root / "data" / "Algorithm" / "images" / "weekly-contest-381-minimum-number-of-pushes-to-type-word-i.png"),
                       help='Flowchart image path (used in single mode)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of images to process in batch mode')
    parser.add_argument('--no-langgraph', action='store_true',
                       help='Do not use LangGraph workflow (use traditional processing flow)')
    parser.add_argument('--output', '-o', default=None,
                       help='Batch processing result output JSON file path')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default is output directory)')
    parser.add_argument('--vision-model-2', default='qwen2_5_vl_7b_local',
                       help='Vision model 2 configuration (for ILR generation), can be model name or complete path')
    parser.add_argument('--language-model-1', default='qwen',
                       help='Language model 1 configuration (for code generation), can be model name or complete path')
    parser.add_argument('--enable-evaluation', action='store_true', default=True,
                       help='Enable evaluation (using src evaluation method), enabled by default')
    parser.add_argument('--disable-evaluation', action='store_true',
                       help='Disable evaluation')
    parser.add_argument('--use-reflection', action='store_true',
                       help='Enable reflection functionality (requires --reflection-model)')
    parser.add_argument('--reflection-model', default=None,
                       help='Reflection model configuration (for reflection iteration), can be model name or complete path')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Maximum number of iterations (default 3, only effective when reflection is enabled)')
    parser.add_argument('--intermediate-representation', default='ilr', choices=['ilr', 'text'],
                       help='Intermediate representation type output by Agent1: ilr-structured JSON, text-text intermediate representation (for ablation study)')
    parser.add_argument('--prompt-variant', default='default', choices=['default', 'self_planning'],
                       help='Prompt variant: default or self_planning')
    parser.add_argument('--stream-output', action='store_true',
                       help='Enable model streaming output, print generated content in real-time in terminal')
    parser.add_argument('--include-problem-text-description',
                       type=str2bool, nargs='?', const=True, default=True,
                       help='Whether to include problem text description from dataset as additional context for ILR/reflection process, default true; can pass false to disable')

    args = parser.parse_args()

    # Check reflection functionality parameters
    if args.use_reflection and not args.reflection_model:
        parser.error("Must provide --reflection-model parameter when using reflection functionality")

    # Load configuration files
    configs_dir = src_dir / "configs"

    # Vision model 2 configuration (for ILR generation)
    vision_model_2_config = load_config(args.vision_model_2, configs_dir)

    # Language model 1 configuration (for code generation)
    language_model_1_config = load_config(args.language_model_1, configs_dir)

    # Reflection model configuration (if reflection is enabled)
    reflection_model_config = None
    if args.use_reflection:
        reflection_model_config = load_config(args.reflection_model, configs_dir)

    if args.stream_output:
        apply_stream_settings(vision_model_2_config, "vision")
        apply_stream_settings(language_model_1_config, "language")
        if reflection_model_config:
            apply_stream_settings(reflection_model_config, "reflection")

    # Set default output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(project_root / "output")

    # Determine whether to enable evaluation
    enable_evaluation = args.enable_evaluation and not args.disable_evaluation

    processor = FlowchartProcessor(
        data_root=str(project_root / "data"),
        vision_model_2_config=vision_model_2_config,
        language_model_1_config=language_model_1_config,
        output_dir=output_dir,
        enable_evaluation=enable_evaluation,
        use_reflection=args.use_reflection,
        reflection_model_config=reflection_model_config,
        max_iterations=args.max_iterations,
        intermediate_representation_type=args.intermediate_representation,
        prompt_variant=args.prompt_variant,
        include_problem_text_description=args.include_problem_text_description,
    )
    
    if args.mode == 'list':
        # List all available datasets
        datasets = processor.list_datasets()
        print("Available datasets:")
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    elif args.mode == 'test':
        # Test components
        test_components()
        return
    
    elif args.mode == 'single':
        # Single flowchart processing
        image_path = args.image
        dataset = args.dataset
        use_langgraph = not args.no_langgraph
        
        if not Path(image_path).exists():
            print(f"Error: Image file does not exist: {image_path}")
            return
        
        print(f"Starting single flowchart processing...")
        print(f"Image: {image_path}")
        print(f"Dataset: {dataset}")
        print(f"Using LangGraph: {use_langgraph}")
        print(f"Intermediate representation: {get_intermediate_representation_display_name(args.intermediate_representation)}")
        
        try:
            result = processor.process_flowchart(image_path, dataset, use_langgraph, save_to_file=True)
            
            print(f"\nProcessing complete!")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Model: {result.get('model')}")
            
            # Parse response to get detailed information
            response_data = json.loads(result.get("response", "{}"))
            status = response_data.get("status", "unknown")
            
            print(f"Status: {status}")
            
            if response_data.get('error'):
                print(f"Error: {response_data.get('error')}")
            
            if response_data.get('problem_info'):
                problem = response_data['problem_info']
                print(f"\nProblem information:")
                print(f"  Title: {problem.get('title', 'Unknown')}")
                print(f"  Difficulty: {problem.get('difficulty', 'Unknown')}")
                print(f"  Number of test cases: {len(problem.get('test_cases', []))}")

            if response_data.get('ilr'):
                ir_display_name = get_intermediate_representation_display_name(
                    response_data.get('intermediate_representation_type', args.intermediate_representation)
                )
                print(f"\nGenerated {ir_display_name}:")
                print(response_data['ilr'])

            if result.get('completion'):
                print(f"\nGenerated code:")
                print(result['completion'])

            if response_data.get('test_result') and use_langgraph:
                test_result = response_data['test_result']
                print(f"\nCode test results:")
                print(f"  Success: {test_result.get('success', False)}")
                print(f"  Passed: {test_result.get('passed', 0)}/{test_result.get('total', 0)}")
                print(f"  Pass rate: {test_result.get('pass_rate', 0):.1f}%")
                if test_result.get('error'):
                    print(f"  Error: {test_result.get('error')}")
        
        except Exception as e:
            print(f"Processing failed: {e}")
    
    elif args.mode == 'batch':
        # Batch processing
        dataset = args.dataset
        limit = args.limit
        use_langgraph = not args.no_langgraph
        
        print(f"Starting batch processing...")
        print(f"Dataset: {dataset}")
        print(f"Limit: {limit if limit else 'unlimited'}")
        print(f"Using LangGraph: {use_langgraph}")
        print(f"Evaluation enabled: {enable_evaluation}")
        print(f"Intermediate representation: {get_intermediate_representation_display_name(args.intermediate_representation)}")
        
        try:
            summary = processor.process_dataset_batch(dataset, limit, use_langgraph, enable_evaluation)
            
            # Save results to file
            if args.output:
                output_file = Path(args.output)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"\nBatch processing results saved to: {output_file}")
            
            # Display detailed statistics
            print(f"\nBatch processing complete!")
            print(f"Dataset: {summary['dataset']}")
            print(f"Total images: {summary['total']}")
            print(f"Success: {summary['success']}")
            print(f"Failed: {summary['failed']}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            
            # Display evaluation results (if any)
            if 'evaluation' in summary:
                eval_results = summary['evaluation']
                print(f"\nEvaluation results:")
                for key, value in eval_results.items():
                    print(f"  {key}: {value}")
            
            # Display failure details (if any)
            failed_results = [r for r in summary['results'] if r['status'] != 'completed']
            if failed_results:
                print(f"\nFailure details:")
                for result in failed_results:
                    print(f"  - {result['image_name']}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"Batch processing failed: {e}")
    
    else:
        parser.print_help()


def test_components():
    """Test each component"""
    print("=== Testing Problem Extractor ===")
    extractor = ProblemExtractor()
    test_image_name = "weekly-contest-381-minimum-number-of-pushes-to-type-word-i"
    problem = extractor.extract_problem_by_image_name(test_image_name)
    
    if problem:
        print(f"Successfully extracted problem: {problem['title']}")
        print(f"Number of test cases: {len(problem['test_cases'])}")
    else:
        print("Problem extraction failed")
    
    print("\n=== Testing Session Manager ===")
    # Use the session directory under the output directory
    session_dir = Path(__file__).parent.parent / "output" / "session"
    manager = SessionManager(session_dir=str(session_dir))
    session_id = manager.create_session("test.png", "Algorithm")
    progress = manager.get_session_progress(session_id)
    print(f"Session progress: {progress}")
    
    print("\n=== Testing LangGraph Agent ===")
    try:
        from agents.langgraph_agent import LangGraphFlowchartAgent
        agent = LangGraphFlowchartAgent()
        print("LangGraph agent initialized successfully")
    except Exception as e:
        print(f"LangGraph agent initialization failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_components()
    else:
        main()
