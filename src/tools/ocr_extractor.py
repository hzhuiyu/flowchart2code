"""
OCR Extraction Tool
Focused on purely extracting text information from flowcharts, no logical reasoning
Serves as an objective reference for ILR generation, used to verify the accuracy of information extraction
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


class OCRExtractor:
    """OCR extraction tool - pure text extraction, no logical reasoning"""

    def __init__(self, vision_model_api=None, output_dir: str = None):
        """
        Initialize OCR extraction tool

        Args:
            vision_model_api: Vision model API interface
            output_dir: Output directory path
        """
        self.vision_model_api = vision_model_api
        # Ensure absolute path is used
        if output_dir:
            self.output_dir = str(Path(output_dir).resolve())
        else:
            self.output_dir = str((Path(__file__).parent.parent.parent / "output").resolve())

    def extract_raw_text(self, image_path: str, task_id: str = None, dataset: str = None) -> Tuple[Dict[str, Any], Dict]:
        """
        Extract raw text information from flowchart (pure OCR, no logical reasoning)

        Args:
            image_path: Flowchart image file path
            task_id: Task ID (used to find cache)
            dataset: Dataset name (used to find cache)

        Returns:
            (Extracted raw text data, token usage info)
        """
        if not self.vision_model_api:
            raise ValueError("Vision model API not configured")

        # Try to read cache
        if task_id and dataset:
            cached_data = self._load_cached_ocr(task_id, dataset)
            if cached_data:
                print(f"  > Using OCR cache: {task_id}")
                # Data structure in nodes.jsonl is directly OCR content, no usage info
                return cached_data, {}

        try:
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Build pure OCR extraction prompt
            prompt = self._build_ocr_prompt()

            # Call vision model
            response, usage = self.vision_model_api.analyze_image(image_data, prompt)

            # Parse response
            ocr_data = self._parse_response(response)

            return ocr_data, usage

        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            raise

    def _load_cached_ocr(self, task_id: str, dataset: str) -> Optional[Dict]:
        """Load cached OCR data"""
        # Cache file path: output/{dataset}/nodes.jsonl (using absolute path)
        cache_file = Path(self.output_dir).resolve() / dataset / "nodes.jsonl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if task_id in line: # Quick pre-check
                        try:
                            data = json.loads(line)
                            if task_id in data:
                                return data[task_id]
                        except:
                            continue
        except:
            pass

        return None

    def _build_ocr_prompt(self) -> str:
        """Build pure OCR extraction prompt - emphasize exact extraction, no inference"""
        return """# Task: Pure OCR Text Extraction from Flowchart

## Objective:
Extract ALL visible text from the flowchart image EXACTLY as it appears. Do NOT interpret, modify, or add any information.

## Important Rules:
1. **EXACT TEXT**: Copy text character-by-character, including typos, abbreviations, and formatting
2. **NO INFERENCE**: Do not guess missing words, complete abbreviations, or fix errors
3. **NO LOGIC CONVERSION**: Do not convert natural language to code or expressions
4. **PRESERVE ORIGINAL**: Keep mathematical symbols as they appear (×, ÷, ≤, ≥, etc.)

## Output Format:
Return a JSON object with the following structure:

{
    "nodes": [
        {
            "node_id": 1,
            "shape_type": "oval/rectangle/diamond/parallelogram",
            "raw_text": "exact text inside the shape",
            "position": "top/middle/bottom (approximate vertical position)"
        }
    ],
    "edges": [
        {
            "from_node": 1,
            "to_node": 2,
            "label": "text on the arrow/line (if any, otherwise empty string)"
        }
    ],
    "other_text": [
        "any text not inside shapes (titles, annotations, etc.)"
    ]
}

## Shape Type Guide:
- oval/ellipse/rounded rectangle: usually Start/End
- rectangle: usually Process
- diamond/rhombus: usually Decision
- parallelogram: usually Input/Output

## Notes:
- List nodes in order from top to bottom, left to right
- For edge labels, common ones are "Yes", "No", "True", "False", "Y", "N"
- If text is partially visible or unclear, mark it as "[unclear: visible_part]"
- Output ONLY the JSON, no explanations or markdown formatting
"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse vision model response"""
        try:
            # Try to parse JSON directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON portion
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # If still fails, return raw response wrapper
            return {
                "raw_response": response,
                "parse_error": True,
                "nodes": [],
                "edges": [],
                "other_text": []
            }

    def extract_and_save(self, image_path: str, task_id: str, dataset: str) -> Dict[str, Any]:
        """
        Extract OCR information and save

        Args:
            image_path: Image path
            task_id: Task ID
            dataset: Dataset name

        Returns:
            Extracted OCR data
        """
        ocr_data, usage = self.extract_raw_text(image_path)

        # Add metadata
        result = {
            "task_id": task_id,
            "image_path": image_path,
            "ocr_data": ocr_data,
            "usage": usage
        }

        # Save to file
        output_path = Path(self.output_dir) / dataset / "ocr_reference.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({task_id: result}, ensure_ascii=False) + '\n')

        return result

    def batch_extract(self, images_dir: str, dataset: str,
                     limit: int = None, model_name: str = None) -> List[Dict[str, Any]]:
        """
        Batch extract OCR information

        Args:
            images_dir: Image directory
            dataset: Dataset name
            limit: Limit processing count
            model_name: Model name (used for output path)

        Returns:
            All extraction results
        """
        images_path = Path(images_dir)
        if not images_path.exists():
            raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

        # Get image files and sort
        image_files = list(images_path.glob('*.png'))
        image_files = sorted(image_files, key=lambda x: self._natural_sort_key(x.stem))

        if limit:
            image_files = image_files[:limit]

        results = []

        # Determine output path
        if model_name:
            output_file = Path(self.output_dir) / dataset / model_name / "ocr_reference.jsonl"
        else:
            output_file = Path(self.output_dir) / dataset / "ocr_reference.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check already processed tasks
        processed_ids = set()
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            processed_ids.update(data.keys())
                        except:
                            pass

        print(f"Starting batch OCR extraction, {len(image_files)} images total, {len(processed_ids)} already processed...")

        for image_file in tqdm(image_files, desc="OCR extraction"):
            task_id = image_file.stem

            # Skip already processed
            if task_id in processed_ids:
                results.append({
                    'task_id': task_id,
                    'success': True,
                    'cached': True
                })
                continue

            try:
                ocr_data, usage = self.extract_raw_text(str(image_file))

                result = {
                    "task_id": task_id,
                    "image_path": str(image_file),
                    "ocr_data": ocr_data,
                    "usage": usage
                }

                # Append save
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({task_id: result}, ensure_ascii=False) + '\n')

                results.append({
                    'task_id': task_id,
                    'success': True,
                    'cached': False,
                    'data': result
                })

            except Exception as e:
                print(f"Error processing {task_id}: {e}")
                results.append({
                    'task_id': task_id,
                    'success': False,
                    'error': str(e)
                })

        success_count = sum(1 for r in results if r['success'])
        cached_count = sum(1 for r in results if r.get('cached', False))
        failed_count = sum(1 for r in results if not r['success'])

        print(f"Batch OCR extraction complete! Success: {success_count}, Cached: {cached_count}, Failed: {failed_count}")

        return results

    def _natural_sort_key(self, s: str) -> list:
        """Natural sort key function"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]


class OCRILRComparator:
    """OCR results and ILR comparison tool"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(Path(__file__).parent.parent.parent / "output")

    def compare_single(self, task_id: str, ocr_data: Dict, ilr_data: Dict) -> Dict[str, Any]:
        """
        Compare OCR and ILR results for a single task

        Args:
            task_id: Task ID
            ocr_data: Raw data extracted by OCR
            ilr_data: Logic data generated by ILR

        Returns:
            Comparison results
        """
        comparison = {
            "task_id": task_id,
            "node_count_match": False,
            "text_mismatches": [],
            "ocr_node_count": 0,
            "ilr_node_count": 0,
            "similarity_score": 0.0
        }

        # Extract node counts
        ocr_nodes = ocr_data.get("nodes", [])

        # ILR may be in string format, needs parsing
        if isinstance(ilr_data, str):
            try:
                ilr_data = json.loads(ilr_data)
            except:
                comparison["error"] = "Unable to parse ILR data"
                return comparison

        ilr_nodes = ilr_data.get("nodes", [])

        comparison["ocr_node_count"] = len(ocr_nodes)
        comparison["ilr_node_count"] = len(ilr_nodes)
        comparison["node_count_match"] = len(ocr_nodes) == len(ilr_nodes)

        # Compare text content
        ocr_texts = [n.get("raw_text", "").strip().lower() for n in ocr_nodes]
        ilr_texts = [n.get("label", "").strip().lower() for n in ilr_nodes]

        # Calculate text similarity
        matched = 0
        for ocr_text in ocr_texts:
            # Check if there's a similar ILR text
            for ilr_text in ilr_texts:
                if self._text_similarity(ocr_text, ilr_text) > 0.7:
                    matched += 1
                    break

        if ocr_texts:
            comparison["similarity_score"] = matched / len(ocr_texts)

        # Find obvious mismatches
        for i, ocr_text in enumerate(ocr_texts):
            best_match = None
            best_score = 0
            for ilr_text in ilr_texts:
                score = self._text_similarity(ocr_text, ilr_text)
                if score > best_score:
                    best_score = score
                    best_match = ilr_text

            if best_score < 0.5:
                comparison["text_mismatches"].append({
                    "ocr_text": ocr_text,
                    "best_ilr_match": best_match,
                    "similarity": best_score
                })

        return comparison

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard similarity)"""
        if not text1 or not text2:
            return 0.0

        # Tokenize
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def batch_compare(self, dataset: str, model_name: str) -> Dict[str, Any]:
        """
        Batch compare OCR and ILR results

        Args:
            dataset: Dataset name
            model_name: Model name

        Returns:
            Batch comparison results
        """
        # Load OCR data
        ocr_file = Path(self.output_dir) / dataset / model_name / "ocr_reference.jsonl"
        ilr_file = Path(self.output_dir) / dataset / model_name / "ilr.jsonl"

        if not ocr_file.exists():
            raise FileNotFoundError(f"OCR file does not exist: {ocr_file}")
        if not ilr_file.exists():
            raise FileNotFoundError(f"ILR file does not exist: {ilr_file}")

        # Load data
        ocr_dict = {}
        with open(ocr_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ocr_dict.update(data)

        ilr_dict = {}
        with open(ilr_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ilr_dict.update(data)

        # Compare
        results = {
            "total": 0,
            "compared": 0,
            "node_count_matches": 0,
            "high_similarity": 0,  # > 0.8
            "medium_similarity": 0,  # 0.5-0.8
            "low_similarity": 0,  # < 0.5
            "details": []
        }

        common_ids = set(ocr_dict.keys()) & set(ilr_dict.keys())
        results["total"] = len(common_ids)

        for task_id in common_ids:
            ocr_data = ocr_dict[task_id].get("ocr_data", {})
            ilr_data = ilr_dict[task_id]

            comparison = self.compare_single(task_id, ocr_data, ilr_data)
            results["details"].append(comparison)
            results["compared"] += 1

            if comparison.get("node_count_match"):
                results["node_count_matches"] += 1

            score = comparison.get("similarity_score", 0)
            if score > 0.8:
                results["high_similarity"] += 1
            elif score > 0.5:
                results["medium_similarity"] += 1
            else:
                results["low_similarity"] += 1

        # Save comparison results
        output_file = Path(self.output_dir) / dataset / model_name / "ocr_ilr_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Comparison complete! Results saved to: {output_file}")
        print(f"Total: {results['total']}, Node count matches: {results['node_count_matches']}")
        print(f"High similarity (>0.8): {results['high_similarity']}, Medium (0.5-0.8): {results['medium_similarity']}, Low (<0.5): {results['low_similarity']}")

        return results


def run_ocr_extraction(config_path: str, images_dir: str, dataset: str,
                       limit: int = None, model_name: str = None):
    """
    Convenience function for running OCR extraction

    Args:
        config_path: Config file path
        images_dir: Image directory
        dataset: Dataset name
        limit: Limit count
        model_name: Model name
    """
    import sys
    from utils.vision_api_client import VisionAPIClient

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Create client and extractor
    api_client = VisionAPIClient(config_dict=config)
    extractor = OCRExtractor(vision_model_api=api_client)

    # Execute batch extraction
    results = extractor.batch_extract(images_dir, dataset, limit=limit, model_name=model_name)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR extraction tool")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing count")
    parser.add_argument("--model-name", default=None, help="Model name (used for output path)")
    parser.add_argument("--compare", action="store_true", help="Perform ILR comparison after extraction")

    args = parser.parse_args()

    # Run extraction
    run_ocr_extraction(
        config_path=args.config,
        images_dir=args.images,
        dataset=args.dataset,
        limit=args.limit,
        model_name=args.model_name
    )

    # If comparison needed
    if args.compare and args.model_name:
        comparator = OCRILRComparator()
        comparator.batch_compare(args.dataset, args.model_name)
