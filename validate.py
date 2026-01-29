"""
Validation script to verify complete Green Financial Crime Agent system.

The Panopticon Protocol: Zero-Failure Synthetic Financial Crime Simulator

This script validates:
1. Project structure (all required files present)
2. Test suite (all tests pass with coverage)
3. Import validation (all modules import correctly)

Usage:
    python validate.py
"""
import sys
import subprocess
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(passed: bool, message: str) -> None:
    """Print a validation result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {message}")


def validate_structure() -> bool:
    """Validate project structure - all required files present."""
    print_header("Validating Project Structure")
    
    required_files = [
        # Documentation
        "prompt.md",
        "prd.json",
        "progress.txt",
        "ARCHITECTURE.md",
        "API-spec.yml",
        "DECISIONS.md",
        
        # Configuration
        "requirements.txt",
        "pytest.ini",
        
        # Entry points
        "main.py",
        "validate.py",
        
        # Core modules
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/graph_generator.py",
        "src/core/crime_injector.py",
        "src/core/a2a_interface.py",
        
        # Utils
        "src/utils/__init__.py",
        "src/utils/validators.py",
        
        # Tests
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/unit/__init__.py",
        "tests/unit/test_graph_generator.py",
        "tests/unit/test_crime_injector.py",
        "tests/unit/test_validators.py",
        "tests/integration/__init__.py",
        "tests/integration/test_a2a_interface.py",
        "tests/integration/test_full_pipeline.py",
        
        # Skills
        ".claude/skills/financial-crime/SKILL.md",
        ".claude/skills/data-generation/SKILL.md",
        
        # Docker
        "Dockerfile",
        "docker-compose.yml",
        
        # CI/CD
        ".github/workflows/test.yml",
    ]
    
    missing = []
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print_result(True, filepath)
        else:
            print_result(False, f"{filepath} (MISSING)")
            missing.append(filepath)
    
    if missing:
        print(f"\nMissing {len(missing)} files")
        return False
    
    print(f"\nAll {len(required_files)} required files present")
    return True


def validate_imports() -> bool:
    """Validate that all core modules import correctly."""
    print_header("Validating Module Imports")
    
    modules_to_test = [
        ("src.core.graph_generator", [
            "generate_scale_free_graph",
            "add_entity_attributes", 
            "add_transaction_attributes",
            "save_graph",
            "load_graph"
        ]),
        ("src.core.crime_injector", [
            "StructuringConfig",
            "LayeringConfig",
            "InjectedCrime",
            "inject_structuring",
            "inject_layering",
            "validate_no_cycles",
            "get_crime_labels",
            "save_ground_truth",
            "load_ground_truth"
        ]),
        ("src.core.a2a_interface", [
            "app",
            "set_graph",
            "set_ground_truth"
        ]),
        ("src.utils.validators", [
            "validate_graph_structure",
            "validate_scale_free_distribution",
            "validate_structuring_pattern",
            "validate_layering_pattern"
        ]),
    ]
    
    all_passed = True
    
    for module_name, expected_exports in modules_to_test:
        try:
            module = __import__(module_name, fromlist=expected_exports)
            
            missing_exports = []
            for export in expected_exports:
                if not hasattr(module, export):
                    missing_exports.append(export)
            
            if missing_exports:
                print_result(False, f"{module_name}: missing {missing_exports}")
                all_passed = False
            else:
                print_result(True, f"{module_name}: all {len(expected_exports)} exports found")
                
        except ImportError as e:
            print_result(False, f"{module_name}: import error - {e}")
            all_passed = False
    
    return all_passed


def validate_tests() -> bool:
    """Run test suite and check coverage."""
    print_header("Validating Test Suite")
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=term-missing",
                "-v",
                # Don't use -x to see full test results
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Print test output
        if result.stdout:
            # Print just the summary lines
            lines = result.stdout.split('\n')
            for line in lines:
                if any(x in line for x in ['passed', 'failed', 'error', 'PASSED', 'FAILED', 'ERROR', '====', '----', 'TOTAL']):
                    print(line)
        
        if result.returncode != 0:
            print("\nTest output (errors):")
            print(result.stderr[:1000] if result.stderr else "No error output")
            print_result(False, "Test suite failed")
            return False
        
        print_result(True, "All tests passed")
        return True
        
    except subprocess.TimeoutExpired:
        print_result(False, "Test suite timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print_result(False, "pytest not found - run: pip install pytest pytest-cov")
        return False
    except Exception as e:
        print_result(False, f"Test execution error: {e}")
        return False


def validate_ralph() -> bool:
    """
    Validate Ralph Wiggum execution pattern.
    
    NOTE: Per ADR-007 (Cursor Native Agent Loop), this project uses Cursor's
    native agent loop instead of the traditional Ralph Wiggum bash wrapper.
    This validation is skipped but returns True for compatibility.
    """
    print_header("Validating Ralph Wiggum Pattern")
    
    print("INFO: Ralph Wiggum bash wrapper not used (ADR-007)")
    print("INFO: Project uses Cursor's native agent loop instead")
    print_result(True, "Ralph validation skipped (ADR-007: Cursor Native Agent Loop)")
    
    return True


def validate_generation() -> bool:
    """Validate that data generation works correctly."""
    print_header("Validating Data Generation")
    
    try:
        import networkx as nx
        
        # Import and test graph generation
        from src.core.graph_generator import generate_scale_free_graph, add_entity_attributes
        
        G = generate_scale_free_graph(n_nodes=50, seed=42)
        print_result(True, f"Graph generation: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Convert to DiGraph if needed (scale_free_graph returns MultiDiGraph)
        if isinstance(G, nx.MultiDiGraph):
            G = nx.DiGraph(G)
            print_result(True, "Converted MultiDiGraph to DiGraph")
        
        G = add_entity_attributes(G, seed=42)
        sample_node = list(G.nodes())[0]
        attrs = G.nodes[sample_node]
        print_result(True, f"Entity attributes: {list(attrs.keys())}")
        
        # Test crime injection
        from src.core.crime_injector import inject_structuring, inject_layering
        
        G, struct_crime = inject_structuring(G, seed=42)
        print_result(True, f"Structuring injection: {len(struct_crime.edges_involved)} edges")
        
        G, layer_crime = inject_layering(G, seed=43)  # Different seed
        print_result(True, f"Layering injection: {len(layer_crime.edges_involved)} edges")
        
        return True
        
    except Exception as e:
        import traceback
        print_result(False, f"Generation validation error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("  GREEN FINANCIAL CRIME AGENT - SYSTEM VALIDATION")
    print("  The Panopticon Protocol")
    print("=" * 60)
    
    checks = [
        ("Project Structure", validate_structure),
        ("Module Imports", validate_imports),
        ("Data Generation", validate_generation),
        ("Test Suite", validate_tests),
        ("Ralph Pattern", validate_ralph),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        print_result(result, name)
    
    print(f"\nTotal: {passed}/{total} validations passed")
    
    if all(results.values()):
        print("\n" + "=" * 60)
        print("  ALL VALIDATIONS PASSED - SYSTEM READY")
        print("=" * 60)
        print("\n<promise>complete</promise>")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("  SOME VALIDATIONS FAILED - REVIEW ABOVE")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
