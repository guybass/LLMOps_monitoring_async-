"""
Example 18: Repository Topology Scanning

Demonstrates how to scan a Python repository to extract:
- Module structure and dependencies
- Function call relationships
- Circular dependency detection
- Complexity metrics
- Visualization data export

This forms the foundation for correlating code structure with token consumption.
"""

import asyncio
import json
from pathlib import Path

from llmops_monitoring.topology import (
    CodeParser,
    DependencyGraphBuilder,
)


async def scan_repository_topology():
    """
    Scan a Python repository and analyze its topology.
    """
    print("=" * 80)
    print("Repository Topology Scanner - Example 18")
    print("=" * 80)

    # Configuration
    # Scan the llamonitor-async repository itself!
    repo_path = Path(__file__).parent.parent.parent
    print(f"\n📁 Scanning repository: {repo_path}")
    print()

    # ========== Step 1: Parse Repository ==========
    print("🔍 Step 1: Parsing repository...")
    print("-" * 80)

    parser = CodeParser()

    # Parse all Python files (excluding tests and virtual envs)
    modules = parser.parse_repository(
        str(repo_path),
        include_patterns=["**/*.py"],
        exclude_patterns=[
            "**/test_*.py",
            "**/tests/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/dist/**",
            "**/build/**"
        ]
    )

    print(f"✅ Parsed {len(modules)} modules")
    print()

    # Show some parsed modules
    print("Sample modules:")
    for i, (module_path, module_info) in enumerate(list(modules.items())[:5]):
        print(f"  {i+1}. {module_path}")
        print(f"     - Functions: {len(module_info.functions)}")
        print(f"     - Classes: {len(module_info.classes)}")
        print(f"     - Lines: {module_info.line_count}")
        print(f"     - Complexity: {module_info.complexity_score:.2f}")
        print(f"     - Imports: {len(module_info.imports)}")
    print()

    # ========== Step 2: Build Dependency Graph ==========
    print("🕸️  Step 2: Building dependency graph...")
    print("-" * 80)

    builder = DependencyGraphBuilder()
    topology = builder.build_repository_topology(modules)

    print(f"✅ Built topology:")
    print(f"   - Module dependencies: {len(topology.module_dependencies)}")
    print(f"   - Function calls: {len(topology.function_calls)}")
    print()

    # ========== Step 3: Analyze Metrics ==========
    print("📊 Step 3: Analyzing metrics...")
    print("-" * 80)

    metrics = topology.metrics

    print(f"Repository Statistics:")
    print(f"  📦 Total modules: {metrics.total_modules}")
    print(f"  🔧 Total functions: {metrics.total_functions}")
    print(f"  🏗️  Total classes: {metrics.total_classes}")
    print(f"  📝 Total lines of code: {metrics.total_lines:,}")
    print()

    print(f"Dependency Statistics:")
    print(f"  🔗 Total dependencies: {metrics.total_dependencies}")
    print(f"  ♻️  Circular dependencies: {metrics.circular_dependencies}")
    print(f"  📏 Maximum depth: {metrics.max_depth}")
    print()

    print(f"Complexity Statistics:")
    print(f"  📈 Average complexity: {metrics.avg_complexity:.2f}")
    print(f"  🔥 Maximum complexity: {metrics.max_complexity:.2f}")
    print(f"  📞 Total function calls: {metrics.total_calls}")
    print()

    # ========== Step 4: Detect Circular Dependencies ==========
    if topology.circular_deps:
        print("⚠️  Step 4: Circular Dependencies Detected!")
        print("-" * 80)

        for i, circ in enumerate(topology.circular_deps[:5], 1):
            print(f"\n  Cycle {i} (length: {circ.cycle_length}):")
            for j, module in enumerate(circ.cycle):
                next_module = circ.cycle[(j + 1) % len(circ.cycle)]
                print(f"    {module}")
                print(f"      ↓ imports")
            print(f"    {circ.cycle[0]} (back to start)")

        if len(topology.circular_deps) > 5:
            print(f"\n  ... and {len(topology.circular_deps) - 5} more cycles")
        print()
    else:
        print("✅ Step 4: No circular dependencies found!")
        print()

    # ========== Step 5: Most Complex Components ==========
    print("🔥 Step 5: Most Complex Components")
    print("-" * 80)

    print("\nMost Complex Modules:")
    for i, module_path in enumerate(metrics.most_complex_modules[:5], 1):
        module_info = modules.get(module_path)
        if module_info:
            print(f"  {i}. {module_path}")
            print(f"     Complexity: {module_info.complexity_score:.2f}")

    print("\nMost Complex Functions:")
    for i, func_name in enumerate(metrics.most_complex_functions[:5], 1):
        print(f"  {i}. {func_name}")
    print()

    # ========== Step 6: Most Imported Modules ==========
    print("📦 Step 6: Most Imported Modules (Dependencies)")
    print("-" * 80)

    for i, module_path in enumerate(metrics.most_imported_modules[:10], 1):
        # Count how many modules import this
        import_count = sum(
            1 for dep in topology.module_dependencies
            if dep.to_module == module_path
        )
        print(f"  {i}. {module_path}")
        print(f"     Imported by: {import_count} modules")
    print()

    # ========== Step 7: Export for Visualization ==========
    print("📤 Step 7: Exporting visualization data...")
    print("-" * 80)

    # Prepare D3.js format
    visualization_data = {
        "nodes": [],
        "links": []
    }

    # Add nodes (modules)
    for module_path, module_info in modules.items():
        visualization_data["nodes"].append({
            "id": module_path,
            "name": module_info.name,
            "functions": len(module_info.functions),
            "classes": len(module_info.classes),
            "lines": module_info.line_count,
            "complexity": module_info.complexity_score
        })

    # Add links (dependencies)
    for dep in topology.module_dependencies:
        visualization_data["links"].append({
            "source": dep.from_module,
            "target": dep.to_module,
            "weight": dep.weight,
            "is_circular": dep.is_circular
        })

    # Save to file
    output_file = repo_path / "topology_visualization.json"
    with open(output_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)

    print(f"✅ Saved visualization data to: {output_file}")
    print()

    # ========== Step 8: Summary ==========
    print("📋 Summary")
    print("=" * 80)
    print(f"""
The topology scanner has analyzed your repository and found:

📊 Code Structure:
  - {metrics.total_modules} Python modules
  - {metrics.total_functions} functions
  - {metrics.total_classes} classes
  - {metrics.total_lines:,} lines of code

🔗 Dependencies:
  - {metrics.total_dependencies} module dependencies
  - {metrics.circular_dependencies} circular dependencies {'⚠️  (needs attention!)' if metrics.circular_dependencies > 0 else '✅'}
  - Maximum import depth: {metrics.max_depth}

📈 Complexity:
  - Average complexity: {metrics.avg_complexity:.2f}
  - Highest complexity: {metrics.max_complexity:.2f}
  - Total function calls tracked: {metrics.total_calls}

💡 Next Steps:
  1. Review circular dependencies and consider refactoring
  2. Focus on high-complexity modules for optimization
  3. Use visualization data to understand module relationships
  4. Integrate with token consumption tracking (Example 19)

Visualization data saved to: {output_file}
    """)

    print("\n✨ Topology scan complete!")
    print("=" * 80)


# ========== Advanced: Module Detail Analysis ==========
async def analyze_specific_module():
    """
    Demonstrate detailed analysis of a specific module.
    """
    print("\n")
    print("=" * 80)
    print("Advanced: Analyzing Specific Module")
    print("=" * 80)

    repo_path = Path(__file__).parent.parent.parent
    parser = CodeParser()

    # Parse specific module
    module_file = repo_path / "llmops_monitoring" / "instrumentation" / "decorators.py"

    if module_file.exists():
        print(f"\n🔍 Analyzing: {module_file.name}")
        print("-" * 80)

        module_info = parser.parse_file(str(module_file))

        if module_info:
            print(f"\nModule: {module_info.module_path}")
            print(f"Location: {module_info.file_path}")
            print(f"Lines: {module_info.line_count}")
            print(f"Complexity: {module_info.complexity_score:.2f}")
            print()

            print(f"Imports ({len(module_info.imports)}):")
            for imp in module_info.imports[:10]:
                print(f"  - {imp.import_type.value}: {imp.module_name}")
                if imp.imported_names:
                    print(f"    └─ {', '.join(imp.imported_names[:5])}")
            print()

            print(f"Functions ({len(module_info.functions)}):")
            for func in module_info.functions:
                print(f"  - {func.name}")
                print(f"    Type: {func.function_type.value}")
                print(f"    Complexity: {func.complexity:.1f}")
                print(f"    Lines: {func.line_count}")
                if func.decorators:
                    print(f"    Decorators: {', '.join(func.decorators)}")
                if func.calls:
                    print(f"    Calls: {len(func.calls)} functions")
                print()

            print(f"Classes ({len(module_info.classes)}):")
            for cls in module_info.classes:
                print(f"  - {cls.name}")
                print(f"    Methods: {cls.method_count}")
                print(f"    Lines: {cls.line_count}")
                if cls.base_classes:
                    print(f"    Inherits: {', '.join(cls.base_classes)}")
                print()


async def main():
    """Run all examples."""
    # Main topology scan
    await scan_repository_topology()

    # Advanced module analysis
    await analyze_specific_module()


if __name__ == "__main__":
    print("\n🚀 Starting Repository Topology Scanner Example...")
    print()

    asyncio.run(main())
