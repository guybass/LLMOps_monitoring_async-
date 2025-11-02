"""
Topology visualization exporter.

Exports repository topology to various visualization formats:
- D3.js (force-directed graph)
- Cytoscape.js (network visualization)
- GraphViz DOT (hierarchical graphs)
- Sankey diagram (flow visualization)
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from llmops_monitoring.topology.models import (
    RepositoryTopology,
    ModuleInfo,
    DependencyEdge,
    CallEdge
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class TopologyVisualizer:
    """
    Export topology data to various visualization formats.

    Supports:
    - D3.js force-directed graphs
    - Cytoscape.js networks
    - GraphViz DOT format
    - Sankey flow diagrams
    """

    def __init__(self):
        """Initialize visualizer."""
        pass

    def export_d3_force_graph(
        self,
        topology: RepositoryTopology,
        output_path: Optional[str] = None,
        include_functions: bool = False
    ) -> Dict[str, Any]:
        """
        Export as D3.js force-directed graph format.

        Args:
            topology: Repository topology
            output_path: Optional file path to save JSON
            include_functions: Include function-level nodes (can be very large)

        Returns:
            D3.js compatible graph data
        """
        logger.info("Exporting to D3.js force graph format...")

        graph_data = {
            "nodes": [],
            "links": []
        }

        # Add module nodes
        for module_path, module_info in topology.modules.items():
            node = {
                "id": module_path,
                "name": module_info.name,
                "type": "module",
                "group": self._get_package_group(module_path),
                "metrics": {
                    "functions": len(module_info.functions),
                    "classes": len(module_info.classes),
                    "lines": module_info.line_count,
                    "complexity": module_info.complexity_score,
                    "imports": module_info.import_count
                }
            }
            graph_data["nodes"].append(node)

        # Add function nodes (optional)
        if include_functions:
            for module_info in topology.modules.values():
                for func in module_info.functions:
                    node = {
                        "id": func.qualified_name,
                        "name": func.name,
                        "type": "function",
                        "group": module_info.module_path,
                        "metrics": {
                            "complexity": func.complexity,
                            "lines": func.line_count,
                            "calls": len(func.calls)
                        }
                    }
                    graph_data["nodes"].append(node)

        # Add module dependency links
        for dep in topology.module_dependencies:
            link = {
                "source": dep.from_module,
                "target": dep.to_module,
                "type": "dependency",
                "weight": dep.weight,
                "import_type": dep.import_type.value,
                "is_circular": dep.is_circular
            }
            graph_data["links"].append(link)

        # Add function call links (if functions included)
        if include_functions:
            for call in topology.function_calls:
                link = {
                    "source": call.caller,
                    "target": call.callee,
                    "type": "call",
                    "weight": call.call_count
                }
                graph_data["links"].append(link)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"D3.js graph saved to: {output_path}")

        logger.info(
            f"D3.js export complete: {len(graph_data['nodes'])} nodes, "
            f"{len(graph_data['links'])} links"
        )

        return graph_data

    def export_cytoscape(
        self,
        topology: RepositoryTopology,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export as Cytoscape.js format.

        Args:
            topology: Repository topology
            output_path: Optional file path to save JSON

        Returns:
            Cytoscape.js compatible elements array
        """
        logger.info("Exporting to Cytoscape.js format...")

        elements = []

        # Add module nodes
        for module_path, module_info in topology.modules.items():
            element = {
                "data": {
                    "id": module_path,
                    "label": module_info.name,
                    "type": "module",
                    "package": self._get_package_group(module_path),
                    "functions": len(module_info.functions),
                    "classes": len(module_info.classes),
                    "lines": module_info.line_count,
                    "complexity": module_info.complexity_score
                }
            }
            elements.append(element)

        # Add dependency edges
        for i, dep in enumerate(topology.module_dependencies):
            element = {
                "data": {
                    "id": f"dep_{i}",
                    "source": dep.from_module,
                    "target": dep.to_module,
                    "type": "dependency",
                    "weight": dep.weight,
                    "import_type": dep.import_type.value,
                    "is_circular": dep.is_circular
                }
            }
            elements.append(element)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(elements, f, indent=2)
            logger.info(f"Cytoscape.js graph saved to: {output_path}")

        logger.info(f"Cytoscape export complete: {len(elements)} elements")

        return elements

    def export_graphviz_dot(
        self,
        topology: RepositoryTopology,
        output_path: Optional[str] = None,
        layout: str = "dot"
    ) -> str:
        """
        Export as GraphViz DOT format.

        Args:
            topology: Repository topology
            output_path: Optional file path to save DOT file
            layout: GraphViz layout engine (dot, neato, fdp, circo, twopi)

        Returns:
            DOT format string
        """
        logger.info("Exporting to GraphViz DOT format...")

        dot_lines = [
            "digraph repository_topology {",
            f"  layout={layout};",
            "  rankdir=LR;",
            "  node [shape=box, style=filled];",
            ""
        ]

        # Define color scheme for packages
        packages = set()
        for module_path in topology.modules.keys():
            package = self._get_package_group(module_path)
            packages.add(package)

        # Create subgraphs for packages
        package_colors = self._generate_colors(len(packages))
        package_color_map = dict(zip(sorted(packages), package_colors))

        for package in sorted(packages):
            dot_lines.append(f'  subgraph cluster_{package.replace(".", "_")} {{')
            dot_lines.append(f'    label="{package}";')
            dot_lines.append(f'    color="{package_color_map[package]}";')
            dot_lines.append(f'    style=filled;')
            dot_lines.append(f'    fillcolor="{package_color_map[package]}22";')

            # Add modules in this package
            for module_path, module_info in topology.modules.items():
                if self._get_package_group(module_path) == package:
                    # Node label with metrics
                    label = f"{module_info.name}\\n"
                    label += f"Funcs: {len(module_info.functions)} "
                    label += f"Classes: {len(module_info.classes)}\\n"
                    label += f"Lines: {module_info.line_count} "
                    label += f"Complexity: {module_info.complexity_score:.1f}"

                    dot_lines.append(
                        f'    "{module_path}" [label="{label}", '
                        f'fillcolor="{package_color_map[package]}"];'
                    )

            dot_lines.append("  }")
            dot_lines.append("")

        # Add edges
        for dep in topology.module_dependencies:
            style = "dashed" if dep.is_circular else "solid"
            color = "red" if dep.is_circular else "black"
            weight = dep.weight

            dot_lines.append(
                f'  "{dep.from_module}" -> "{dep.to_module}" '
                f'[style={style}, color={color}, weight={weight}, '
                f'label="{dep.import_type.value}"];'
            )

        dot_lines.append("}")

        dot_content = "\n".join(dot_lines)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(dot_content)
            logger.info(f"GraphViz DOT saved to: {output_path}")
            logger.info(f"Generate PNG with: dot -Tpng {output_path} -o output.png")

        logger.info("GraphViz DOT export complete")

        return dot_content

    def export_sankey(
        self,
        topology: RepositoryTopology,
        output_path: Optional[str] = None,
        include_token_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export as Sankey diagram format (for Plotly or D3-sankey).

        Useful for visualizing token flow through code components.

        Args:
            topology: Repository topology
            output_path: Optional file path to save JSON
            include_token_data: Include token consumption data if available

        Returns:
            Sankey diagram data
        """
        logger.info("Exporting to Sankey diagram format...")

        # Create node list
        nodes = []
        node_map = {}

        for i, module_path in enumerate(topology.modules.keys()):
            nodes.append({
                "id": i,
                "label": module_path
            })
            node_map[module_path] = i

        # Create links (flows)
        links = []
        for dep in topology.module_dependencies:
            if dep.from_module in node_map and dep.to_module in node_map:
                link = {
                    "source": node_map[dep.from_module],
                    "target": node_map[dep.to_module],
                    "value": dep.weight,
                    "type": dep.import_type.value
                }
                links.append(link)

        sankey_data = {
            "nodes": nodes,
            "links": links
        }

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(sankey_data, f, indent=2)
            logger.info(f"Sankey diagram saved to: {output_path}")

        logger.info(
            f"Sankey export complete: {len(nodes)} nodes, {len(links)} flows"
        )

        return sankey_data

    def export_html_viewer(
        self,
        topology: RepositoryTopology,
        output_path: str,
        visualization_type: str = "d3-force"
    ) -> None:
        """
        Export as standalone HTML file with interactive visualization.

        Args:
            topology: Repository topology
            output_path: File path to save HTML
            visualization_type: Type of visualization (d3-force, cytoscape)
        """
        logger.info(f"Generating HTML viewer ({visualization_type})...")

        if visualization_type == "d3-force":
            self._export_d3_html(topology, output_path)
        elif visualization_type == "cytoscape":
            self._export_cytoscape_html(topology, output_path)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

        logger.info(f"HTML viewer saved to: {output_path}")

    def _export_d3_html(self, topology: RepositoryTopology, output_path: str) -> None:
        """Generate D3.js force graph HTML."""
        graph_data = self.export_d3_force_graph(topology, include_functions=False)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Repository Topology - D3.js Force Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
            cursor: pointer;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .link.circular {{
            stroke: #f00;
            stroke-dasharray: 5,5;
        }}
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div class="tooltip" style="display: none;"></div>

    <script>
        const data = {json.dumps(graph_data, indent=2)};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const tooltip = d3.select(".tooltip");

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", d => d.is_circular ? "link circular" : "link")
            .attr("stroke-width", d => Math.sqrt(d.weight));

        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => 5 + Math.log(d.metrics.lines + 1) * 2)
            .attr("fill", d => d3.schemeCategory10[hash(d.group) % 10])
            .on("mouseover", function(event, d) {{
                tooltip.style("display", "block")
                    .html(`
                        <strong>${{d.name}}</strong><br>
                        Path: ${{d.id}}<br>
                        Functions: ${{d.metrics.functions}}<br>
                        Classes: ${{d.metrics.classes}}<br>
                        Lines: ${{d.metrics.lines}}<br>
                        Complexity: ${{d.metrics.complexity.toFixed(2)}}
                    `);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY + 10) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.style("display", "none");
            }})
            .call(drag(simulation));

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }});

        function drag(simulation) {{
            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}

        function hash(str) {{
            let hash = 0;
            for (let i = 0; i < str.length; i++) {{
                hash = str.charCodeAt(i) + ((hash << 5) - hash);
            }}
            return Math.abs(hash);
        }}
    </script>
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _export_cytoscape_html(self, topology: RepositoryTopology, output_path: str) -> None:
        """Generate Cytoscape.js HTML."""
        elements = self.export_cytoscape(topology)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Repository Topology - Cytoscape.js</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
        }}
        #cy {{
            width: 100vw;
            height: 100vh;
            display: block;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>

    <script>
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {json.dumps(elements, indent=2)},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'width': 'data(lines)',
                        'height': 'data(lines)',
                        'background-color': '#4A90E2',
                        'font-size': '10px',
                        'text-valign': 'center',
                        'text-halign': 'center'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'data(weight)',
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                {{
                    selector: 'edge[is_circular = true]',
                    style: {{
                        'line-color': '#f00',
                        'line-style': 'dashed'
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: true,
                idealEdgeLength: 100,
                nodeOverlap: 20
            }}
        }});

        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            console.log('Node clicked:', node.data());
            alert(`Module: ${{node.data('label')}}\\nFunctions: ${{node.data('functions')}}\\nComplexity: ${{node.data('complexity')}}`);
        }});
    </script>
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _get_package_group(self, module_path: str) -> str:
        """Get top-level package name for grouping."""
        parts = module_path.split('.')
        return parts[0] if parts else module_path

    def _generate_colors(self, count: int) -> List[str]:
        """Generate distinct colors for packages."""
        colors = [
            "#4A90E2", "#50C878", "#FF6B6B", "#FFD93D",
            "#A78BFA", "#F472B6", "#34D399", "#FBBF24",
            "#60A5FA", "#F87171", "#818CF8", "#FB923C"
        ]
        # Repeat if needed
        while len(colors) < count:
            colors.extend(colors)
        return colors[:count]
