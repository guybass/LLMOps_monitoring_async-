"""
Token Consumption Correlator.

Links runtime token consumption metrics with code topology to identify:
- Which functions/modules consume the most tokens
- Optimization opportunities (high token/low value)
- Correlations between code complexity and token usage
- Token consumption hotspots in the codebase

This enables data-driven optimization of LLM applications.
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import inspect

from llmops_monitoring.topology.models import (
    RepositoryTopology,
    ModuleInfo,
    FunctionInfo,
    TokenConsumption,
    ComponentUsage,
    HotspotAnalysis
)
from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class TrackedFunction:
    """Function with tracking metadata."""
    qualified_name: str
    function_info: FunctionInfo
    total_tokens: int = 0
    total_calls: int = 0
    total_cost: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


class TokenConsumptionCorrelator:
    """
    Correlate runtime token consumption with code topology.

    Usage:
        correlator = TokenConsumptionCorrelator()

        # Load topology
        correlator.load_topology(topology)

        # Track runtime metrics
        correlator.track_event(event)

        # Analyze
        hotspots = correlator.analyze_hotspots()
        optimization_candidates = correlator.find_optimization_candidates()
    """

    def __init__(self):
        """Initialize correlator."""
        self.topology: Optional[RepositoryTopology] = None
        self.tracked_functions: Dict[str, TrackedFunction] = {}
        self.tracked_modules: Dict[str, Dict[str, Any]] = {}

        # Function name to qualified name mapping
        self.function_name_map: Dict[str, str] = {}

        # Current call stack (for context tracking)
        self.call_stack: List[str] = []

    def load_topology(self, topology: RepositoryTopology) -> None:
        """
        Load repository topology for correlation.

        Args:
            topology: Repository topology from scanner
        """
        self.topology = topology

        # Build function registry
        for module_info in topology.modules.values():
            # Track module
            self.tracked_modules[module_info.module_path] = {
                "module_info": module_info,
                "total_tokens": 0,
                "total_calls": 0,
                "total_cost": 0.0,
                "functions": []
            }

            # Track functions
            for func in module_info.functions:
                tracked = TrackedFunction(
                    qualified_name=func.qualified_name,
                    function_info=func
                )
                self.tracked_functions[func.qualified_name] = tracked
                self.tracked_modules[module_info.module_path]["functions"].append(func.qualified_name)

                # Add to name mapping (for lookup by simple name)
                self.function_name_map[func.name] = func.qualified_name

            # Track methods
            for cls in module_info.classes:
                for method in cls.methods:
                    tracked = TrackedFunction(
                        qualified_name=method.qualified_name,
                        function_info=method
                    )
                    self.tracked_functions[method.qualified_name] = tracked
                    self.tracked_modules[module_info.module_path]["functions"].append(method.qualified_name)

                    # Add to name mapping
                    self.function_name_map[method.name] = method.qualified_name

        logger.info(
            f"Loaded topology: {len(self.tracked_functions)} functions, "
            f"{len(self.tracked_modules)} modules"
        )

    def track_event(self, event: MetricEvent) -> None:
        """
        Track a metric event and correlate with topology.

        Args:
            event: Metric event from monitoring
        """
        if not self.topology:
            logger.warning("No topology loaded. Call load_topology() first.")
            return

        # Resolve function from event metadata
        function_name = self._resolve_function_from_event(event)

        if not function_name:
            return

        tracked_func = self.tracked_functions.get(function_name)

        if not tracked_func:
            # Try to find by simple name
            if function_name in self.function_name_map:
                function_name = self.function_name_map[function_name]
                tracked_func = self.tracked_functions.get(function_name)

        if tracked_func:
            # Update function stats
            tracked_func.total_tokens += event.total_tokens
            tracked_func.total_calls += 1
            tracked_func.total_cost += event.cost_usd

            if tracked_func.first_seen is None:
                tracked_func.first_seen = event.timestamp
            tracked_func.last_seen = event.timestamp

            # Update module stats
            module_path = self._get_module_from_qualified_name(function_name)
            if module_path and module_path in self.tracked_modules:
                module_data = self.tracked_modules[module_path]
                module_data["total_tokens"] += event.total_tokens
                module_data["total_calls"] += 1
                module_data["total_cost"] += event.cost_usd

            logger.debug(f"Tracked event for {function_name}: {event.total_tokens} tokens")
        else:
            logger.debug(f"Function not found in topology: {function_name}")

    def track_function_call(self, function_name: str) -> None:
        """
        Manually track a function call (decorator integration).

        Args:
            function_name: Qualified function name
        """
        self.call_stack.append(function_name)

    def track_function_return(self) -> None:
        """Track function return (pop call stack)."""
        if self.call_stack:
            self.call_stack.pop()

    def get_current_function(self) -> Optional[str]:
        """Get current function from call stack."""
        return self.call_stack[-1] if self.call_stack else None

    def get_top_token_consumers(
        self,
        limit: int = 20,
        component_type: str = "function"
    ) -> List[ComponentUsage]:
        """
        Get top token consuming components.

        Args:
            limit: Maximum number of results
            component_type: 'function' or 'module'

        Returns:
            List of ComponentUsage sorted by tokens
        """
        if component_type == "function":
            # Sort functions by total tokens
            sorted_funcs = sorted(
                self.tracked_functions.values(),
                key=lambda f: f.total_tokens,
                reverse=True
            )[:limit]

            # Convert to ComponentUsage
            results = []
            for tracked in sorted_funcs:
                if tracked.total_tokens == 0:
                    continue

                token_data = TokenConsumption(
                    component_id=tracked.function_info.function_id,
                    component_type="function",
                    component_name=tracked.qualified_name,
                    total_tokens=tracked.total_tokens,
                    total_calls=tracked.total_calls,
                    avg_tokens_per_call=tracked.total_tokens / tracked.total_calls if tracked.total_calls > 0 else 0,
                    total_cost_usd=tracked.total_cost,
                    first_seen=tracked.first_seen,
                    last_seen=tracked.last_seen
                )

                usage = self._create_component_usage(tracked.function_info, token_data)
                results.append(usage)

            return results

        elif component_type == "module":
            # Sort modules by total tokens
            sorted_modules = sorted(
                self.tracked_modules.items(),
                key=lambda m: m[1]["total_tokens"],
                reverse=True
            )[:limit]

            results = []
            for module_path, module_data in sorted_modules:
                if module_data["total_tokens"] == 0:
                    continue

                module_info = module_data["module_info"]

                # Create synthetic function info for module
                func_info = FunctionInfo(
                    name=module_info.name,
                    qualified_name=module_info.module_path,
                    file_path=module_info.file_path,
                    complexity=module_info.complexity_score,
                    line_count=module_info.line_count
                )

                token_data = TokenConsumption(
                    component_id=module_info.module_id,
                    component_type="module",
                    component_name=module_info.module_path,
                    total_tokens=module_data["total_tokens"],
                    total_calls=module_data["total_calls"],
                    avg_tokens_per_call=module_data["total_tokens"] / module_data["total_calls"] if module_data["total_calls"] > 0 else 0,
                    total_cost_usd=module_data["total_cost"]
                )

                usage = self._create_component_usage(func_info, token_data)
                results.append(usage)

            return results

        return []

    def find_optimization_candidates(
        self,
        min_tokens: int = 1000,
        min_calls: int = 5
    ) -> List[ComponentUsage]:
        """
        Find functions that are good optimization candidates.

        Criteria:
        - High token consumption
        - High complexity
        - Frequently called
        - High tokens-per-line ratio

        Args:
            min_tokens: Minimum total tokens to consider
            min_calls: Minimum number of calls to consider

        Returns:
            List of ComponentUsage with optimization scores
        """
        candidates = []

        for tracked in self.tracked_functions.values():
            if tracked.total_tokens < min_tokens or tracked.total_calls < min_calls:
                continue

            token_data = TokenConsumption(
                component_id=tracked.function_info.function_id,
                component_type="function",
                component_name=tracked.qualified_name,
                total_tokens=tracked.total_tokens,
                total_calls=tracked.total_calls,
                avg_tokens_per_call=tracked.total_tokens / tracked.total_calls if tracked.total_calls > 0 else 0,
                total_cost_usd=tracked.total_cost,
                first_seen=tracked.first_seen,
                last_seen=tracked.last_seen
            )

            usage = self._create_component_usage(tracked.function_info, token_data)

            # Calculate optimization potential
            usage.optimization_potential = self._calculate_optimization_score(usage)

            if usage.optimization_potential > 0.5:  # Threshold
                candidates.append(usage)

        # Sort by optimization potential
        candidates.sort(key=lambda c: c.optimization_potential, reverse=True)

        return candidates

    def analyze_hotspots(
        self,
        top_n: int = 20
    ) -> HotspotAnalysis:
        """
        Perform comprehensive hotspot analysis.

        Args:
            top_n: Number of top components to include

        Returns:
            HotspotAnalysis with insights and recommendations
        """
        analysis = HotspotAnalysis()

        # Get top consumers
        analysis.top_functions_by_tokens = self.get_top_token_consumers(
            limit=top_n,
            component_type="function"
        )

        analysis.top_modules_by_tokens = [
            usage.function_info.qualified_name
            for usage in self.get_top_token_consumers(limit=top_n, component_type="module")
        ]

        # Find high complexity + high tokens
        for usage in analysis.top_functions_by_tokens:
            if usage.function_info.complexity > 10 and usage.token_consumption.total_tokens > 5000:
                analysis.high_complexity_high_tokens.append(usage.function_info.qualified_name)

        # Get optimization candidates
        analysis.optimization_candidates = self.find_optimization_candidates(
            min_tokens=1000,
            min_calls=5
        )

        # Calculate totals
        analysis.total_tokens_analyzed = sum(
            f.total_tokens for f in self.tracked_functions.values()
        )

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        # Calculate potential savings (rough estimate)
        if analysis.optimization_candidates:
            # Assume 20% reduction in top candidates
            potential_tokens_saved = sum(
                c.token_consumption.total_tokens * 0.2
                for c in analysis.optimization_candidates[:10]
            )
            if analysis.total_tokens_analyzed > 0:
                analysis.potential_savings = (potential_tokens_saved / analysis.total_tokens_analyzed) * 100

        logger.info(f"Hotspot analysis complete: {len(analysis.optimization_candidates)} optimization candidates")

        return analysis

    def get_function_report(self, qualified_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed report for a specific function.

        Args:
            qualified_name: Qualified function name

        Returns:
            Dictionary with function metrics and insights
        """
        tracked = self.tracked_functions.get(qualified_name)

        if not tracked:
            return None

        func_info = tracked.function_info

        report = {
            "name": func_info.name,
            "qualified_name": func_info.qualified_name,
            "file_path": func_info.file_path,
            "line_range": f"{func_info.line_start}-{func_info.line_end}",

            # Code metrics
            "complexity": func_info.complexity,
            "line_count": func_info.line_count,
            "is_async": func_info.is_async,
            "decorators": func_info.decorators,

            # Token metrics
            "total_tokens": tracked.total_tokens,
            "total_calls": tracked.total_calls,
            "avg_tokens_per_call": tracked.total_tokens / tracked.total_calls if tracked.total_calls > 0 else 0,
            "total_cost_usd": tracked.total_cost,

            # Derived metrics
            "tokens_per_line": tracked.total_tokens / func_info.line_count if func_info.line_count > 0 else 0,
            "cost_per_complexity": tracked.total_cost / func_info.complexity if func_info.complexity > 0 else 0,

            # Time range
            "first_seen": tracked.first_seen.isoformat() if tracked.first_seen else None,
            "last_seen": tracked.last_seen.isoformat() if tracked.last_seen else None,

            # Context
            "calls_made": func_info.calls,
            "parent_class": func_info.parent_class
        }

        return report

    def export_correlation_data(self) -> Dict[str, Any]:
        """
        Export all correlation data for persistence or visualization.

        Returns:
            Dictionary with all tracked data
        """
        return {
            "topology_summary": {
                "total_modules": len(self.tracked_modules),
                "total_functions": len(self.tracked_functions),
                "snapshot_id": str(self.topology.topology_id) if self.topology else None
            },
            "functions": [
                {
                    "qualified_name": tracked.qualified_name,
                    "total_tokens": tracked.total_tokens,
                    "total_calls": tracked.total_calls,
                    "total_cost": tracked.total_cost,
                    "avg_tokens_per_call": tracked.total_tokens / tracked.total_calls if tracked.total_calls > 0 else 0,
                    "complexity": tracked.function_info.complexity,
                    "line_count": tracked.function_info.line_count,
                    "first_seen": tracked.first_seen.isoformat() if tracked.first_seen else None,
                    "last_seen": tracked.last_seen.isoformat() if tracked.last_seen else None
                }
                for tracked in self.tracked_functions.values()
                if tracked.total_tokens > 0
            ],
            "modules": [
                {
                    "module_path": module_path,
                    "total_tokens": data["total_tokens"],
                    "total_calls": data["total_calls"],
                    "total_cost": data["total_cost"],
                    "function_count": len(data["functions"])
                }
                for module_path, data in self.tracked_modules.items()
                if data["total_tokens"] > 0
            ]
        }

    def _resolve_function_from_event(self, event: MetricEvent) -> Optional[str]:
        """
        Resolve function name from metric event.

        Tries multiple strategies:
        1. Check event.metadata for 'function_name' or 'qualified_name'
        2. Check current call stack
        3. Inspect Python call stack
        """
        # Strategy 1: Check metadata
        if hasattr(event, 'metadata') and event.metadata:
            if 'function_name' in event.metadata:
                return event.metadata['function_name']
            if 'qualified_name' in event.metadata:
                return event.metadata['qualified_name']

        # Strategy 2: Check managed call stack
        if self.call_stack:
            return self.call_stack[-1]

        # Strategy 3: Inspect Python call stack
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                function_name = caller_frame.f_code.co_name
                return function_name
        except:
            pass

        return None

    def _get_module_from_qualified_name(self, qualified_name: str) -> Optional[str]:
        """Extract module path from qualified name."""
        if not self.topology:
            return None

        # Qualified name format: module.path.Class.method or module.path.function
        parts = qualified_name.split('.')

        # Try progressively shorter paths
        for i in range(len(parts), 0, -1):
            potential_module = '.'.join(parts[:i])
            if potential_module in self.tracked_modules:
                return potential_module

        return None

    def _create_component_usage(
        self,
        function_info: FunctionInfo,
        token_data: TokenConsumption
    ) -> ComponentUsage:
        """Create ComponentUsage from function and token data."""
        usage = ComponentUsage(
            function_info=function_info,
            token_consumption=token_data
        )

        # Calculate derived metrics
        if function_info.line_count > 0:
            usage.tokens_per_line = token_data.total_tokens / function_info.line_count

        if function_info.complexity > 0:
            usage.cost_per_complexity = token_data.total_cost_usd / function_info.complexity

        return usage

    def _calculate_optimization_score(self, usage: ComponentUsage) -> float:
        """
        Calculate optimization potential score (0.0 to 1.0).

        Factors:
        - High token consumption (weight: 0.3)
        - High complexity (weight: 0.2)
        - High tokens-per-line (weight: 0.2)
        - High call frequency (weight: 0.2)
        - High cost (weight: 0.1)
        """
        score = 0.0

        # Normalize token consumption (assume 10k tokens is high)
        token_factor = min(usage.token_consumption.total_tokens / 10000, 1.0)
        score += token_factor * 0.3

        # Normalize complexity (assume 20 is high)
        complexity_factor = min(usage.function_info.complexity / 20, 1.0)
        score += complexity_factor * 0.2

        # Normalize tokens per line (assume 100 is high)
        if usage.tokens_per_line > 0:
            tpl_factor = min(usage.tokens_per_line / 100, 1.0)
            score += tpl_factor * 0.2

        # Normalize call frequency (assume 100 calls is high)
        call_factor = min(usage.token_consumption.total_calls / 100, 1.0)
        score += call_factor * 0.2

        # Normalize cost (assume $1 is high)
        cost_factor = min(usage.token_consumption.total_cost_usd / 1.0, 1.0)
        score += cost_factor * 0.1

        return score

    def _generate_recommendations(self, analysis: HotspotAnalysis) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        if analysis.optimization_candidates:
            top_candidate = analysis.optimization_candidates[0]
            recommendations.append(
                f"🎯 Priority: Optimize '{top_candidate.function_info.name}' "
                f"({top_candidate.token_consumption.total_tokens:,} tokens, "
                f"complexity {top_candidate.function_info.complexity:.1f})"
            )

        if analysis.high_complexity_high_tokens:
            recommendations.append(
                f"⚠️  {len(analysis.high_complexity_high_tokens)} functions have both "
                f"high complexity and high token consumption - consider refactoring"
            )

        if len(analysis.optimization_candidates) > 10:
            recommendations.append(
                f"📊 {len(analysis.optimization_candidates)} optimization opportunities "
                f"identified across the codebase"
            )

        if analysis.potential_savings > 10:
            recommendations.append(
                f"💰 Estimated potential savings: {analysis.potential_savings:.1f}% "
                f"of total token consumption"
            )

        # Module-level recommendations
        if analysis.top_modules_by_tokens:
            top_module = analysis.top_modules_by_tokens[0]
            recommendations.append(
                f"📦 Consider auditing module '{top_module}' - highest token consumption"
            )

        if not recommendations:
            recommendations.append(
                "✅ No major optimization opportunities detected. "
                "Continue monitoring for patterns."
            )

        return recommendations
