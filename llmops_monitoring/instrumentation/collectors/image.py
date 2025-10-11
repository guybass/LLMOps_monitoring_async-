"""
Image metric collector.

Measures various aspects of image content with flexible configuration.
"""

from typing import Any, Dict, List, Optional
import sys

from llmops_monitoring.instrumentation.base import MetricCollector
from llmops_monitoring.schema.events import ImageMetrics


class ImageCollector(MetricCollector):
    """
    Collector for image-based metrics.

    Supports flexible measurement options:
    - count: Number of images
    - total_pixels: Sum of width × height for all images
    - file_size_bytes: Total file size
    - dimensions: Width and height (for single image)
    - format: Image format (png, jpg, etc.)

    Users can configure which metrics to collect.
    """

    def __init__(
        self,
        measure: Optional[List[str]] = None,
        image_extractor: Optional[callable] = None,
        use_pil: bool = True
    ):
        """
        Initialize image collector.

        Args:
            measure: List of metrics to collect. Options:
                     ['count', 'total_pixels', 'file_size_bytes', 'dimensions', 'format']
                     If None, collects all metrics.
            image_extractor: Custom function to extract images from result.
                            Signature: (result, args, kwargs) -> List[image_data]
            use_pil: If True, use PIL/Pillow for detailed image analysis.
                    If False, use basic size calculation.
        """
        self.measure = measure or ["count", "total_pixels", "file_size_bytes"]
        self.image_extractor = image_extractor or self._default_extract_images
        self.use_pil = use_pil
        self._pil_available = self._check_pil()

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract image metrics from result."""
        images = self.image_extractor(result, args, kwargs)

        if not images:
            return {}

        metrics = {}

        if "count" in self.measure:
            metrics["count"] = len(images)

        if self.use_pil and self._pil_available:
            metrics.update(self._collect_with_pil(images))
        else:
            metrics.update(self._collect_basic(images))

        return {"image_metrics": ImageMetrics(**metrics)}

    @property
    def metric_type(self) -> str:
        return "image"

    def should_collect(self, result: Any, args: tuple, kwargs: Dict[str, Any]) -> bool:
        """Only collect if we can extract images."""
        images = self.image_extractor(result, args, kwargs)
        return images is not None and len(images) > 0

    def _check_pil(self) -> bool:
        """Check if PIL/Pillow is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def _collect_with_pil(self, images: List[Any]) -> Dict[str, Any]:
        """Collect detailed metrics using PIL/Pillow."""
        from PIL import Image
        import io

        total_pixels = 0
        total_size = 0
        formats = []

        for img_data in images:
            try:
                # Handle different image data types
                if isinstance(img_data, Image.Image):
                    img = img_data
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                    total_size += len(img_data)
                elif isinstance(img_data, str):
                    # Assume file path
                    img = Image.open(img_data)
                    total_size += sys.getsizeof(img_data)
                else:
                    continue

                width, height = img.size
                total_pixels += width * height
                if img.format:
                    formats.append(img.format.lower())

            except Exception:
                # Skip images we can't process
                continue

        metrics = {}

        if "total_pixels" in self.measure:
            metrics["total_pixels"] = total_pixels

        if "file_size_bytes" in self.measure:
            metrics["file_size_bytes"] = total_size

        if "format" in self.measure and formats:
            # Most common format
            metrics["format"] = max(set(formats), key=formats.count)

        # For single image, add dimensions
        if len(images) == 1 and "dimensions" in self.measure:
            try:
                img = images[0]
                if isinstance(img, Image.Image):
                    metrics["width"], metrics["height"] = img.size
            except Exception:
                pass

        return metrics

    def _collect_basic(self, images: List[Any]) -> Dict[str, Any]:
        """Collect basic metrics without PIL."""
        metrics = {}
        total_size = 0

        for img_data in images:
            if isinstance(img_data, bytes):
                total_size += len(img_data)
            elif isinstance(img_data, str):
                total_size += sys.getsizeof(img_data)

        if "file_size_bytes" in self.measure:
            metrics["file_size_bytes"] = total_size

        return metrics

    @staticmethod
    def _default_extract_images(
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Default image extraction logic.

        Tries common patterns:
        1. result is a list of images
        2. result.images attribute
        3. result["images"] dict key
        4. Single image: result is an image
        """
        images = []

        # List of images
        if isinstance(result, list):
            images = result

        # Has images attribute
        elif hasattr(result, 'images'):
            images = result.images if isinstance(result.images, list) else [result.images]

        # Dict with images key
        elif isinstance(result, dict) and "images" in result:
            img_data = result["images"]
            images = img_data if isinstance(img_data, list) else [img_data]

        # Single image (check if it looks like image data)
        elif isinstance(result, bytes) and len(result) > 100:
            # Likely binary image data
            images = [result]

        # PIL Image object
        elif hasattr(result, 'size') and hasattr(result, 'format'):
            images = [result]

        return images


class MultiModalCollector(MetricCollector):
    """
    Collector for multi-modal operations (text + images).

    Combines TextCollector and ImageCollector.
    """

    def __init__(
        self,
        text_measure: Optional[List[str]] = None,
        image_measure: Optional[List[str]] = None
    ):
        from llmops_monitoring.instrumentation.collectors.text import TextCollector

        self.text_collector = TextCollector(measure=text_measure)
        self.image_collector = ImageCollector(measure=image_measure)

    def collect(
        self,
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Collect both text and image metrics."""
        metrics = {}

        # Collect text metrics
        if self.text_collector.should_collect(result, args, kwargs):
            text_metrics = self.text_collector.collect(result, args, kwargs, context)
            metrics.update(text_metrics)

        # Collect image metrics
        if self.image_collector.should_collect(result, args, kwargs):
            image_metrics = self.image_collector.collect(result, args, kwargs, context)
            metrics.update(image_metrics)

        return metrics

    @property
    def metric_type(self) -> str:
        return "multimodal"
