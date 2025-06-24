# oni_vision_core.py
"""
Enhanced Vision Core: Modular and hierarchical vision processing system
Following biological vision principles with clear separation of processing levels
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class VisionFeatures:
    """Container for features at each processing level"""
    raw_data: Any
    confidence: float
    metadata: Dict[str, Any]

class ProcessorBase(ABC):
    """Base class for all vision processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        
    @abstractmethod
    def process(self, input_data: Any) -> VisionFeatures:
        pass
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False

# =============================================================================
# LOW LEVEL PROCESSORS - Basic visual features and preprocessing
# =============================================================================

class EdgeDetector(ProcessorBase):
    """Detects edges using multiple algorithms"""
    
    def __init__(self):
        super().__init__("EdgeDetector")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        
        # Multiple edge detection methods
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        edges = {
            'canny': cv2.Canny(gray, 50, 150),
            'sobel_x': cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
            'sobel_y': cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3),
            'laplacian': cv2.Laplacian(gray, cv2.CV_64F)
        }
        
        # Combine edge information
        edge_magnitude = np.sqrt(edges['sobel_x']**2 + edges['sobel_y']**2)
        edge_direction = np.arctan2(edges['sobel_y'], edges['sobel_x'])
        
        return VisionFeatures(
            raw_data={
                'edges': edges,
                'magnitude': edge_magnitude,
                'direction': edge_direction,
                'edge_density': np.mean(edges['canny'] > 0)
            },
            confidence=0.9,
            metadata={'method': 'multi_edge_detection'}
        )

class ColorAnalyzer(ProcessorBase):
    """Analyzes color distribution and properties"""
    
    def __init__(self):
        super().__init__("ColorAnalyzer")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        
        # Color space conversions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Color statistics
        color_stats = {
            'bgr_mean': np.mean(image, axis=(0, 1)),
            'bgr_std': np.std(image, axis=(0, 1)),
            'hsv_hist': [cv2.calcHist([hsv], [i], None, [256], [0, 256]) for i in range(3)],
            'dominant_colors': self._extract_dominant_colors(image),
            'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
            'contrast': np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        }
        
        return VisionFeatures(
            raw_data=color_stats,
            confidence=0.95,
            metadata={'color_spaces': ['BGR', 'HSV', 'LAB']}
        )
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        from sklearn.cluster import KMeans
        
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]

class TextureAnalyzer(ProcessorBase):
    """Analyzes texture patterns and properties"""
    
    def __init__(self):
        super().__init__("TextureAnalyzer")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        from skimage.feature import local_binary_pattern
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Local Binary Pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Gabor filters for texture
        gabor_responses = []
        for theta in range(0, 180, 30):
            kernel = cv2.getGaborKernel((21, 21), 3, np.radians(theta), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            gabor_responses.append(cv2.filter2D(gray, cv2.CV_8UC3, kernel))
        
        texture_features = {
            'lbp_histogram': np.histogram(lbp.ravel(), bins=256)[0],
            'gabor_responses': gabor_responses,
            'texture_energy': np.var(gray),
            'entropy': self._calculate_entropy(gray)
        }
        
        return VisionFeatures(
            raw_data=texture_features,
            confidence=0.85,
            metadata={'methods': ['LBP', 'Gabor', 'Statistical']}
        )
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        hist = np.histogram(image, bins=256)[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

class OpticalFlowProcessor(ProcessorBase):
    """Processes optical flow between frames"""
    
    def __init__(self):
        super().__init__("OpticalFlowProcessor")
        self.prev_frame = None
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return VisionFeatures(
                raw_data={'flow': None, 'magnitude': 0, 'direction': None},
                confidence=0.0,
                metadata={'status': 'first_frame'}
            )
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, None, None)
        
        # Alternative: Farneback optical flow for dense flow
        flow_dense = cv2.calcOpticalFlowFarneback(self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
        
        self.prev_frame = gray
        
        return VisionFeatures(
            raw_data={
                'flow_dense': flow_dense,
                'magnitude': magnitude,
                'angle': angle,
                'avg_motion': np.mean(magnitude),
                'motion_vectors': flow
            },
            confidence=0.8,
            metadata={'method': 'Lucas-Kanade + Farneback'}
        )

class LowLevelProcessor:
    """Orchestrates all low-level vision processing"""
    
    def __init__(self):
        self.processors = {
            'edges': EdgeDetector(),
            'colors': ColorAnalyzer(),
            'textures': TextureAnalyzer(),
            'motion': OpticalFlowProcessor()
        }
    
    def extract_features(self, image: np.ndarray) -> Dict[str, VisionFeatures]:
        """Extract all low-level features"""
        features = {}
        for name, processor in self.processors.items():
            if processor.is_enabled():
                try:
                    features[name] = processor.process(image)
                except Exception as e:
                    print(f"Error in {name} processor: {e}")
                    features[name] = VisionFeatures(
                        raw_data=None,
                        confidence=0.0,
                        metadata={'error': str(e)}
                    )
        return features

# =============================================================================
# MID LEVEL PROCESSORS - Grouping and spatial relationships
# =============================================================================

class SegmentationProcessor(ProcessorBase):
    """Advanced image segmentation"""
    
    def __init__(self):
        super().__init__("SegmentationProcessor")
        
    def process(self, image: np.ndarray, low_features: Dict = None) -> VisionFeatures:
        import cv2
        from sklearn.cluster import KMeans
        
        # Superpixel segmentation using SLIC
        try:
            from skimage.segmentation import slic, mark_boundaries
            segments = slic(image, n_segments=300, compactness=10, sigma=1)
            
            # Watershed segmentation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # K-means color segmentation
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            segmented_image = labels.reshape(image.shape[:2])
            
            segmentation_data = {
                'superpixels': segments,
                'num_superpixels': len(np.unique(segments)),
                'color_segments': segmented_image,
                'num_color_regions': 8,
                'segment_properties': self._analyze_segments(segments, image)
            }
            
        except ImportError:
            # Fallback to basic segmentation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation_data = {
                'contours': contours,
                'num_objects': len(contours),
                'binary_mask': binary
            }
        
        return VisionFeatures(
            raw_data=segmentation_data,
            confidence=0.8,
            metadata={'methods': ['superpixel', 'watershed', 'kmeans']}
        )
    
    def _analyze_segments(self, segments: np.ndarray, image: np.ndarray) -> Dict:
        """Analyze properties of image segments"""
        segment_props = {}
        unique_segments = np.unique(segments)
        
        for segment_id in unique_segments[:10]:  # Analyze first 10 segments
            mask = segments == segment_id
            if np.sum(mask) > 0:
                segment_props[segment_id] = {
                    'area': np.sum(mask),
                    'centroid': np.mean(np.where(mask), axis=1),
                    'mean_color': np.mean(image[mask], axis=0),
                    'bbox': self._get_bounding_box(mask)
                }
        
        return segment_props
    
    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of a binary mask"""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return (0, 0, 0, 0)
        return (min(rows), min(cols), max(rows), max(cols))

class DepthEstimator(ProcessorBase):
    """Estimates depth from monocular images"""
    
    def __init__(self):
        super().__init__("DepthEstimator")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        # Simplified depth estimation using image gradients and texture
        import cv2
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use blur as inverse depth cue
        blur_kernel = cv2.getGaussianKernel(15, 5)
        blur_measure = cv2.filter2D(gray, -1, blur_kernel)
        
        # Use edge density as depth cue
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(float), (15, 15), 5)
        
        # Combine cues (this is a simplified approach)
        depth_estimate = edge_density * 0.7 + (255 - blur_measure) * 0.3
        depth_estimate = cv2.normalize(depth_estimate, None, 0, 255, cv2.NORM_MINMAX)
        
        return VisionFeatures(
            raw_data={
                'depth_map': depth_estimate,
                'confidence_map': edge_density / 255.0,
                'depth_statistics': {
                    'mean_depth': np.mean(depth_estimate),
                    'depth_variance': np.var(depth_estimate),
                    'near_ratio': np.sum(depth_estimate > 200) / depth_estimate.size,
                    'far_ratio': np.sum(depth_estimate < 50) / depth_estimate.size
                }
            },
            confidence=0.6,  # Lower confidence for monocular depth
            metadata={'method': 'gradient_based_estimation'}
        )

class MotionAnalyzer(ProcessorBase):
    """Analyzes motion patterns and trajectories"""
    
    def __init__(self):
        super().__init__("MotionAnalyzer")
        self.trajectory_buffer = []
        self.max_buffer_size = 30
        
    def process(self, optical_flow: VisionFeatures) -> VisionFeatures:
        if optical_flow.raw_data.get('flow_dense') is None:
            return VisionFeatures(
                raw_data={'trajectories': [], 'motion_patterns': {}},
                confidence=0.0,
                metadata={'status': 'no_optical_flow'}
            )
        
        flow = optical_flow.raw_data['flow_dense']
        magnitude = optical_flow.raw_data['magnitude']
        
        # Detect motion regions
        motion_threshold = np.percentile(magnitude, 75)
        motion_mask = magnitude > motion_threshold
        
        # Track trajectories
        motion_centers = self._find_motion_centers(motion_mask)
        self.trajectory_buffer.append(motion_centers)
        
        if len(self.trajectory_buffer) > self.max_buffer_size:
            self.trajectory_buffer.pop(0)
        
        # Analyze motion patterns
        motion_patterns = self._analyze_motion_patterns(flow, magnitude)
        
        return VisionFeatures(
            raw_data={
                'motion_mask': motion_mask,
                'motion_centers': motion_centers,
                'trajectories': self.trajectory_buffer,
                'motion_patterns': motion_patterns
            },
            confidence=0.85,
            metadata={'trajectory_length': len(self.trajectory_buffer)}
        )
    
    def _find_motion_centers(self, motion_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find centers of motion regions"""
        import cv2
        
        contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
        
        return centers
    
    def _analyze_motion_patterns(self, flow: np.ndarray, magnitude: np.ndarray) -> Dict:
        """Analyze overall motion patterns in the scene"""
        # Calculate dominant motion direction
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        
        # Weight by magnitude
        weighted_x = np.sum(flow_x * magnitude) / np.sum(magnitude + 1e-6)
        weighted_y = np.sum(flow_y * magnitude) / np.sum(magnitude + 1e-6)
        
        dominant_angle = np.arctan2(weighted_y, weighted_x) * 180 / np.pi
        
        return {
            'dominant_direction': dominant_angle,
            'motion_coherence': self._calculate_coherence(flow, magnitude),
            'average_speed': np.mean(magnitude),
            'motion_distribution': np.histogram(magnitude.flatten(), bins=20)[0]
        }
    
    def _calculate_coherence(self, flow: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate motion coherence (how aligned the motion vectors are)"""
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Normalize flow vectors
        norm_flow = flow / (magnitude[..., np.newaxis] + 1e-6)
        
        # Calculate coherence as alignment of vectors
        mean_direction = np.mean(norm_flow, axis=(0, 1))
        coherence = np.mean(np.dot(norm_flow.reshape(-1, 2), mean_direction))
        
        return max(0.0, coherence)

class SpatialRelationAnalyzer(ProcessorBase):
    """Analyzes spatial relationships between objects and regions"""
    
    def __init__(self):
        super().__init__("SpatialRelationAnalyzer")
        
    def process(self, segmentation: VisionFeatures, depth: VisionFeatures) -> VisionFeatures:
        if not segmentation.raw_data:
            return VisionFeatures(
                raw_data={'relationships': []},
                confidence=0.0,
                metadata={'status': 'no_segmentation'}
            )
        
        # Analyze spatial relationships
        relationships = []
        
        if 'segment_properties' in segmentation.raw_data:
            segments = segmentation.raw_data['segment_properties']
            
            # Calculate pairwise relationships
            segment_ids = list(segments.keys())
            for i, seg1_id in enumerate(segment_ids):
                for seg2_id in segment_ids[i+1:]:
                    relationship = self._calculate_relationship(
                        segments[seg1_id], 
                        segments[seg2_id],
                        depth.raw_data if depth.raw_data else None
                    )
                    relationships.append({
                        'segment1': seg1_id,
                        'segment2': seg2_id,
                        'relationship': relationship
                    })
        
        return VisionFeatures(
            raw_data={
                'spatial_relationships': relationships,
                'scene_layout': self._analyze_scene_layout(segmentation, depth)
            },
            confidence=0.75,
            metadata={'num_relationships': len(relationships)}
        )
    
    def _calculate_relationship(self, seg1: Dict, seg2: Dict, depth_data: Dict = None) -> Dict:
        """Calculate spatial relationship between two segments"""
        # Distance between centroids
        c1, c2 = seg1['centroid'], seg2['centroid']
        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        # Relative position
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        
        relationship = {
            'distance': distance,
            'relative_position': (dx, dy),
            'size_ratio': seg2['area'] / (seg1['area'] + 1e-6),
            'direction': np.arctan2(dy, dx) * 180 / np.pi
        }
        
        # Add depth relationship if available
        if depth_data and 'depth_map' in depth_data:
            depth_map = depth_data['depth_map']
            seg1_depth = np.mean(depth_map[int(c1[0]):int(c1[0])+5, int(c1[1]):int(c1[1])+5])
            seg2_depth = np.mean(depth_map[int(c2[0]):int(c2[0])+5, int(c2[1]):int(c2[1])+5])
            relationship['depth_ordering'] = 'seg1_closer' if seg1_depth > seg2_depth else 'seg2_closer'
        
        return relationship
    
    def _analyze_scene_layout(self, segmentation: VisionFeatures, depth: VisionFeatures) -> Dict:
        """Analyze overall scene layout"""
        layout = {
            'composition': 'unknown',
            'dominant_regions': [],
            'depth_layers': []
        }
        
        if segmentation.raw_data and 'segment_properties' in segmentation.raw_data:
            segments = segmentation.raw_data['segment_properties']
            
            # Find dominant regions by area
            areas = [(seg_id, props['area']) for seg_id, props in segments.items()]
            areas.sort(key=lambda x: x[1], reverse=True)
            layout['dominant_regions'] = areas[:5]  # Top 5 regions
            
            # Analyze composition (rule of thirds, etc.)
            layout['composition'] = self._analyze_composition(segments)
        
        return layout
    
    def _analyze_composition(self, segments: Dict) -> str:
        """Analyze image composition"""
        # Simplified composition analysis
        centroids = [props['centroid'] for props in segments.values()]
        
        if not centroids:
            return 'empty'
        
        # Check if objects follow rule of thirds
        third_points = [(1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)]
        
        # This is a simplified analysis
        return 'balanced'  # Placeholder

class MidLevelProcessor:
    """Orchestrates all mid-level vision processing"""
    
    def __init__(self):
        self.processors = {
            'segmentation': SegmentationProcessor(),
            'depth': DepthEstimator(),
            'motion': MotionAnalyzer(),
            'spatial': SpatialRelationAnalyzer()
        }
    
    def extract_features(self, image: np.ndarray, low_features: Dict[str, VisionFeatures] = None) -> Dict[str, VisionFeatures]:
        """Extract all mid-level features"""
        features = {}
        
        # Process segmentation
        if self.processors['segmentation'].is_enabled():
            features['segmentation'] = self.processors['segmentation'].process(image, low_features)
        
        # Process depth
        if self.processors['depth'].is_enabled():
            features['depth'] = self.processors['depth'].process(image)
        
        # Process motion (requires optical flow from low level)
        if self.processors['motion'].is_enabled() and low_features and 'motion' in low_features:
            features['motion'] = self.processors['motion'].process(low_features['motion'])
        
        # Process spatial relationships (requires segmentation and depth)
        if self.processors['spatial'].is_enabled():
            seg_feat = features.get('segmentation', VisionFeatures(None, 0.0, {}))
            depth_feat = features.get('depth', VisionFeatures(None, 0.0, {}))
            features['spatial'] = self.processors['spatial'].process(seg_feat, depth_feat)
        
        return features

# =============================================================================
# HIGH LEVEL PROCESSORS - Object recognition and semantic understanding
# =============================================================================

class ObjectDetector(ProcessorBase):
    """Object detection and classification"""
    
    def __init__(self):
        super().__init__("ObjectDetector")
        # In a real implementation, you'd load pre-trained models here
        
    def process(self, image: np.ndarray, mid_features: Dict = None) -> VisionFeatures:
        # Placeholder for object detection
        # In practice, you'd use YOLO, R-CNN, or similar models
        
        detected_objects = {
            'bounding_boxes': [],
            'classifications': [],
            'confidence_scores': [],
            'object_count': 0
        }
        
        # Simulate object detection based on segmentation if available
        if mid_features and 'segmentation' in mid_features:
            seg_data = mid_features['segmentation'].raw_data
            if seg_data and 'segment_properties' in seg_data:
                segments = seg_data['segment_properties']
                
                for seg_id, props in list(segments.items())[:5]:  # Process top 5 segments
                    bbox = props.get('bbox', (0, 0, 0, 0))
                    area = props.get('area', 0)
                    
                    if area > 1000:  # Filter small objects
                        detected_objects['bounding_boxes'].append(bbox)
                        detected_objects['classifications'].append(self._classify_segment(props))
                        detected_objects['confidence_scores'].append(0.7)
                
                detected_objects['object_count'] = len(detected_objects['bounding_boxes'])
        
        return VisionFeatures(
            raw_data=detected_objects,
            confidence=0.8,
            metadata={'method': 'segment_based_detection'}
        )
    
    def _classify_segment(self, segment_props: Dict) -> str:
        """Classify a segment based on its properties"""
        area = segment_props.get('area', 0)
        
        # Simple heuristic classification
        if area > 10000:
            return 'large_object'
        elif area > 5000:
            return 'medium_object'
        else:
            return 'small_object'

class FaceDetector(ProcessorBase):
    """Face detection and analysis"""
    
    def __init__(self):
        super().__init__("FaceDetector")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        
        # Load OpenCV's face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_data = {
                'faces': faces.tolist() if len(faces) > 0 else [],
                'face_count': len(faces),
                'face_properties': []
            }
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_data['face_properties'].append({
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'aspect_ratio': w / h,
                    'brightness': np.mean(face_roi)
                })
            
        except Exception as e:
            face_data = {
                'faces': [],
                'face_count': 0,
                'error': str(e)
            }
        
        return VisionFeatures(
            raw_data=face_data,
            confidence=0.85 if face_data['face_count'] > 0 else 0.2,
            metadata={'method': 'haar_cascade'}
        )

class TextDetector(ProcessorBase):
    """Text detection and OCR"""
    
    def __init__(self):
        super().__init__("TextDetector")
        
    def process(self, image: np.ndarray) -> VisionFeatures:
        import cv2
        
        try:
            # Use EAST text detector if available, fallback to simple method
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Simple text region detection using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            
            _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 10 and w > 2 * h:  # Filter for text-like regions
                    text_regions.append((x, y, w, h))
            
            text_data = {
                'text_regions': text_regions,
                'text_count': len(text_regions),
                'extracted_text': []  # Placeholder for OCR results
            }
            
        except Exception as e:
            text_data = {
                'text_regions': [],
                'text_count': 0,
                'error': str(e)
            }
        
        return VisionFeatures(
            raw_data=text_data,
            confidence=0.7 if text_data['text_count'] > 0 else 0.1,
            metadata={'method': 'morphological_text_detection'}
        )

class HighLevelProcessor:
    """Orchestrates all high-level vision processing"""
    
    def __init__(self):
        self.processors = {
            'objects': ObjectDetector(),
            'faces': FaceDetector(),
            'text': TextDetector(),
            'emotions': EmotionRecognizer(),
            'activities': ActivityRecognizer()
        }
    
    def extract_features(self, image: np.ndarray, mid_features: Dict[str, VisionFeatures] = None) -> Dict[str, VisionFeatures]:
        """Extract all high-level features"""
        features = {}
        
        for name, processor in self.processors.items():
            if processor.is_enabled():
                try:
                    if name == 'objects':
                        features[name] = processor.process(image, mid_features)
                    elif name == 'emotions':
                        # Emotions require face detection
                        face_features = features.get('faces')
                        if not face_features:
                            face_features = self.processors['faces'].process(image)
                        features[name] = processor.process(image, face_features)
                    elif name == 'activities':
                        # Activities use all available features
                        features[name] = processor.process(image, mid_features, features)
                    else:
                        features[name] = processor.process(image)
                except Exception as e:
                    print(f"Error in {name} processor: {e}")
                    features[name] = VisionFeatures(
                        raw_data=None,
                        confidence=0.0,
                        metadata={'error': str(e)}
                    )
        
        return features

class EmotionRecognizer(ProcessorBase):
    """Emotion recognition from facial expressions"""
    
    def __init__(self):
        super().__init__("EmotionRecognizer")
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']
        
    def process(self, image: np.ndarray, face_features: VisionFeatures) -> VisionFeatures:
        emotion_data = {
            'emotions': [],
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0
        }
        
        if not face_features.raw_data or face_features.raw_data['face_count'] == 0:
            return VisionFeatures(
                raw_data=emotion_data,
                confidence=0.0,
                metadata={'status': 'no_faces_detected'}
            )
        
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        for face_props in face_features.raw_data['face_properties']:
            x, y, w, h = face_props['bbox']
            face_roi = gray[y:y+h, x:x+w]
            
            # Simple emotion estimation based on facial features
            emotion_scores = self._analyze_facial_features(face_roi)
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            emotion_data['emotions'].append({
                'bbox': (x, y, w, h),
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion]
            })
        
        # Overall dominant emotion
        if emotion_data['emotions']:
            all_scores = {}
            for emotion in self.emotions:
                all_scores[emotion] = np.mean([e['emotion_scores'][emotion] for e in emotion_data['emotions']])
            
            emotion_data['dominant_emotion'] = max(all_scores, key=all_scores.get)
            emotion_data['emotion_confidence'] = all_scores[emotion_data['dominant_emotion']]
        
        return VisionFeatures(
            raw_data=emotion_data,
            confidence=emotion_data['emotion_confidence'],
            metadata={'num_faces': len(emotion_data['emotions'])}
        )
    
    def _analyze_facial_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze facial features for emotion recognition"""
        # Simplified emotion analysis based on image properties
        brightness = np.mean(face_roi)
        contrast = np.std(face_roi)
        
        # Simple heuristic-based emotion scores
        emotion_scores = {
            'neutral': 0.4,
            'happy': min(0.8, brightness / 128.0),
            'sad': max(0.1, 1.0 - brightness / 128.0),
            'angry': min(0.7, contrast / 50.0),
            'surprised': min(0.6, contrast / 40.0),
            'fear': 0.2,
            'disgust': 0.1
        }
        
        # Normalize scores
        total = sum(emotion_scores.values())
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores

class ActivityRecognizer(ProcessorBase):
    """Recognizes activities and actions in the scene"""
    
    def __init__(self):
        super().__init__("ActivityRecognizer")
        self.activities = ['sitting', 'standing', 'walking', 'running', 'interacting', 'working', 'relaxing']
        
    def process(self, image: np.ndarray, mid_features: Dict = None, high_features: Dict = None) -> VisionFeatures:
        activity_data = {
            'detected_activities': [],
            'scene_activity_level': 0.0,
            'primary_activity': 'unknown'
        }
        
        # Analyze motion for activity recognition
        motion_level = 0.0
        if mid_features and 'motion' in mid_features:
            motion_data = mid_features['motion'].raw_data
            if motion_data and 'motion_patterns' in motion_data:
                motion_level = motion_data['motion_patterns'].get('average_speed', 0.0)
        
        # Analyze objects for context
        object_context = []
        if high_features and 'objects' in high_features:
            object_data = high_features['objects'].raw_data
            if object_data:
                object_context = object_data.get('classifications', [])
        
        # Analyze faces for social context
        face_count = 0
        if high_features and 'faces' in high_features:
            face_data = high_features['faces'].raw_data
            if face_data:
                face_count = face_data.get('face_count', 0)
        
        # Simple activity classification based on available features
        activity_scores = self._classify_activities(motion_level, object_context, face_count)
        
        activity_data['detected_activities'] = activity_scores
        activity_data['scene_activity_level'] = motion_level
        activity_data['primary_activity'] = max(activity_scores, key=activity_scores.get) if activity_scores else 'unknown'
        
        return VisionFeatures(
            raw_data=activity_data,
            confidence=0.6,
            metadata={'motion_level': motion_level, 'face_count': face_count}
        )
    
    def _classify_activities(self, motion_level: float, object_context: List, face_count: int) -> Dict[str, float]:
        """Classify activities based on scene features"""
        activity_scores = {}
        
        # Motion-based activity classification
        if motion_level > 10:
            activity_scores['running'] = 0.8
            activity_scores['walking'] = 0.6
        elif motion_level > 5:
            activity_scores['walking'] = 0.7
            activity_scores['standing'] = 0.4
        elif motion_level > 1:
            activity_scores['standing'] = 0.6
            activity_scores['interacting'] = 0.5
        else:
            activity_scores['sitting'] = 0.7
            activity_scores['relaxing'] = 0.6
        
        # Social context
        if face_count > 1:
            activity_scores['interacting'] = activity_scores.get('interacting', 0.0) + 0.3
        
        # Object context (simplified)
        if 'large_object' in object_context:
            activity_scores['working'] = activity_scores.get('working', 0.0) + 0.2
        
        # Normalize scores
        if activity_scores:
            total = sum(activity_scores.values())
            activity_scores = {k: v/total for k, v in activity_scores.items()}
        
        return activity_scores

# =============================================================================
# ENHANCED BEHAVIOR AND ENVIRONMENT MODULES
# =============================================================================

class BehaviorRecognizer:
    """Enhanced behavior recognition with more sophisticated analysis"""
    
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
        self.gesture_recognizer = GestureRecognizer()
        self.group_behavior_analyzer = GroupBehaviorAnalyzer()
        self.threat_detector = ThreatDetector()
        
    def recognize(self, mid_features: Dict[str, VisionFeatures], high_features: Dict[str, VisionFeatures]) -> Dict:
        """Comprehensive behavior analysis"""
        
        # Individual behavior analysis
        individual_behaviors = self.pose_analyzer.analyze_poses(mid_features, high_features)
        
        # Gesture recognition
        gestures = self.gesture_recognizer.recognize_gestures(mid_features, high_features)
        
        # Group behavior analysis
        group_behaviors = self.group_behavior_analyzer.analyze_group_dynamics(mid_features, high_features)
        
        # Threat assessment
        threat_assessment = self.threat_detector.assess_threats(mid_features, high_features)
        
        return {
            'individual_behaviors': individual_behaviors,
            'gestures': gestures,
            'group_behaviors': group_behaviors,
            'threat_assessment': threat_assessment,
            'overall_behavior_score': self._calculate_overall_score(individual_behaviors, group_behaviors, threat_assessment)
        }
    
    def _calculate_overall_score(self, individual: Dict, group: Dict, threat: Dict) -> Dict:
        """Calculate overall behavior scores"""
        return {
            'activity_level': np.mean([individual.get('activity_level', 0), group.get('activity_level', 0)]),
            'social_engagement': group.get('social_engagement', 0),
            'threat_level': threat.get('threat_level', 0),
            'attention_focus': individual.get('attention_focus', 0)
        }

class PoseAnalyzer(ProcessorBase):
    """Human pose analysis and body language interpretation"""
    
    def __init__(self):
        super().__init__("PoseAnalyzer")
        
    def analyze_poses(self, mid_features: Dict, high_features: Dict) -> Dict:
        """Analyze human poses and body language"""
        
        pose_data = {
            'detected_poses': [],
            'body_language_indicators': {},
            'activity_level': 0.0,
            'attention_focus': 0.0
        }
        
        # Use face and motion data for pose estimation
        if high_features and 'faces' in high_features:
            face_data = high_features['faces'].raw_data
            if face_data and face_data['face_count'] > 0:
                
                # Simplified pose analysis based on face orientation and motion
                for face_props in face_data.get('face_properties', []):
                    pose_analysis = self._analyze_single_pose(face_props, mid_features)
                    pose_data['detected_poses'].append(pose_analysis)
                
                # Overall activity level
                if mid_features and 'motion' in mid_features:
                    motion_data = mid_features['motion'].raw_data
                    if motion_data:
                        pose_data['activity_level'] = motion_data.get('motion_patterns', {}).get('average_speed', 0.0) / 10.0
        
        return pose_data
    
    def _analyze_single_pose(self, face_props: Dict, mid_features: Dict) -> Dict:
        """Analyze a single person's pose"""
        bbox = face_props['bbox']
        
        # Simplified pose analysis
        pose_analysis = {
            'bbox': bbox,
            'head_orientation': self._estimate_head_orientation(face_props),
            'body_posture': self._estimate_body_posture(bbox, mid_features),
            'engagement_level': face_props.get('brightness', 0) / 255.0
        }
        
        return pose_analysis
    
    def _estimate_head_orientation(self, face_props: Dict) -> str:
        """Estimate head orientation from face properties"""
        aspect_ratio = face_props.get('aspect_ratio', 1.0)
        
        if aspect_ratio > 1.2:
            return 'turned_left'
        elif aspect_ratio < 0.8:
            return 'turned_right'
        else:
            return 'facing_forward'
    
    def _estimate_body_posture(self, face_bbox: Tuple, mid_features: Dict) -> str:
        """Estimate body posture from available features"""
        # Simplified posture estimation
        x, y, w, h = face_bbox
        
        # Use relative position in frame
        if y < 100:  # Face in upper part of frame
            return 'standing'
        elif y > 300:  # Face in lower part of frame
            return 'sitting'
        else:
            return 'neutral'

class GestureRecognizer(ProcessorBase):
    """Hand gesture and body gesture recognition"""
    
    def __init__(self):
        super().__init__("GestureRecognizer")
        self.gesture_buffer = []
        self.buffer_size = 10
        
    def recognize_gestures(self, mid_features: Dict, high_features: Dict) -> Dict:
        """Recognize hand and body gestures"""
        
        gesture_data = {
            'detected_gestures': [],
            'gesture_sequence': [],
            'confidence': 0.0
        }
        
        # Use motion and segmentation data for gesture recognition
        if mid_features and 'motion' in mid_features:
            motion_data = mid_features['motion'].raw_data
            
            if motion_data and 'motion_centers' in motion_data:
                motion_centers = motion_data['motion_centers']
                
                # Analyze motion patterns for gestures
                gestures = self._analyze_motion_for_gestures(motion_centers)
                gesture_data['detected_gestures'] = gestures
                
                # Update gesture sequence buffer
                self.gesture_buffer.append(gestures)
                if len(self.gesture_buffer) > self.buffer_size:
                    self.gesture_buffer.pop(0)
                
                gesture_data['gesture_sequence'] = self.gesture_buffer
        
        return gesture_data
    
    def _analyze_motion_for_gestures(self, motion_centers: List) -> List[Dict]:
        """Analyze motion centers for gesture patterns"""
        gestures = []
        
        for center in motion_centers:
            # Simplified gesture classification based on motion center properties
            gesture = {
                'type': 'hand_movement',
                'location': center,
                'confidence': 0.6,
                'gesture_class': self._classify_gesture_by_location(center)
            }
            gestures.append(gesture)
        
        return gestures
    
    def _classify_gesture_by_location(self, center: Tuple[int, int]) -> str:
        """Classify gesture based on location in frame"""
        x, y = center
        
        # Simple location-based gesture classification
        if y < 200:  # Upper part of frame
            return 'raising_hand'
        elif x < 200:  # Left side
            return 'pointing_left'
        elif x > 400:  # Right side
            return 'pointing_right'
        else:
            return 'general_movement'

class GroupBehaviorAnalyzer(ProcessorBase):
    """Group dynamics and crowd behavior analysis"""
    
    def __init__(self):
        super().__init__("GroupBehaviorAnalyzer")
        
    def analyze_group_dynamics(self, mid_features: Dict, high_features: Dict) -> Dict:
        """Analyze group behavior and social dynamics"""
        
        group_data = {
            'group_size': 0,
            'social_engagement': 0.0,
            'group_cohesion': 0.0,
            'activity_level': 0.0,
            'interaction_patterns': []
        }
        
        # Count people in scene
        if high_features and 'faces' in high_features:
            face_data = high_features['faces'].raw_data
            if face_data:
                group_data['group_size'] = face_data.get('face_count', 0)
        
        # Analyze group cohesion using spatial relationships
        if mid_features and 'spatial' in mid_features:
            spatial_data = mid_features['spatial'].raw_data
            if spatial_data and 'spatial_relationships' in spatial_data:
                relationships = spatial_data['spatial_relationships']
                group_data['group_cohesion'] = self._calculate_group_cohesion(relationships)
        
        # Analyze group activity level
        if mid_features and 'motion' in mid_features:
            motion_data = mid_features['motion'].raw_data
            if motion_data:
                group_data['activity_level'] = motion_data.get('motion_patterns', {}).get('average_speed', 0.0) / 10.0
        
        # Social engagement analysis
        group_data['social_engagement'] = self._calculate_social_engagement(group_data)
        
        return group_data
    
    def _calculate_group_cohesion(self, relationships: List[Dict]) -> float:
        """Calculate how cohesive the group is based on spatial relationships"""
        if not relationships:
            return 0.0
        
        # Simplified cohesion calculation based on average distances
        distances = [rel['relationship']['distance'] for rel in relationships]
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 range (closer = more cohesive)
        cohesion = max(0.0, 1.0 - avg_distance / 500.0)
        return cohesion
    
    def _calculate_social_engagement(self, group_data: Dict) -> float:
        """Calculate social engagement level"""
        group_size = group_data['group_size']
        activity_level = group_data['activity_level']
        cohesion = group_data['group_cohesion']
        
        if group_size < 2:
            return 0.0
        
        # Engagement increases with group size, activity, and cohesion
        engagement = (group_size / 10.0) * 0.3 + activity_level * 0.4 + cohesion * 0.3
        return min(1.0, engagement)

class ThreatDetector(ProcessorBase):
    """Threat and anomaly detection"""
    
    def __init__(self):
        super().__init__("ThreatDetector")
        self.baseline_activity = 0.0
        self.frame_count = 0
        
    def assess_threats(self, mid_features: Dict, high_features: Dict) -> Dict:
        """Assess potential threats and anomalies"""
        
        threat_data = {
            'threat_level': 0.0,
            'anomaly_score': 0.0,
            'threat_indicators': [],
            'risk_factors': []
        }
        
        # Analyze motion patterns for threats
        if mid_features and 'motion' in mid_features:
            motion_data = mid_features['motion'].raw_data
            if motion_data:
                motion_threat = self._analyze_motion_threats(motion_data)
                threat_data['threat_level'] += motion_threat * 0.4
                
                if motion_threat > 0.5:
                    threat_data['threat_indicators'].append('unusual_motion_pattern')
        
        # Analyze group behavior for threats
        if high_features and 'faces' in high_features:
            face_data = high_features['faces'].raw_data
            if face_data:
                crowd_threat = self._analyze_crowd_threats(face_data)
                threat_data['threat_level'] += crowd_threat * 0.3
                
                if crowd_threat > 0.5:
                    threat_data['threat_indicators'].append('unusual_crowd_behavior')
        
        # Analyze emotional indicators
        if high_features and 'emotions' in high_features:
            emotion_data = high_features['emotions'].raw_data
            if emotion_data:
                emotion_threat = self._analyze_emotion_threats(emotion_data)
                threat_data['threat_level'] += emotion_threat * 0.3
                
                if emotion_threat > 0.5:
                    threat_data['threat_indicators'].append('negative_emotions_detected')
        
        # Normalize threat level
        threat_data['threat_level'] = min(1.0, threat_data['threat_level'])
        
        # Calculate anomaly score
        threat_data['anomaly_score'] = self._calculate_anomaly_score(mid_features, high_features)
        
        return threat_data
    
    def _analyze_motion_threats(self, motion_data: Dict) -> float:
        """Analyze motion patterns for threat indicators"""
        threat_score = 0.0
        
        patterns = motion_data.get('motion_patterns', {})
        avg_speed = patterns.get('average_speed', 0.0)
        coherence = patterns.get('motion_coherence', 0.0)
        
        # High speed movement
        if avg_speed > 15:
            threat_score += 0.3
        
        # Erratic movement (low coherence)
        if coherence < 0.3:
            threat_score += 0.4
        
        # Sudden motion changes
        if hasattr(self, 'prev_motion_data'):
            prev_speed = self.prev_motion_data.get('motion_patterns', {}).get('average_speed', 0.0)
            speed_change = abs(avg_speed - prev_speed)
            if speed_change > 10:
                threat_score += 0.2
        
        self.prev_motion_data = motion_data
        
        return min(1.0, threat_score)
    
    def _analyze_crowd_threats(self, face_data: Dict) -> float:
        """Analyze crowd behavior for threat indicators"""
        threat_score = 0.0
        
        face_count = face_data.get('face_count', 0)
        
        # Unusual crowd density
        if face_count > 10:
            threat_score += 0.3
        elif face_count == 0:
            threat_score += 0.1  # Empty area might be suspicious
        
        return min(1.0, threat_score)
    
    def _analyze_emotion_threats(self, emotion_data: Dict) -> float:
        """Analyze emotional indicators for threats"""
        threat_score = 0.0
        
        emotions = emotion_data.get('emotions', [])
        
        for emotion_info in emotions:
            emotion_scores = emotion_info.get('emotion_scores', {})
            
            # Negative emotions increase threat score
            if emotion_scores.get('angry', 0) > 0.6:
                threat_score += 0.4
            if emotion_scores.get('fear', 0) > 0.6:
                threat_score += 0.3
            if emotion_scores.get('disgust', 0) > 0.5:
                threat_score += 0.2
        
        return min(1.0, threat_score)
    
    def _calculate_anomaly_score(self, mid_features: Dict, high_features: Dict) -> float:
        """Calculate overall anomaly score"""
        # Update baseline activity
        current_activity = 0.0
        
        if mid_features and 'motion' in mid_features:
            motion_data = mid_features['motion'].raw_data
            if motion_data:
                current_activity = motion_data.get('motion_patterns', {}).get('average_speed', 0.0)
        
        # Update running average
        self.frame_count += 1
        alpha = 0.1  # Learning rate
        self.baseline_activity = (1 - alpha) * self.baseline_activity + alpha * current_activity
        
        # Calculate anomaly as deviation from baseline
        if self.baseline_activity > 0:
            anomaly = abs(current_activity - self.baseline_activity) / (self.baseline_activity + 1e-6)
            return min(1.0, anomaly)
        
        return 0.0

# =============================================================================
# ENHANCED ENVIRONMENT ANALYZER
# =============================================================================

class EnvironmentAnalyzer:
    """Enhanced environmental analysis with more sophisticated features"""
    
    def __init__(self):
        self.room_analyzer = RoomAnalyzer()
        self.lighting_analyzer = LightingAnalyzer()
        self.population_analyzer = PopulationAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        
    def analyze(self, image: np.ndarray, infrared: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive environmental analysis"""
        
        # Room structure analysis
        room_analysis = self.room_analyzer.analyze_room(image)
        
        # Lighting conditions
        lighting_analysis = self.lighting_analyzer.analyze_lighting(image)
        
        # Population and occupancy
        population_analysis = self.population_analyzer.analyze_population(image, infrared)
        
        # Context and setting
        context_analysis = self.context_analyzer.analyze_context(image, room_analysis, lighting_analysis)
        
        return {
            'room_structure': room_analysis,
            'lighting_conditions': lighting_analysis,
            'population_analysis': population_analysis,
            'context_analysis': context_analysis,
            'environment_score': self._calculate_environment_score(room_analysis, lighting_analysis, population_analysis)
        }
    
    def _calculate_environment_score(self, room: Dict, lighting: Dict, population: Dict) -> Dict:
        """Calculate overall environment scores"""
        return {
            'comfort_level': (lighting.get('comfort_score', 0.5) + room.get('spaciousness', 0.5)) / 2.0,
            'activity_suitability': population.get('density_score', 0.5),
            'visibility_quality': lighting.get('visibility_score', 0.5),
            'safety_assessment': room.get('safety_score', 0.5)
        }

class RoomAnalyzer(ProcessorBase):
    """Room structure and layout analysis"""
    
    def __init__(self):
        super().__init__("RoomAnalyzer")
        
    def analyze_room(self, image: np.ndarray) -> Dict:
        """Analyze room structure and layout"""
        import cv2
        
        room_data = {
            'room_type': 'unknown',
            'estimated_size': 'medium',
            'layout_features': [],
            'spaciousness': 0.5,
            'safety_score': 0.5
        }
        
        # Convert to grayscale for structural analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect lines for room structure
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            room_data['layout_features'] = self._analyze_lines(lines)
            room_data['room_type'] = self._classify_room_type(lines, image)
        
        # Estimate room size from perspective
        room_data['estimated_size'] = self._estimate_room_size(gray)
        room_data['spaciousness'] = self._calculate_spaciousness(gray)
        
        return room_data
    
    def _analyze_lines(self, lines: np.ndarray) -> List[str]:
        """Analyze detected lines for room features"""
        features = []
        
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines += 1
            elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
                vertical_lines += 1
        
        if horizontal_lines > 5:
            features.append('horizontal_structures')
        if vertical_lines > 5:
            features.append('vertical_structures')
        if horizontal_lines > 3 and vertical_lines > 3:
            features.append('rectangular_layout')
        
        return features
    
    def _classify_room_type(self, lines: np.ndarray, image: np.ndarray) -> str:
        """Classify room type based on structural features"""
        # Simplified room classification
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1.5:
            return 'corridor_or_hallway'
        elif aspect_ratio < 0.8:
            return 'narrow_room'
        else:
            return 'standard_room'
    
    def _estimate_room_size(self, gray: np.ndarray) -> str:
        """Estimate room size from image characteristics"""
        # Use edge density and gradient information
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            return 'small'
        elif edge_density > 0.05:
            return 'medium'
        else:
            return 'large'
    
    def _calculate_spaciousness(self, gray: np.ndarray) -> float:
        """Calculate spaciousness score"""
        # Use variance and edge density as spaciousness indicators
        variance = np.var(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher variance and lower edge density suggest more spacious rooms
        spaciousness = (variance / 1000.0) * 0.6 + (1.0 - edge_density * 20) * 0.4
        return max(0.0, min(1.0, spaciousness))

class LightingAnalyzer(ProcessorBase):
    """Lighting conditions analysis"""
    
    def __init__(self):
        super().__init__("LightingAnalyzer")
        
    def analyze_lighting(self, image: np.ndarray) -> Dict:
        """Analyze lighting conditions"""
        import cv2
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_
