import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import tempfile
import base64

logger = logging.getLogger(__name__)

class PhotoshopController:
    """Controller for automating Adobe Photoshop through ExtendScript."""
    
    def __init__(self, photoshop_path: str = None):
        """
        Initialize Photoshop controller.
        
        Args:
            photoshop_path: Path to Photoshop executable
        """
        self.photoshop_path = photoshop_path or self._find_photoshop_installation()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ps_controller_"))
        self.jsx_bridge_path = self._setup_jsx_bridge()
        
    def __del__(self):
        """Clean up resources."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _find_photoshop_installation(self) -> str:
        """Find Photoshop installation automatically."""
        common_paths = [
            "C:/Program Files/Adobe/Adobe Photoshop 2023/Photoshop.exe",
            "C:/Program Files/Adobe/Adobe Photoshop 2022/Photoshop.exe",
            "C:/Program Files/Adobe/Adobe Photoshop 2021/Photoshop.exe",
            "/Applications/Adobe Photoshop 2023/Adobe Photoshop 2023.app/Contents/MacOS/Adobe Photoshop 2023",
            "/Applications/Adobe Photoshop 2022/Adobe Photoshop 2022.app/Contents/MacOS/Adobe Photoshop 2022"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find Photoshop in registry (Windows)
        if os.name == 'nt':
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Adobe\Photoshop") as key:
                    version = winreg.EnumKey(key, 0)  # Get the first version
                    with winreg.OpenKey(key, version) as version_key:
                        install_path, _ = winreg.QueryValueEx(version_key, "ApplicationPath")
                        return os.path.join(install_path, "Photoshop.exe")
            except:
                pass
        
        raise FileNotFoundError("Photoshop installation not found")
    
    def _setup_jsx_bridge(self) -> Path:
        """Set up JSX bridge script for communication with Photoshop."""
        jsx_bridge_path = self.temp_dir / "ps_bridge.jsx"
        
        jsx_content = """
// Photoshop JSX Bridge Script
(function() {
    // Function to write output to file
    function writeFile(filePath, content) {
        var file = new File(filePath);
        file.encoding = "UTF-8";
        file.open("w");
        file.write(content);
        file.close();
    }
    
    // Function to read input from file
    function readFile(filePath) {
        var file = new File(filePath);
        file.encoding = "UTF-8";
        file.open("r");
        var content = file.read();
        file.close();
        return content;
    }
    
    // Function to execute a command
    function executeCommand(command) {
        try {
            // Parse command
            var cmd = JSON.parse(command);
            var action = cmd.action;
            var params = cmd.params || {};
            var result = { success: false };
            
            // Execute action
            switch (action) {
                case "create_document":
                    result = createDocument(params);
                    break;
                case "open_document":
                    result = openDocument(params);
                    break;
                case "save_document":
                    result = saveDocument(params);
                    break;
                case "create_layer":
                    result = createLayer(params);
                    break;
                case "add_text":
                    result = addText(params);
                    break;
                case "add_shape":
                    result = addShape(params);
                    break;
                case "apply_filter":
                    result = applyFilter(params);
                    break;
                case "apply_adjustment":
                    result = applyAdjustment(params);
                    break;
                case "resize_image":
                    result = resizeImage(params);
                    break;
                case "crop_image":
                    result = cropImage(params);
                    break;
                case "execute_script":
                    result = executeScript(params);
                    break;
                default:
                    result = { success: false, error: "Unknown action: " + action };
            }
            
            return JSON.stringify(result);
        } catch (e) {
            return JSON.stringify({ success: false, error: e.toString() });
        }
    }
    
    // Create a new document
    function createDocument(params) {
        try {
            var name = params.name || "Untitled";
            var width = params.width || 1920;
            var height = params.height || 1080;
            var resolution = params.resolution || 72;
            var mode = params.mode || "RGB";
            var backgroundColor = params.backgroundColor || "white";
            
            // Create document
            var doc = app.documents.add(width, height, resolution, name, NewDocumentMode[mode], DocumentFill[backgroundColor]);
            
            return {
                success: true,
                documentId: app.activeDocument.id,
                name: doc.name,
                width: doc.width.value,
                height: doc.height.value
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Open an existing document
    function openDocument(params) {
        try {
            var filePath = params.path;
            
            // Open document
            var doc = app.open(File(filePath));
            
            return {
                success: true,
                documentId: app.activeDocument.id,
                name: doc.name,
                width: doc.width.value,
                height: doc.height.value
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Save document
    function saveDocument(params) {
        try {
            var filePath = params.path;
            var format = params.format || "JPEG";
            var quality = params.quality || 12;
            
            // Get active document
            var doc = app.activeDocument;
            
            // Save based on format
            if (format === "JPEG") {
                var jpegOptions = new JPEGSaveOptions();
                jpegOptions.quality = quality;
                doc.saveAs(File(filePath), jpegOptions);
            } else if (format === "PNG") {
                var pngOptions = new PNGSaveOptions();
                doc.saveAs(File(filePath), pngOptions);
            } else if (format === "PSD") {
                var psdOptions = new PhotoshopSaveOptions();
                doc.saveAs(File(filePath), psdOptions);
            } else if (format === "TIFF") {
                var tiffOptions = new TiffSaveOptions();
                doc.saveAs(File(filePath), tiffOptions);
            } else {
                throw new Error("Unsupported format: " + format);
            }
            
            return {
                success: true,
                path: filePath,
                format: format
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create a new layer
    function createLayer(params) {
        try {
            var name = params.name || "New Layer";
            
            // Get active document
            var doc = app.activeDocument;
            
            // Create layer
            var layer = doc.artLayers.add();
            layer.name = name;
            
            return {
                success: true,
                layerId: layer.id,
                name: layer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Add text to document
    function addText(params) {
        try {
            var text = params.text || "Text";
            var position = params.position || [100, 100];
            var fontSize = params.fontSize || 72;
            var color = params.color || [0, 0, 0];
            var font = params.font || "Arial";
            
            // Get active document
            var doc = app.activeDocument;
            
            // Create text layer
            var layer = doc.artLayers.add();
            layer.kind = LayerKind.TEXT;
            layer.name = text.substring(0, 20) + "...";
            
            // Set text properties
            var textItem = layer.textItem;
            textItem.contents = text;
            textItem.position = [position[0], position[1]];
            textItem.size = fontSize;
            
            // Set color
            var colorObj = new SolidColor();
            colorObj.rgb.red = color[0];
            colorObj.rgb.green = color[1];
            colorObj.rgb.blue = color[2];
            textItem.color = colorObj;
            
            // Set font
            if (font) {
                textItem.font = font;
            }
            
            return {
                success: true,
                layerId: layer.id,
                name: layer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Add shape to document
    function addShape(params) {
        try {
            var shapeType = params.shapeType || "rectangle";
            var position = params.position || [100, 100];
            var size = params.size || [500, 300];
            var color = params.color || [255, 0, 0];
            
            // Get active document
            var doc = app.activeDocument;
            
            // Create selection based on shape type
            if (shapeType === "rectangle") {
                var x1 = position[0];
                var y1 = position[1];
                var x2 = x1 + size[0];
                var y2 = y1 + size[1];
                
                var region = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
                doc.selection.select(region);
            } else if (shapeType === "ellipse") {
                var x = position[0] + size[0] / 2;
                var y = position[1] + size[1] / 2;
                
                doc.selection.ellipse([position[0], position[1], position[0] + size[0], position[1] + size[1]]);
            } else {
                throw new Error("Unsupported shape type: " + shapeType);
            }
            
            // Fill selection with color
            var colorObj = new SolidColor();
            colorObj.rgb.red = color[0];
            colorObj.rgb.green = color[1];
            colorObj.rgb.blue = color[2];
            
            doc.selection.fill(colorObj);
            doc.selection.deselect();
            
            // Get the created layer
            var layer = doc.activeLayer;
            
            return {
                success: true,
                layerId: layer.id,
                name: layer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Apply filter to layer
    function applyFilter(params) {
        try {
            var filterType = params.filterType || "GaussianBlur";
            var options = params.options || {};
            
            // Get active document and layer
            var doc = app.activeDocument;
            var layer = doc.activeLayer;
            
            // Apply filter based on type
            if (filterType === "GaussianBlur") {
                var radius = options.radius || 10;
                app.doAction("GaussianBlur", "Filter Gallery");
                // In a real implementation, you would use the appropriate filter method
                // For example: app.activeDocument.activeLayer.applyGaussianBlur(radius);
            } else if (filterType === "Sharpen") {
                app.doAction("Sharpen", "Filter Gallery");
            } else if (filterType === "MotionBlur") {
                var angle = options.angle || 0;
                var distance = options.distance || 10;
                app.doAction("MotionBlur", "Filter Gallery");
            } else {
                throw new Error("Unsupported filter type: " + filterType);
            }
            
            return {
                success: true,
                filterType: filterType,
                layerId: layer.id
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Apply adjustment to layer
    function applyAdjustment(params) {
        try {
            var adjustmentType = params.adjustmentType || "Levels";
            var options = params.options || {};
            
            // Get active document
            var doc = app.activeDocument;
            
            // Create adjustment layer
            var layer = doc.artLayers.add();
            layer.kind = LayerKind.ADJUSTMENT;
            layer.name = adjustmentType + " Adjustment";
            
            // Apply adjustment based on type
            if (adjustmentType === "Levels") {
                // In a real implementation, you would set the adjustment properties
                // For example: layer.adjustmentLayer.levels = ...
            } else if (adjustmentType === "Curves") {
                // In a real implementation, you would set the adjustment properties
            } else if (adjustmentType === "HueSaturation") {
                var hue = options.hue || 0;
                var saturation = options.saturation || 0;
                var lightness = options.lightness || 0;
                // In a real implementation, you would set the adjustment properties
            } else {
                throw new Error("Unsupported adjustment type: " + adjustmentType);
            }
            
            return {
                success: true,
                adjustmentType: adjustmentType,
                layerId: layer.id,
                name: layer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Resize image
    function resizeImage(params) {
        try {
            var width = params.width;
            var height = params.height;
            var resampleMethod = params.resampleMethod || "bicubic";
            
            // Get active document
            var doc = app.activeDocument;
            
            // Resize image
            doc.resizeImage(
                UnitValue(width, "px"),
                UnitValue(height, "px"),
                doc.resolution,
                ResampleMethod[resampleMethod]
            );
            
            return {
                success: true,
                width: doc.width.value,
                height: doc.height.value
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Crop image
    function cropImage(params) {
        try {
            var bounds = params.bounds || [0, 0, 500, 500];
            
            // Get active document
            var doc = app.activeDocument;
            
            // Crop image
            doc.crop(bounds);
            
            return {
                success: true,
                width: doc.width.value,
                height: doc.height.value,
                bounds: bounds
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Execute custom script
    function executeScript(params) {
        try {
            var script = params.script;
            var result = eval(script);
            
            return {
                success: true,
                result: result
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Main execution
    var inputFile = new File(arguments[0]);
    var outputFile = new File(arguments[1]);
    
    if (inputFile.exists) {
        var command = readFile(inputFile.fsName);
        var result = executeCommand(command);
        writeFile(outputFile.fsName, result);
    } else {
        writeFile(outputFile.fsName, JSON.stringify({ success: false, error: "Input file not found" }));
    }
})();
"""
        
        with open(jsx_bridge_path, 'w') as f:
            f.write(jsx_content)
        
        return jsx_bridge_path
    
    def execute_jsx(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JSX command in Photoshop."""
        try:
            # Create input and output files
            input_file = self.temp_dir / f"ps_input_{time.time()}.json"
            output_file = self.temp_dir / f"ps_output_{time.time()}.json"
            
            # Write command to input file
            with open(input_file, 'w') as f:
                json.dump(command, f)
            
            # Build command
            cmd = [
                self.photoshop_path,
                "-r",
                str(self.jsx_bridge_path),
                str(input_file),
                str(output_file)
            ]
            
            # Execute command
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Read result from output file
            if output_file.exists():
                with open(output_file, 'r') as f:
                    result = json.load(f)
            else:
                result = {"success": False, "error": "Output file not created"}
            
            # Clean up
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute JSX command: {e}")
            return {"success": False, "error": str(e)}
    
    def create_document(self, name: str, width: int = 1920, height: int = 1080, resolution: int = 72, mode: str = "RGB", background_color: str = "white") -> Dict[str, Any]:
        """Create a new document in Photoshop."""
        command = {
            "action": "create_document",
            "params": {
                "name": name,
                "width": width,
                "height": height,
                "resolution": resolution,
                "mode": mode,
                "backgroundColor": background_color
            }
        }
        return self.execute_jsx(command)
    
    def open_document(self, file_path: str) -> Dict[str, Any]:
        """Open an existing document in Photoshop."""
        command = {
            "action": "open_document",
            "params": {
                "path": file_path
            }
        }
        return self.execute_jsx(command)
    
    def save_document(self, file_path: str, format: str = "JPEG", quality: int = 12) -> Dict[str, Any]:
        """Save the active document in Photoshop."""
        command = {
            "action": "save_document",
            "params": {
                "path": file_path,
                "format": format,
                "quality": quality
            }
        }
        return self.execute_jsx(command)
    
    def create_layer(self, name: str = "New Layer") -> Dict[str, Any]:
        """Create a new layer in the active document."""
        command = {
            "action": "create_layer",
            "params": {
                "name": name
            }
        }
        return self.execute_jsx(command)
    
    def add_text(self, text: str, position: List[float] = None, font_size: float = 72, color: List[float] = None, font: str = None) -> Dict[str, Any]:
        """Add text to the active document."""
        command = {
            "action": "add_text",
            "params": {
                "text": text,
                "position": position or [100, 100],
                "fontSize": font_size,
                "color": color or [0, 0, 0],
                "font": font
            }
        }
        return self.execute_jsx(command)
    
    def add_shape(self, shape_type: str = "rectangle", position: List[float] = None, size: List[float] = None, color: List[float] = None) -> Dict[str, Any]:
        """Add a shape to the active document."""
        command = {
            "action": "add_shape",
            "params": {
                "shapeType": shape_type,
                "position": position or [100, 100],
                "size": size or [500, 300],
                "color": color or [255, 0, 0]
            }
        }
        return self.execute_jsx(command)
    
    def apply_filter(self, filter_type: str = "GaussianBlur", options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply a filter to the active layer."""
        command = {
            "action": "apply_filter",
            "params": {
                "filterType": filter_type,
                "options": options or {}
            }
        }
        return self.execute_jsx(command)
    
    def apply_adjustment(self, adjustment_type: str = "Levels", options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply an adjustment to the active document."""
        command = {
            "action": "apply_adjustment",
            "params": {
                "adjustmentType": adjustment_type,
                "options": options or {}
            }
        }
        return self.execute_jsx(command)
    
    def resize_image(self, width: int, height: int, resample_method: str = "bicubic") -> Dict[str, Any]:
        """Resize the active document."""
        command = {
            "action": "resize_image",
            "params": {
                "width": width,
                "height": height,
                "resampleMethod": resample_method
            }
        }
        return self.execute_jsx(command)
    
    def crop_image(self, bounds: List[float]) -> Dict[str, Any]:
        """Crop the active document."""
        command = {
            "action": "crop_image",
            "params": {
                "bounds": bounds
            }
        }
        return self.execute_jsx(command)
    
    def execute_script(self, script: str) -> Dict[str, Any]:
        """Execute a custom ExtendScript in Photoshop."""
        command = {
            "action": "execute_script",
            "params": {
                "script": script
            }
        }
        return self.execute_jsx(command)
    
    def create_photo_manipulation(self, base_image_path: str, elements: List[Dict[str, Any]], output_path: str = None) -> Dict[str, Any]:
        """Create a photo manipulation by compositing multiple elements."""
        # Open base image
        open_result = self.open_document(base_image_path)
        
        if not open_result.get("success"):
            return open_result
        
        # Add each element
        for i, element in enumerate(elements):
            element_path = element.get("path")
            position = element.get("position", [0, 0])
            scale = element.get("scale", 100)
            rotation = element.get("rotation", 0)
            
            # Create a script to place the element
            script = f"""
// Open element
var elementFile = File("{element_path}");
var elementDoc = app.open(elementFile);

// Copy element
elementDoc.selection.selectAll();
elementDoc.selection.copy();
elementDoc.close(SaveOptions.DONOTSAVECHANGES);

// Paste into main document
app.activeDocument.paste();
var layer = app.activeDocument.activeLayer;
layer.name = "Element {i+1}";

// Position element
layer.translate({position[0]}, {position[1]});

// Scale element
layer.resize({scale}, {scale}, AnchorPosition.MIDDLECENTER);

// Rotate element
layer.rotate({rotation}, AnchorPosition.MIDDLECENTER);

// Return layer info
{{ id: layer.id, name: layer.name }}
"""
            script_result = self.execute_script(script)
            
            if not script_result.get("success"):
                return script_result
            
            # Apply filters if specified
            filters = element.get("filters", [])
            for filter_info in filters:
                filter_type = filter_info.get("type")
                filter_options = filter_info.get("options", {})
                
                filter_result = self.apply_filter(filter_type, filter_options)
                
                if not filter_result.get("success"):
                    return filter_result
        
        # Save result if output path is provided
        if output_path:
            save_result = self.save_document(output_path)
            
            if not save_result.get("success"):
                return save_result
        
        return {
            "success": True,
            "baseImage": base_image_path,
            "elementCount": len(elements),
            "outputPath": output_path
        }
    
    def create_social_media_post(self, template: str, text: str, image_path: str = None, output_path: str = None) -> Dict[str, Any]:
        """Create a social media post using a template."""
        # Create document based on template
        width = 1200
        height = 630
        
        if template == "facebook":
            width = 1200
            height = 630
        elif template == "instagram":
            width = 1080
            height = 1080
        elif template == "twitter":
            width = 1200
            height = 675
        elif template == "linkedin":
            width = 1200
            height = 627
        
        doc_result = self.create_document(f"{template.capitalize()} Post", width, height)
        
        if not doc_result.get("success"):
            return doc_result
        
        # Add background
        bg_result = self.add_shape("rectangle", [0, 0], [width, height], [240, 240, 240])
        
        if not bg_result.get("success"):
            return bg_result
        
        # Add image if provided
        if image_path:
            script = f"""
// Place image
var imageFile = File("{image_path}");
app.activeDocument.placedItems.add(imageFile);
var layer = app.activeDocument.activeLayer;
layer.name = "Image";

// Resize to fit
var docWidth = app.activeDocument.width.value;
var docHeight = app.activeDocument.height.value;
var imageWidth = layer.bounds[2] - layer.bounds[0];
var imageHeight = layer.bounds[3] - layer.bounds[1];

var scale = Math.min(docWidth / imageWidth, docHeight / imageHeight) * 0.8;
layer.resize(scale * 100, scale * 100, AnchorPosition.MIDDLECENTER);

// Center image
layer.translate((docWidth - layer.bounds[2] + layer.bounds[0]) / 2, (docHeight - layer.bounds[3] + layer.bounds[1]) / 2);

// Return layer info
{{ id: layer.id, name: layer.name }}
"""
            image_result = self.execute_script(script)
            
            if not image_result.get("success"):
                return image_result
        
        # Add text
        text_position = [width / 2, height * 0.8]
        text_result = self.add_text(text, text_position, 48, [0, 0, 0], "Arial")
        
        if not text_result.get("success"):
            return text_result
        
        # Center text
        center_script = """
// Center text
var layer = app.activeDocument.activeLayer;
var docWidth = app.activeDocument.width.value;
layer.textItem.justification = Justification.CENTER;

// Return layer info
{ id: layer.id, name: layer.name }
"""
        center_result = self.execute_script(center_script)
        
        if not center_result.get("success"):
            return center_result
        
        # Add template-specific elements
        if template == "instagram":
            # Add Instagram-like frame
            frame_result = self.add_shape("rectangle", [10, 10], [width - 20, height - 20], [255, 255, 255])
            
            if not frame_result.get("success"):
                return frame_result
        elif template == "twitter":
            # Add Twitter-like header
            header_result = self.add_shape("rectangle", [0, 0], [width, 100], [29, 161, 242])
            
            if not header_result.get("success"):
                return header_result
        
        # Save result if output path is provided
        if output_path:
            save_result = self.save_document(output_path)
            
            if not save_result.get("success"):
                return save_result
        
        return {
            "success": True,
            "template": template,
            "width": width,
            "height": height,
            "outputPath": output_path
        }
    
    def run_ai_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Photoshop."""
        return self.execute_script(agent_script)
    
    def create_image_from_text(self, text: str, output_path: str = None) -> Dict[str, Any]:
        """Create an image based on text description."""
        # Parse text to determine image type
        text_lower = text.lower()
        
        width = 1920
        height = 1080
        
        # Create document
        doc_result = self.create_document("Text-Based Image", width, height)
        
        if not doc_result.get("success"):
            return doc_result
        
        # Create background
        bg_color = [240, 240, 240]
        
        if "dark" in text_lower:
            bg_color = [30, 30, 30]
        elif "blue" in text_lower:
            bg_color = [100, 150, 255]
        elif "green" in text_lower:
            bg_color = [100, 255, 150]
        elif "red" in text_lower:
            bg_color = [255, 100, 100]
        
        bg_result = self.add_shape("rectangle", [0, 0], [width, height], bg_color)
        
        if not bg_result.get("success"):
            return bg_result
        
        # Add elements based on text
        if "landscape" in text_lower or "nature" in text_lower:
            # Create a landscape scene
            # Sky
            sky_result = self.add_shape("rectangle", [0, 0], [width, height / 2], [135, 206, 235])
            
            if not sky_result.get("success"):
                return sky_result
            
            # Ground
            ground_result = self.add_shape("rectangle", [0, height / 2], [width, height / 2], [34, 139, 34])
            
            if not ground_result.get("success"):
                return ground_result
            
            # Sun
            sun_result = self.add_shape("ellipse", [width * 0.8, height * 0.2], [100, 100], [255, 255, 0])
            
            if not sun_result.get("success"):
                return sun_result
            
        elif "portrait" in text_lower or "person" in text_lower:
            # Create a portrait-like composition
            # Background
            bg_result = self.add_shape("rectangle", [0, 0], [width, height], [200, 200, 220])
            
            if not bg_result.get("success"):
                return bg_result
            
            # Head shape
            head_result = self.add_shape("ellipse", [width / 2 - 100, height / 4], [200, 250], [255, 213, 170])
            
            if not head_result.get("success"):
                return head_result
            
            # Body shape
            body_result = self.add_shape("rectangle", [width / 2 - 150, height / 2], [300, 400], [100, 100, 150])
            
            if not body_result.get("success"):
                return body_result
            
        elif "abstract" in text_lower:
            # Create abstract art
            # Generate random shapes
            for i in range(10):
                x = width * (0.1 + 0.8 * (i % 3) / 2)
                y = height * (0.1 + 0.8 * (i // 3) / 3)
                size = [width * 0.2, height * 0.2]
                color = [
                    int(128 + 127 * Math.sin(i * 0.5)),
                    int(128 + 127 * Math.sin(i * 0.5 + 2)),
                    int(128 + 127 * Math.sin(i * 0.5 + 4))
                ]
                
                shape_type = "rectangle" if i % 2 == 0 else "ellipse"
                
                shape_result = self.add_shape(shape_type, [x, y], size, color)
                
                if not shape_result.get("success"):
                    return shape_result
        
        # Add text
        text_result = self.add_text(text, [width / 2, height * 0.9], 36, [0, 0, 0] if sum(bg_color) > 384 else [255, 255, 255])
        
        if not text_result.get("success"):
            return text_result
        
        # Center text
        center_script = """
// Center text
var layer = app.activeDocument.activeLayer;
var docWidth = app.activeDocument.width.value;
layer.textItem.justification = Justification.CENTER;

// Return layer info
{ id: layer.id, name: layer.name }
"""
        center_result = self.execute_script(center_script)
        
        if not center_result.get("success"):
            return center_result
        
        # Apply filters based on text
        if "blur" in text_lower:
            blur_result = self.apply_filter("GaussianBlur", {"radius": 10})
            
            if not blur_result.get("success"):
                return blur_result
        
        if "sharpen" in text_lower:
            sharpen_result = self.apply_filter("Sharpen")
            
            if not sharpen_result.get("success"):
                return sharpen_result
        
        # Save result if output path is provided
        if output_path:
            save_result = self.save_document(output_path)
            
            if not save_result.get("success"):
                return save_result
        
        return {
            "success": True,
            "description": text,
            "width": width,
            "height": height,
            "outputPath": output_path
        }

# Example usage
if __name__ == "__main__":
    controller = PhotoshopController()
    controller.create_document("Test Document")
    controller.add_text("Hello, Photoshop!")
    controller.save_document("output.jpg")