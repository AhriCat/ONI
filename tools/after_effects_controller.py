import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import tempfile
import base64
import socket
import threading

logger = logging.getLogger(__name__)

class AfterEffectsController:
    """Controller for automating Adobe After Effects through ExtendScript."""
    
    def __init__(self, after_effects_path: str = None):
        """
        Initialize After Effects controller.
        
        Args:
            after_effects_path: Path to After Effects executable
        """
        self.after_effects_path = after_effects_path or self._find_after_effects_installation()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ae_controller_"))
        self.jsx_bridge_path = self._setup_jsx_bridge()
        
    def __del__(self):
        """Clean up resources."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _find_after_effects_installation(self) -> str:
        """Find After Effects installation automatically."""
        common_paths = [
            "C:/Program Files/Adobe/Adobe After Effects 2023/Support Files/AfterFX.exe",
            "C:/Program Files/Adobe/Adobe After Effects 2022/Support Files/AfterFX.exe",
            "C:/Program Files/Adobe/Adobe After Effects 2021/Support Files/AfterFX.exe",
            "/Applications/Adobe After Effects 2023/Adobe After Effects 2023.app/Contents/MacOS/AfterEffects",
            "/Applications/Adobe After Effects 2022/Adobe After Effects 2022.app/Contents/MacOS/AfterEffects"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find After Effects in registry (Windows)
        if os.name == 'nt':
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Adobe\After Effects") as key:
                    version = winreg.EnumKey(key, 0)  # Get the first version
                    with winreg.OpenKey(key, version) as version_key:
                        install_path, _ = winreg.QueryValueEx(version_key, "InstallPath")
                        return os.path.join(install_path, "Support Files", "AfterFX.exe")
            except:
                pass
        
        raise FileNotFoundError("After Effects installation not found")
    
    def _setup_jsx_bridge(self) -> Path:
        """Set up JSX bridge script for communication with After Effects."""
        jsx_bridge_path = self.temp_dir / "ae_bridge.jsx"
        
        jsx_content = """
// After Effects JSX Bridge Script
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
                case "create_composition":
                    result = createComposition(params);
                    break;
                case "import_footage":
                    result = importFootage(params);
                    break;
                case "add_layer":
                    result = addLayer(params);
                    break;
                case "add_effect":
                    result = addEffect(params);
                    break;
                case "add_keyframe":
                    result = addKeyframe(params);
                    break;
                case "render_composition":
                    result = renderComposition(params);
                    break;
                case "create_text_layer":
                    result = createTextLayer(params);
                    break;
                case "create_shape_layer":
                    result = createShapeLayer(params);
                    break;
                case "create_camera":
                    result = createCamera(params);
                    break;
                case "create_null":
                    result = createNull(params);
                    break;
                case "create_solid":
                    result = createSolid(params);
                    break;
                case "create_light":
                    result = createLight(params);
                    break;
                case "parent_layer":
                    result = parentLayer(params);
                    break;
                case "apply_animation_preset":
                    result = applyAnimationPreset(params);
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
    
    // Create a new composition
    function createComposition(params) {
        try {
            var compName = params.name || "New Composition";
            var width = params.width || 1920;
            var height = params.height || 1080;
            var duration = params.duration || 10;
            var frameRate = params.frameRate || 30;
            
            var comp = app.project.items.addComp(compName, width, height, 1, duration, frameRate);
            
            return {
                success: true,
                compId: comp.id,
                name: comp.name,
                width: comp.width,
                height: comp.height,
                duration: comp.duration
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Import footage
    function importFootage(params) {
        try {
            var filePath = params.path;
            var importOptions = new ImportOptions(File(filePath));
            var footage = app.project.importFile(importOptions);
            
            return {
                success: true,
                footageId: footage.id,
                name: footage.name,
                width: footage.width,
                height: footage.height,
                duration: footage.duration
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Add layer to composition
    function addLayer(params) {
        try {
            var compId = params.compId;
            var footageId = params.footageId;
            
            var comp = app.project.itemByID(compId);
            var footage = app.project.itemByID(footageId);
            
            var layer = comp.layers.add(footage);
            
            return {
                success: true,
                layerId: layer.index,
                name: layer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Add effect to layer
    function addEffect(params) {
        try {
            var compId = params.compId;
            var layerIndex = params.layerIndex;
            var effectName = params.effectName;
            
            var comp = app.project.itemByID(compId);
            var layer = comp.layer(layerIndex);
            
            var effect = layer.Effects.addProperty(effectName);
            
            return {
                success: true,
                effectName: effect.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Add keyframe to property
    function addKeyframe(params) {
        try {
            var compId = params.compId;
            var layerIndex = params.layerIndex;
            var propertyName = params.propertyName;
            var time = params.time || 0;
            var value = params.value;
            
            var comp = app.project.itemByID(compId);
            var layer = comp.layer(layerIndex);
            var property = layer.property(propertyName);
            
            property.setValueAtTime(time, value);
            
            return {
                success: true,
                propertyName: propertyName,
                time: time,
                value: value
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Render composition
    function renderComposition(params) {
        try {
            var compId = params.compId;
            var outputPath = params.outputPath;
            var outputModule = params.outputModule || "H.264";
            
            var comp = app.project.itemByID(compId);
            
            // Add to render queue
            var renderItem = app.project.renderQueue.items.add(comp);
            var outputModule = renderItem.outputModules[1];
            
            outputModule.applyTemplate(outputModule);
            outputModule.file = new File(outputPath);
            
            // Start rendering
            app.project.renderQueue.render();
            
            return {
                success: true,
                outputPath: outputPath
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create text layer
    function createTextLayer(params) {
        try {
            var compId = params.compId;
            var text = params.text || "Text Layer";
            var position = params.position || [comp.width/2, comp.height/2];
            
            var comp = app.project.itemByID(compId);
            var textLayer = comp.layers.addText(text);
            
            // Set position
            textLayer.position.setValue(position);
            
            // Set other text properties if provided
            if (params.fontSize) {
                var textProp = textLayer.property("Source Text").value;
                textProp.fontSize = params.fontSize;
                textLayer.property("Source Text").setValue(textProp);
            }
            
            if (params.fillColor) {
                var textProp = textLayer.property("Source Text").value;
                textProp.fillColor = params.fillColor;
                textLayer.property("Source Text").setValue(textProp);
            }
            
            return {
                success: true,
                layerId: textLayer.index,
                name: textLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create shape layer
    function createShapeLayer(params) {
        try {
            var compId = params.compId;
            var shapeType = params.shapeType || "rectangle";
            var position = params.position || [comp.width/2, comp.height/2];
            var size = params.size || [100, 100];
            
            var comp = app.project.itemByID(compId);
            var shapeLayer = comp.layers.addShape();
            
            // Add shape group
            var shapeGroup = shapeLayer.property("Contents").addProperty("ADBE Vector Group");
            
            // Add shape based on type
            var shape;
            if (shapeType === "rectangle") {
                shape = shapeGroup.property("Contents").addProperty("ADBE Vector Shape - Rect");
                shape.property("Size").setValue(size);
            } else if (shapeType === "ellipse") {
                shape = shapeGroup.property("Contents").addProperty("ADBE Vector Shape - Ellipse");
                shape.property("Size").setValue(size);
            } else if (shapeType === "star") {
                shape = shapeGroup.property("Contents").addProperty("ADBE Vector Shape - Star");
                shape.property("Points").setValue(5);
                shape.property("Outer Radius").setValue(size[0]/2);
                shape.property("Inner Radius").setValue(size[0]/4);
            }
            
            // Add fill
            var fill = shapeGroup.property("Contents").addProperty("ADBE Vector Graphic - Fill");
            if (params.fillColor) {
                fill.property("Color").setValue(params.fillColor);
            }
            
            // Set position
            shapeLayer.position.setValue(position);
            
            return {
                success: true,
                layerId: shapeLayer.index,
                name: shapeLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create camera
    function createCamera(params) {
        try {
            var compId = params.compId;
            var cameraType = params.cameraType || "TwoNode";
            var position = params.position || [0, 0, -500];
            
            var comp = app.project.itemByID(compId);
            var cameraLayer = comp.layers.addCamera(cameraType, position);
            
            return {
                success: true,
                layerId: cameraLayer.index,
                name: cameraLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create null object
    function createNull(params) {
        try {
            var compId = params.compId;
            var position = params.position || [comp.width/2, comp.height/2];
            
            var comp = app.project.itemByID(compId);
            var nullLayer = comp.layers.addNull();
            
            // Set position
            nullLayer.position.setValue(position);
            
            return {
                success: true,
                layerId: nullLayer.index,
                name: nullLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create solid
    function createSolid(params) {
        try {
            var compId = params.compId;
            var color = params.color || [1, 1, 1];
            var name = params.name || "Solid";
            var width = params.width || comp.width;
            var height = params.height || comp.height;
            
            var comp = app.project.itemByID(compId);
            var solidLayer = comp.layers.addSolid(color, name, width, height, 1);
            
            return {
                success: true,
                layerId: solidLayer.index,
                name: solidLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Create light
    function createLight(params) {
        try {
            var compId = params.compId;
            var lightType = params.lightType || "Point";
            var position = params.position || [0, 0, -500];
            
            var comp = app.project.itemByID(compId);
            var lightLayer = comp.layers.addLight(lightType, position);
            
            return {
                success: true,
                layerId: lightLayer.index,
                name: lightLayer.name
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Parent layer
    function parentLayer(params) {
        try {
            var compId = params.compId;
            var childLayerIndex = params.childLayerIndex;
            var parentLayerIndex = params.parentLayerIndex;
            
            var comp = app.project.itemByID(compId);
            var childLayer = comp.layer(childLayerIndex);
            
            childLayer.parent = comp.layer(parentLayerIndex);
            
            return {
                success: true,
                childLayerId: childLayerIndex,
                parentLayerId: parentLayerIndex
            };
        } catch (e) {
            return { success: false, error: e.toString() };
        }
    }
    
    // Apply animation preset
    function applyAnimationPreset(params) {
        try {
            var compId = params.compId;
            var layerIndex = params.layerIndex;
            var presetPath = params.presetPath;
            
            var comp = app.project.itemByID(compId);
            var layer = comp.layer(layerIndex);
            
            layer.applyPreset(File(presetPath));
            
            return {
                success: true,
                layerId: layerIndex,
                presetPath: presetPath
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
        """Execute JSX command in After Effects."""
        try:
            # Create input and output files
            input_file = self.temp_dir / f"ae_input_{time.time()}.json"
            output_file = self.temp_dir / f"ae_output_{time.time()}.json"
            
            # Write command to input file
            with open(input_file, 'w') as f:
                json.dump(command, f)
            
            # Build command
            cmd = [
                self.after_effects_path,
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
    
    def create_composition(self, name: str, width: int = 1920, height: int = 1080, duration: float = 10.0, frame_rate: float = 30.0) -> Dict[str, Any]:
        """Create a new composition in After Effects."""
        command = {
            "action": "create_composition",
            "params": {
                "name": name,
                "width": width,
                "height": height,
                "duration": duration,
                "frameRate": frame_rate
            }
        }
        return self.execute_jsx(command)
    
    def import_footage(self, file_path: str) -> Dict[str, Any]:
        """Import footage into After Effects."""
        command = {
            "action": "import_footage",
            "params": {
                "path": file_path
            }
        }
        return self.execute_jsx(command)
    
    def add_layer(self, comp_id: int, footage_id: int) -> Dict[str, Any]:
        """Add a layer to a composition."""
        command = {
            "action": "add_layer",
            "params": {
                "compId": comp_id,
                "footageId": footage_id
            }
        }
        return self.execute_jsx(command)
    
    def add_effect(self, comp_id: int, layer_index: int, effect_name: str) -> Dict[str, Any]:
        """Add an effect to a layer."""
        command = {
            "action": "add_effect",
            "params": {
                "compId": comp_id,
                "layerIndex": layer_index,
                "effectName": effect_name
            }
        }
        return self.execute_jsx(command)
    
    def add_keyframe(self, comp_id: int, layer_index: int, property_name: str, time: float, value: Any) -> Dict[str, Any]:
        """Add a keyframe to a property."""
        command = {
            "action": "add_keyframe",
            "params": {
                "compId": comp_id,
                "layerIndex": layer_index,
                "propertyName": property_name,
                "time": time,
                "value": value
            }
        }
        return self.execute_jsx(command)
    
    def render_composition(self, comp_id: int, output_path: str, output_module: str = "H.264") -> Dict[str, Any]:
        """Render a composition."""
        command = {
            "action": "render_composition",
            "params": {
                "compId": comp_id,
                "outputPath": output_path,
                "outputModule": output_module
            }
        }
        return self.execute_jsx(command)
    
    def create_text_layer(self, comp_id: int, text: str, position: List[float] = None, font_size: float = 72, fill_color: List[float] = None) -> Dict[str, Any]:
        """Create a text layer in a composition."""
        command = {
            "action": "create_text_layer",
            "params": {
                "compId": comp_id,
                "text": text,
                "position": position,
                "fontSize": font_size,
                "fillColor": fill_color
            }
        }
        return self.execute_jsx(command)
    
    def create_shape_layer(self, comp_id: int, shape_type: str = "rectangle", position: List[float] = None, size: List[float] = None, fill_color: List[float] = None) -> Dict[str, Any]:
        """Create a shape layer in a composition."""
        command = {
            "action": "create_shape_layer",
            "params": {
                "compId": comp_id,
                "shapeType": shape_type,
                "position": position,
                "size": size,
                "fillColor": fill_color
            }
        }
        return self.execute_jsx(command)
    
    def create_camera(self, comp_id: int, camera_type: str = "TwoNode", position: List[float] = None) -> Dict[str, Any]:
        """Create a camera in a composition."""
        command = {
            "action": "create_camera",
            "params": {
                "compId": comp_id,
                "cameraType": camera_type,
                "position": position
            }
        }
        return self.execute_jsx(command)
    
    def create_null(self, comp_id: int, position: List[float] = None) -> Dict[str, Any]:
        """Create a null object in a composition."""
        command = {
            "action": "create_null",
            "params": {
                "compId": comp_id,
                "position": position
            }
        }
        return self.execute_jsx(command)
    
    def create_solid(self, comp_id: int, color: List[float] = None, name: str = "Solid", width: int = None, height: int = None) -> Dict[str, Any]:
        """Create a solid in a composition."""
        command = {
            "action": "create_solid",
            "params": {
                "compId": comp_id,
                "color": color,
                "name": name,
                "width": width,
                "height": height
            }
        }
        return self.execute_jsx(command)
    
    def create_light(self, comp_id: int, light_type: str = "Point", position: List[float] = None) -> Dict[str, Any]:
        """Create a light in a composition."""
        command = {
            "action": "create_light",
            "params": {
                "compId": comp_id,
                "lightType": light_type,
                "position": position
            }
        }
        return self.execute_jsx(command)
    
    def parent_layer(self, comp_id: int, child_layer_index: int, parent_layer_index: int) -> Dict[str, Any]:
        """Parent a layer to another layer."""
        command = {
            "action": "parent_layer",
            "params": {
                "compId": comp_id,
                "childLayerIndex": child_layer_index,
                "parentLayerIndex": parent_layer_index
            }
        }
        return self.execute_jsx(command)
    
    def apply_animation_preset(self, comp_id: int, layer_index: int, preset_path: str) -> Dict[str, Any]:
        """Apply an animation preset to a layer."""
        command = {
            "action": "apply_animation_preset",
            "params": {
                "compId": comp_id,
                "layerIndex": layer_index,
                "presetPath": preset_path
            }
        }
        return self.execute_jsx(command)
    
    def execute_script(self, script: str) -> Dict[str, Any]:
        """Execute a custom ExtendScript in After Effects."""
        command = {
            "action": "execute_script",
            "params": {
                "script": script
            }
        }
        return self.execute_jsx(command)
    
    def create_motion_graphics_template(self, comp_id: int, output_path: str) -> Dict[str, Any]:
        """Create a Motion Graphics template from a composition."""
        script = f"""
// Create Motion Graphics template
var comp = app.project.itemByID({comp_id});
var file = new File("{output_path}");

// Create Essential Graphics panel if it doesn't exist
var mainWindow = app.findMenuCommandId("Essential Graphics");
app.executeCommand(app.findMenuCommandId("Essential Graphics"));

// Add composition to Essential Graphics panel
var egtPanel = app.getPanel("Essential Graphics");
egtPanel.setCompAsSource(comp);

// Export as Motion Graphics template
egtPanel.exportMGT(file);

// Return result
{{ success: true, outputPath: "{output_path}" }}
"""
        return self.execute_script(script)
    
    def create_animation_from_text(self, text: str, output_path: str = None) -> Dict[str, Any]:
        """Create an animation based on text description."""
        # Create a composition
        comp_result = self.create_composition("Text Animation", 1920, 1080, 5.0, 30.0)
        
        if not comp_result.get("success"):
            return comp_result
        
        comp_id = comp_result["compId"]
        
        # Create text layer
        text_result = self.create_text_layer(comp_id, text, [960, 540], 72, [1, 1, 1, 1])
        
        if not text_result.get("success"):
            return text_result
        
        layer_index = text_result["layerId"]
        
        # Add animation based on text content
        text_lower = text.lower()
        
        if "fade" in text_lower:
            # Add fade in/out animation
            self.add_keyframe(comp_id, layer_index, "Opacity", 0, 0)
            self.add_keyframe(comp_id, layer_index, "Opacity", 1, 100)
            self.add_keyframe(comp_id, layer_index, "Opacity", 4, 100)
            self.add_keyframe(comp_id, layer_index, "Opacity", 5, 0)
        elif "scale" in text_lower:
            # Add scale animation
            self.add_keyframe(comp_id, layer_index, "Scale", 0, [0, 0])
            self.add_keyframe(comp_id, layer_index, "Scale", 1, [100, 100])
            self.add_keyframe(comp_id, layer_index, "Scale", 4, [100, 100])
            self.add_keyframe(comp_id, layer_index, "Scale", 5, [0, 0])
        elif "rotate" in text_lower:
            # Add rotation animation
            self.add_keyframe(comp_id, layer_index, "Rotation", 0, 0)
            self.add_keyframe(comp_id, layer_index, "Rotation", 5, 360)
        else:
            # Default: Add position animation
            self.add_keyframe(comp_id, layer_index, "Position", 0, [960, 1200])
            self.add_keyframe(comp_id, layer_index, "Position", 2.5, [960, 540])
            self.add_keyframe(comp_id, layer_index, "Position", 5, [960, -100])
        
        # Add text animator
        self.add_effect(comp_id, layer_index, "ADBE Text Animator")
        
        # Render if output path is provided
        if output_path:
            render_result = self.render_composition(comp_id, output_path)
            if not render_result.get("success"):
                return render_result
        
        return {
            "success": True,
            "compId": comp_id,
            "layerId": layer_index,
            "text": text,
            "outputPath": output_path
        }
    
    def create_lower_third(self, title: str, subtitle: str = None, style: str = "simple", comp_id: int = None) -> Dict[str, Any]:
        """Create a lower third animation."""
        # Create a composition if not provided
        if not comp_id:
            comp_result = self.create_composition("Lower Third", 1920, 1080, 10.0, 30.0)
            
            if not comp_result.get("success"):
                return comp_result
            
            comp_id = comp_result["compId"]
        
        # Create background shape
        shape_result = self.create_shape_layer(comp_id, "rectangle", [960, 900], [1600, 150], [0, 0, 0.8, 0.8])
        
        if not shape_result.get("success"):
            return shape_result
        
        bg_layer_index = shape_result["layerId"]
        
        # Create title text
        title_result = self.create_text_layer(comp_id, title, [200, 880], 48, [1, 1, 1, 1])
        
        if not title_result.get("success"):
            return title_result
        
        title_layer_index = title_result["layerId"]
        
        # Create subtitle text if provided
        if subtitle:
            subtitle_result = self.create_text_layer(comp_id, subtitle, [200, 930], 32, [0.8, 0.8, 0.8, 1])
            
            if not subtitle_result.get("success"):
                return subtitle_result
            
            subtitle_layer_index = subtitle_result["layerId"]
        
        # Add animation based on style
        if style == "simple":
            # Simple slide in from left
            self.add_keyframe(comp_id, bg_layer_index, "Position", 0, [-800, 900])
            self.add_keyframe(comp_id, bg_layer_index, "Position", 1, [960, 900])
            
            self.add_keyframe(comp_id, title_layer_index, "Position", 0, [-1560, 880])
            self.add_keyframe(comp_id, title_layer_index, "Position", 1.2, [200, 880])
            
            if subtitle:
                self.add_keyframe(comp_id, subtitle_layer_index, "Position", 0, [-1560, 930])
                self.add_keyframe(comp_id, subtitle_layer_index, "Position", 1.4, [200, 930])
        elif style == "fade":
            # Fade in
            self.add_keyframe(comp_id, bg_layer_index, "Opacity", 0, 0)
            self.add_keyframe(comp_id, bg_layer_index, "Opacity", 1, 100)
            
            self.add_keyframe(comp_id, title_layer_index, "Opacity", 0, 0)
            self.add_keyframe(comp_id, title_layer_index, "Opacity", 1.2, 100)
            
            if subtitle:
                self.add_keyframe(comp_id, subtitle_layer_index, "Opacity", 0, 0)
                self.add_keyframe(comp_id, subtitle_layer_index, "Opacity", 1.4, 100)
        elif style == "scale":
            # Scale up
            self.add_keyframe(comp_id, bg_layer_index, "Scale", 0, [0, 100])
            self.add_keyframe(comp_id, bg_layer_index, "Scale", 1, [100, 100])
            
            self.add_keyframe(comp_id, title_layer_index, "Scale", 0, [0, 0])
            self.add_keyframe(comp_id, title_layer_index, "Scale", 1.2, [100, 100])
            
            if subtitle:
                self.add_keyframe(comp_id, subtitle_layer_index, "Scale", 0, [0, 0])
                self.add_keyframe(comp_id, subtitle_layer_index, "Scale", 1.4, [100, 100])
        
        return {
            "success": True,
            "compId": comp_id,
            "bgLayerId": bg_layer_index,
            "titleLayerId": title_layer_index,
            "subtitleLayerId": subtitle_layer_index if subtitle else None
        }
    
    def create_kinetic_typography(self, text: str, comp_id: int = None) -> Dict[str, Any]:
        """Create kinetic typography animation."""
        # Create a composition if not provided
        if not comp_id:
            comp_result = self.create_composition("Kinetic Typography", 1920, 1080, 10.0, 30.0)
            
            if not comp_result.get("success"):
                return comp_result
            
            comp_id = comp_result["compId"]
        
        # Split text into words
        words = text.split()
        
        # Create a text layer for each word
        layer_indices = []
        for i, word in enumerate(words):
            # Calculate position
            position = [960, 540 + (i - len(words)/2) * 100]
            
            # Create text layer
            text_result = self.create_text_layer(comp_id, word, position, 72, [1, 1, 1, 1])
            
            if not text_result.get("success"):
                return text_result
            
            layer_index = text_result["layerId"]
            layer_indices.append(layer_index)
            
            # Add animation with staggered timing
            start_time = i * 0.2
            
            # Position animation
            self.add_keyframe(comp_id, layer_index, "Position", start_time, [1920, position[1]])
            self.add_keyframe(comp_id, layer_index, "Position", start_time + 1, position)
            self.add_keyframe(comp_id, layer_index, "Position", start_time + 3, position)
            self.add_keyframe(comp_id, layer_index, "Position", start_time + 4, [0, position[1]])
            
            # Scale animation
            self.add_keyframe(comp_id, layer_index, "Scale", start_time, [0, 0])
            self.add_keyframe(comp_id, layer_index, "Scale", start_time + 1, [100, 100])
            self.add_keyframe(comp_id, layer_index, "Scale", start_time + 3, [100, 100])
            self.add_keyframe(comp_id, layer_index, "Scale", start_time + 4, [0, 0])
        
        return {
            "success": True,
            "compId": comp_id,
            "layerIndices": layer_indices,
            "wordCount": len(words)
        }
    
    def run_ai_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in After Effects."""
        return self.execute_script(agent_script)

# Example usage
if __name__ == "__main__":
    controller = AfterEffectsController()
    comp_result = controller.create_composition("Test Composition")
    if comp_result["success"]:
        comp_id = comp_result["compId"]
        controller.create_text_layer(comp_id, "Hello, After Effects!")
        controller.render_composition(comp_id, "output.mp4")