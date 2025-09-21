const { mat4, vec3 } = glMatrix;

import Camera from "./Camera.js";
import Model from "./Model.js";
import Renderer from "./Renderer.js";

//--------------------------------------------------------------------------------
// Global objects

const camera = new Camera();
const model = new Model();
const renderer = new Renderer();

let isDragging = false;
let isPanning = false;
let lastMouseX = 0;
let lastMouseY = 0;

//--------------------------------------------------------------------------------
// Drag and Drop handling

function showErrorPopup(message) {
  let popup = document.createElement("div");
  popup.innerText = message;
  popup.style.position = "fixed";
  popup.style.top = "50%";
  popup.style.left = "50%";
  popup.style.transform = "translate(-50%, -50%)";
  popup.style.backgroundColor = "rgba(255, 0, 0, 0.8)";
  popup.style.color = "white";
  popup.style.padding = "15px 25px";
  popup.style.borderRadius = "8px";
  popup.style.fontSize = "18px";
  popup.style.fontWeight = "bold";
  popup.style.zIndex = "1000";

  document.body.appendChild(popup);

  setTimeout(() => {
    popup.remove();
  }, 3000); // Auto-hide after 3 seconds

  console.error('ERROR: ' + message);
}

function setupDragAndDrop(canvas) {
  canvas.ondragover = function(event) { 
    event.preventDefault(); 
  };

  canvas.ondrop = async function(event) {
    event.preventDefault();
    
    let file = event.dataTransfer.files[0];
    if (!file) return;
    
    // Check file extension
    const extension = file.name.slice(file.name.lastIndexOf(".") + 1).toLowerCase();
    if (extension !== "glb" && extension !== "gltf") {
      showErrorPopup("Unsupported file type: " + file.name + ". Only .glb and .gltf files are supported.");
      return;
    }
    
    console.log(`Dropped file: ${file.name} (Size: ${file.size} bytes)`);
    
    try {
      // Create a blob URL for the dropped file
      const fileURL = URL.createObjectURL(file);
      
      // Load the new model
      await model.load(fileURL);
      
      // Reset camera to frame the new model
      const bounds = model.getBounds();
      camera.resetToModel(bounds.min, bounds.max);
      
      // Update renderer with new model
      await renderer.updateModel(model);
      
      // Clean up the blob URL
      URL.revokeObjectURL(fileURL);
      
      console.log(`Successfully loaded model: ${file.name}`);
      
    } catch (error) {
      showErrorPopup("Failed to load model: " + file.name);
      console.error("Model loading error:", error);
    }
  };
}

//--------------------------------------------------------------------------------
// Input handling

function onMouseDown(event) {
  if (event.button === 0) { // Left mouse button
    isDragging = true;
  } else if (event.button === 1) { // Middle mouse button
    isPanning = true;
    event.preventDefault(); // Prevent default middle-click behavior
  }

  lastMouseX = event.clientX;
  lastMouseY = event.clientY;
}

function onMouseUp(event) {
  if (event.button === 0) {
    isDragging = false;
  } else if (event.button === 1) {
    isPanning = false;
  }
}

function onMouseMove(event) {
  if (!isDragging && !isPanning) return;

  const deltaX = event.clientX - lastMouseX;
  const deltaY = event.clientY - lastMouseY;
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;

  if (isPanning) {
    camera.pan(deltaX, deltaY);
  } else if (isDragging) {
    camera.tumble(-deltaX, -deltaY);
  }
}

function onMouseWheel(event) {
  const delta = -Math.sign(event.deltaY);
  camera.zoom(0, delta * 10);
}

//--------------------------------------------------------------------------------
// Application

async function launchApp() {
  // Get canvas
  const canvas = document.getElementById("gpuCanvas");

  // Resize helper to fill window & account for device pixel ratio
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(window.innerWidth * dpr);
    const displayHeight = Math.floor(window.innerHeight * dpr);
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      camera.resizeViewport(displayWidth, displayHeight);
      if (renderer && renderer.device) {
        renderer.resize(displayWidth, displayHeight);
      }
    }
  }
  // Initial size
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Setup input handlers
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("wheel", onMouseWheel);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  // Setup drag-and-drop handlers
  setupDragAndDrop(canvas);

  // Load model
  try {
    await model.load(
      //"https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/FlightHelmet/glTF/FlightHelmet.gltf"
      "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf"
      //"https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf"
    );
    // After model loads, reset camera to frame it
    const bounds = model.getBounds();
    camera.resetToModel(bounds.min, bounds.max);
  } catch (error) {
    console.error("Failed to load model:", error);
    return;
  }

  // Initialize renderer
  try {
    await renderer.initialize(canvas, camera, model);
  } catch (error) {
    console.error("Failed to initialize renderer:", error);
    return;
  }

  // Animation control
  let isAnimating = true;
  window.addEventListener("keydown", async (event) => {
    if (event.key === "a" || event.key === "A") {
      // Shift-A resets the model orientation
      if (event.shiftKey) {
        model.resetOrientation();
      } else {
        // 'a' toggles model animation
        isAnimating = !isAnimating;
      }
    } else if (event.key === "Home") {
      // HOME key resets camera to frame model
      const bounds = model.getBounds();
      camera.resetToModel(bounds.min, bounds.max);
    } else if (event.key === "r" || event.key === "R") {
      // 'R' key reloads shaders
      try {
        console.log("Reloading shaders...");
        await renderer.reloadShaders();
        console.log("Shaders reloaded successfully!");
      } catch (error) {
        console.error("Failed to reload shaders:", error);
        showErrorPopup("Failed to reload shaders: " + error.message);
      }
    }
  });

  // Main render loop
  let lastTime = null;
  function frame(currentTime = performance.now()) {
    // Calculate deltaTime
    let deltaTime = 16.67;
    if (lastTime !== null) {
      deltaTime = currentTime - lastTime;
      if (deltaTime <= 0 || deltaTime > 100) deltaTime = 16.67;
    }
    lastTime = currentTime;

    // Update model
    model.update(deltaTime, isAnimating);

    // Render frame
    renderer.render(model, camera);

    // Continue loop
    requestAnimationFrame(frame);
  }

  // Start rendering
  frame();
}

// Initialize app
if (navigator.gpu) {
  launchApp();
} else {
  console.error("WebGPU is not supported on this browser.");
}
