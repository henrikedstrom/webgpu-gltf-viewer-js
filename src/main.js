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
    camera.pan(-deltaX, deltaY);
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

  // Setup input handlers
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("wheel", onMouseWheel);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  // Initialize camera
  camera.resizeViewport(canvas.width, canvas.height);

  // Load model
  try {
    await model.load(
      "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf"
    );
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
  window.addEventListener("keydown", (event) => {
    if (event.key === "a" || event.key === "A") {
      isAnimating = !isAnimating;
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