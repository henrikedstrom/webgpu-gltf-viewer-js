/**
 * @file main.js
 * @brief Application entry point. Initializes and runs the WebGPU glTF viewer.
 */

import Application from "./Application.js";

// Initialize app
if (navigator.gpu) {
  const app = new Application(window.innerWidth, window.innerHeight);
  app.run();
} else {
  console.error("WebGPU is not supported on this browser.");
}
