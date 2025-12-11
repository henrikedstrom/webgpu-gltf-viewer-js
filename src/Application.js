/**
 * @file Application.js
 * @brief Main application class managing lifecycle, input, and rendering.
 */

import Camera from "./Camera.js";
import Environment from "./Environment.js";
import Model from "./Model.js";
import Renderer from "./Renderer.js";
import OrbitControls from "./OrbitControls.js";

export default class Application {
  // Private field declarations
  #width;
  #height;
  #animateModel;
  #camera;
  #environment;
  #model;
  #renderer;
  #canvas;
  #controls;
  #lastTime;

  /**
   * @brief Constructs a new application instance.
   * @param {number} width - Initial window width
   * @param {number} height - Initial window height
   */
  constructor(width, height) {
    this.#width = width;
    this.#height = height;
    this.#animateModel = true;

    // Create core objects
    this.#camera = new Camera();
    this.#environment = new Environment();
    this.#model = new Model();
    this.#renderer = new Renderer();

    // Expose for debugging
    window.debug = {
      camera: this.#camera,
      environment: this.#environment,
      model: this.#model,
      renderer: this.#renderer,
    };

    this.#canvas = null;
    this.#controls = null;
    this.#lastTime = null;
  }

  /**
   * @brief Initializes and runs the application.
   */
  async run() {
    // Get canvas
    this.#canvas = document.getElementById("gpuCanvas");
    if (!this.#canvas) {
      console.error("Canvas element 'gpuCanvas' not found");
      return;
    }

    // Setup resize handler
    this.#setupResize();

    // Create and attach orbit controls
    this.#controls = new OrbitControls(this.#canvas, this.#camera);
    this.#controls.attach();

    // Setup keyboard handler
    window.addEventListener("keydown", this.#onKeyDown.bind(this));

    // Setup drag-and-drop handlers
    this.#setupDragAndDrop(this.#canvas);

    // Load default environment
    try {
      await this.#environment.load("./assets/environments/helipad.hdr");
    } catch (error) {
      console.error("Failed to load environment:", error);
      return;
    }

    // Load default model
    try {
      await this.#model.load("./assets/models/DamagedHelmet.glb");
      // After model loads, reset camera to frame it
      this.#repositionCamera();
    } catch (error) {
      console.error("Failed to load model:", error);
      return;
    }

    // Initialize renderer
    try {
      await this.#renderer.initialize(
        this.#canvas,
        this.#camera,
        this.#environment,
        this.#model
      );
    } catch (error) {
      console.error("Failed to initialize renderer:", error);
      return;
    }

    // Start render loop
    this.#startRenderLoop();
  }

  /**
   * @brief Sets up canvas resize handling.
   * @private
   */
  #setupResize() {
    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const displayWidth = Math.floor(window.innerWidth * dpr);
      const displayHeight = Math.floor(window.innerHeight * dpr);
      if (
        this.#canvas.width !== displayWidth ||
        this.#canvas.height !== displayHeight
      ) {
        this.#canvas.width = displayWidth;
        this.#canvas.height = displayHeight;
        this.#width = displayWidth;
        this.#height = displayHeight;
        this.#camera.resizeViewport(displayWidth, displayHeight);
        if (this.#renderer) {
          this.#renderer.resize(displayWidth, displayHeight);
        }
      }
    };

    // Initial size
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
  }

  /**
   * @brief Sets up drag-and-drop file handling.
   * @param {HTMLCanvasElement} canvas - The canvas element
   * @private
   */
  #setupDragAndDrop(canvas) {
    canvas.ondragover = function (event) {
      event.preventDefault();
    };

    canvas.ondrop = async (event) => {
      event.preventDefault();

      let file = event.dataTransfer.files[0];
      if (!file) return;

      // Check file extension
      const extension = file.name.slice(file.name.lastIndexOf(".") + 1).toLowerCase();

      console.log(`Dropped file: ${file.name} (Size: ${file.size} bytes)`);

      // Create a blob URL for the dropped file
      const fileURL = URL.createObjectURL(file);

      try {
        if (extension === "glb" || extension === "gltf") {
          // Load model
          await this.#model.load(fileURL);

          // Reset camera to frame the new model
          this.#repositionCamera();

          // Update renderer with new model
          await this.#renderer.updateModel(this.#model);

          console.log(`Successfully loaded model: ${file.name}`);
        } else if (extension === "hdr") {
          // Load environment
          const success = await this.#environment.load(fileURL);
          if (success) {
            // Update renderer with new environment
            await this.#renderer.updateEnvironment(this.#environment);
            console.log(`Successfully loaded environment: ${file.name}`);
          } else {
            throw new Error("Failed to load HDR data");
          }
        } else {
          this.#showErrorPopup(
            "Unsupported file type: " +
              file.name +
              ". Supported formats: .glb, .gltf, .hdr"
          );
          return;
        }
      } catch (error) {
        this.#showErrorPopup(
          `Failed to load ${extension.toUpperCase()} file: ${file.name}`
        );
        console.error(`${extension.toUpperCase()} loading error:`, error);
      } finally {
        // Always clean up the blob URL
        URL.revokeObjectURL(fileURL);
      }
    };
  }

  /**
   * @brief Displays an error popup message.
   * @param {string} message - The error message to display
   * @private
   */
  #showErrorPopup(message) {
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

    console.error("ERROR: " + message);
  }

  /**
   * @brief Repositions the camera to frame the model.
   * @private
   */
  #repositionCamera() {
    const bounds = this.#model.getBounds();
    this.#camera.resetToModel(bounds.min, bounds.max);
  }

  /**
   * @brief Handles keyboard input events.
   * @param {KeyboardEvent} event - The keyboard event
   * @private
   */
  #onKeyDown(event) {
    if (event.key === "a" || event.key === "A") {
      // Shift-A resets the model orientation
      if (event.shiftKey) {
        this.#model.resetOrientation();
      } else {
        // 'a' toggles model animation
        this.#animateModel = !this.#animateModel;
      }
    } else if (event.key === "Home") {
      // HOME key resets camera to frame model
      this.#repositionCamera();
    } else if (event.key === "r" || event.key === "R") {
      // 'R' key reloads shaders
      this.#reloadShaders();
    }
  }

  /**
   * @brief Reloads shaders from disk.
   * @private
   */
  async #reloadShaders() {
    try {
      console.log("Reloading shaders...");
      await this.#renderer.reloadShaders();
      console.log("Shaders reloaded successfully!");
    } catch (error) {
      console.error("Failed to reload shaders:", error);
      this.#showErrorPopup("Failed to reload shaders: " + error.message);
    }
  }

  /**
   * @brief Starts the render loop.
   * @private
   */
  #startRenderLoop() {
    const frame = (currentTime = performance.now()) => {
      // Calculate deltaTime
      let deltaTime = 16.67;
      if (this.#lastTime !== null) {
        deltaTime = currentTime - this.#lastTime;
        if (deltaTime <= 0 || deltaTime > 100) deltaTime = 16.67;
      }
      this.#lastTime = currentTime;

      // Process frame
      this.#processFrame(deltaTime);

      // Continue loop
      requestAnimationFrame(frame);
    };

    // Start rendering
    frame();
  }

  /**
   * @brief Processes a single frame (update and render).
   * @param {number} deltaTime - Time since last frame in milliseconds
   * @private
   */
  #processFrame(deltaTime) {
    // Update model animation
    this.#model.update(deltaTime, this.#animateModel);

    // Prepare camera uniforms
    const cameraUniforms = {
      viewMatrix: this.#camera.getViewMatrix(),
      projectionMatrix: this.#camera.getProjectionMatrix(),
      cameraPosition: this.#camera.getWorldPosition(),
    };

    // Render frame
    this.#renderer.render(this.#model.getTransform(), cameraUniforms);
  }
}

