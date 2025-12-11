/**
 * @file OrbitControls.js
 * @brief Handles mouse input for camera orbit controls (tumble, pan, zoom).
 */

export default class OrbitControls {
  // Constants
  static #kZoomSensitivity = 30.0;

  // Private field declarations
  #mouseTumble;
  #mousePan;
  #mouseLastPos;

  // Arrow function class fields - automatically bound to 'this'
  #onMouseDown = (event) => {
    if (event.button === 0) {
      // Left mouse button
      if (event.shiftKey) {
        // Shift + Left click = Pan
        this.#mousePan = true;
      } else {
        // Left click = Tumble
        this.#mouseTumble = true;
      }
    } else if (event.button === 1) {
      // Middle mouse button = Pan
      this.#mousePan = true;
      event.preventDefault(); // Prevent default middle-click behavior
    }

    this.#mouseLastPos.x = event.clientX;
    this.#mouseLastPos.y = event.clientY;
  };

  #onMouseUp = (event) => {
    if (event.button === 0) {
      // Left mouse button
      this.#mouseTumble = false;
      this.#mousePan = false;
    } else if (event.button === 1) {
      // Middle mouse button
      this.#mousePan = false;
    }
  };

  #onMouseMove = (event) => {
    if (!this.#mouseTumble && !this.#mousePan) return;

    const deltaX = event.clientX - this.#mouseLastPos.x;
    const deltaY = event.clientY - this.#mouseLastPos.y;
    this.#mouseLastPos.x = event.clientX;
    this.#mouseLastPos.y = event.clientY;

    if (this.#mousePan) {
      this.camera.pan(deltaX, deltaY);
    } else if (this.#mouseTumble) {
      // Negative delta for natural rotation (matching C++ behavior)
      this.camera.tumble(-deltaX, -deltaY);
    }
  };

  #onMouseWheel = (event) => {
    // Normalize deltaY to match GLFW's scale (GLFW gives ~1.0 per tick, JS gives ~100)
    const normalizedYOffset = event.deltaY / 100.0;
    // Apply zoom sensitivity (matching C++ kZoomSensitivity = 30.0)
    const zoomDelta = normalizedYOffset * OrbitControls.#kZoomSensitivity;
    this.camera.zoom(0, -zoomDelta); // Negative for natural zoom direction
  };

  /**
   * @brief Constructs a new orbit controls instance.
   * @param {HTMLCanvasElement} canvas - The canvas element to attach controls to
   * @param {Object} camera - The camera instance to control
   */
  constructor(canvas, camera) {
    this.canvas = canvas;
    this.camera = camera;

    // Mouse state
    this.#mouseTumble = false;
    this.#mousePan = false;
    this.#mouseLastPos = { x: 0, y: 0 };
  }

  /**
   * @brief Attaches event listeners to the canvas.
   */
  attach() {
    this.canvas.addEventListener("mousedown", this.#onMouseDown);
    this.canvas.addEventListener("mouseup", this.#onMouseUp);
    this.canvas.addEventListener("mousemove", this.#onMouseMove);
    this.canvas.addEventListener("wheel", this.#onMouseWheel);
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  /**
   * @brief Removes event listeners from the canvas.
   */
  detach() {
    this.canvas.removeEventListener("mousedown", this.#onMouseDown);
    this.canvas.removeEventListener("mouseup", this.#onMouseUp);
    this.canvas.removeEventListener("mousemove", this.#onMouseMove);
    this.canvas.removeEventListener("wheel", this.#onMouseWheel);
  }
}

