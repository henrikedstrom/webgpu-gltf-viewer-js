const { mat4, vec3 } = glMatrix;

export default class Camera {
  // Constants
  static #kTumbleSpeed = 0.004;
  static #kTiltClamp = 0.98; // Prevent gimbal lock.
  static #kPanSpeed = 0.02;
  static #kZoomSpeed = 0.05;
  static #kDefaultFOV = Math.PI / 4; // 45 degrees
  static #kNearClipFactor = 0.01;
  static #kFarClipFactor = 100.0;

  // Private field declarations
  #width;
  #height;
  #near;
  #far;
  #position;
  #target;
  #forward;
  #right;
  #up;
  #baseUp;
  #panFactor;
  #zoomFactor;
  #tmpMat4; // Preallocated temporary matrix

  constructor(width = 800, height = 600) {
    this.#width = width;
    this.#height = height;

    this.#near = 0.1;
    this.#far = 100.0;

    this.#position = vec3.fromValues(0.0, 0.0, -3.0);
    this.#target = vec3.fromValues(0.0, 0.0, 0.0);

    this.#forward = vec3.fromValues(0.0, 0.0, 1.0);
    this.#right = vec3.fromValues(1.0, 0.0, 0.0);
    this.#up = vec3.fromValues(0.0, 1.0, 0.0);
    this.#baseUp = vec3.fromValues(0.0, 1.0, 0.0);

    // Dynamic movement factors (scaled by model size)
    this.#panFactor = Camera.#kPanSpeed;
    this.#zoomFactor = Camera.#kZoomSpeed;

    this.#tmpMat4 = mat4.create(); // Preallocated temporary matrix
  }

  // Resize the viewport
  resizeViewport(width, height) {
    if (width > 0 && height > 0) {
      this.#width = width;
      this.#height = height;
    }
  }

  // Tumble the camera (rotate around target)
  tumble(dx, dy) {
    const originalPosition = vec3.clone(this.#position);
    const originalForward = vec3.clone(this.#forward);

    // Step 1: Rotate around the world Y-axis (up-axis)
    const tmp = vec3.create();
    vec3.sub(tmp, this.#position, this.#target); // tmp = position - target

    const degreesX = dx * Camera.#kTumbleSpeed; // Horizontal rotation
    mat4.rotateY(this.#tmpMat4, mat4.create(), degreesX);
    vec3.transformMat4(tmp, tmp, this.#tmpMat4);

    vec3.add(this.#position, this.#target, tmp); // Update position
    this.#updateBasisVectors();

    // Step 2: Rotate around the camera's local X-axis (right-axis)
    const degreesY = dy * Camera.#kTumbleSpeed; // Vertical rotation
    const upRotationAxis = vec3.clone(this.#right);
    mat4.rotate(this.#tmpMat4, mat4.create(), degreesY, upRotationAxis);
    vec3.transformMat4(tmp, tmp, this.#tmpMat4);

    vec3.add(this.#position, this.#target, tmp); // Update position
    this.#updateBasisVectors();

    // Step 3: Clamp forward vector to prevent flipping
    if (Math.abs(this.#forward[1]) > Camera.#kTiltClamp) {
      vec3.copy(this.#position, originalPosition);
      vec3.copy(this.#forward, originalForward);
    }
  }

  // Zoom the camera (move along forward vector)
  zoom(dx, dy) {
    const delta = (-dx + dy) * this.#zoomFactor;

    const forwardDelta = vec3.create();
    vec3.scale(forwardDelta, this.#forward, delta);

    const newPosition = vec3.create();
    vec3.add(newPosition, this.#position, forwardDelta);

    const distanceToTarget = vec3.distance(newPosition, this.#target);
    if (distanceToTarget > this.#near && distanceToTarget < this.#far) {
      vec3.copy(this.#position, newPosition);
    }
  }

  // Pan the camera (move along right and up vectors)
  pan(dx, dy) {
    const deltaX = -dx * this.#panFactor;
    const deltaY = dy * this.#panFactor;

    const rightDelta = vec3.create();
    vec3.scale(rightDelta, this.#right, deltaX);

    const upDelta = vec3.create();
    vec3.scale(upDelta, this.#up, deltaY);

    vec3.add(this.#position, this.#position, rightDelta);
    vec3.add(this.#position, this.#position, upDelta);
    vec3.add(this.#target, this.#target, rightDelta);
    vec3.add(this.#target, this.#target, upDelta);
  }

  // Reset camera to frame a model given its AABB
  resetToModel(minBounds, maxBounds) {
    // Validate bounds (any axis where max <= min is invalid)
    if (
      maxBounds[0] <= minBounds[0] ||
      maxBounds[1] <= minBounds[1] ||
      maxBounds[2] <= minBounds[2]
    ) {
      // Default to unit cube centered at origin
      vec3.set(minBounds, -0.5, -0.5, -0.5);
      vec3.set(maxBounds, 0.5, 0.5, 0.5);
      console.warn(
        "Invalid model bounds provided to resetToModel, defaulting to unit cube."
      );
    }

    // center = (min + max) * 0.5
    const center = vec3.create();
    vec3.add(center, minBounds, maxBounds);
    vec3.scale(center, center, 0.5);

    // radius = 0.5 * length(max - min)
    const diag = vec3.create();
    vec3.sub(diag, maxBounds, minBounds);
    const radius = 0.5 * vec3.length(diag);

    // distance = radius / sin(FOV/2)
    const halfFov = Camera.#kDefaultFOV * 0.5;
    const sinHalf = Math.sin(halfFov);
    const safeRadius = isFinite(radius) && radius > 1e-6 ? radius : 1.0;
    const distance = safeRadius / (sinHalf > 1e-6 ? sinHalf : 0.70710678); // fallback denom ~sin(45/2)

    // Position the camera at +Z looking toward -Z
    vec3.set(this.#position, center[0], center[1], center[2] + distance);
    vec3.copy(this.#target, center);

    // Near / far planes
    this.#near = safeRadius * Camera.#kNearClipFactor;
    this.#far = distance + safeRadius * Camera.#kFarClipFactor;
    if (this.#near < 1e-4) this.#near = 1e-4;
    if (this.#far <= this.#near + 1.0) this.#far = this.#near + 1.0;

    // Movement factors scaled by model size
    this.#panFactor = safeRadius * Camera.#kPanSpeed;
    this.#zoomFactor = safeRadius * Camera.#kZoomSpeed;

    // Recompute basis
    this.#updateBasisVectors();
  }

  // Get the view matrix
  getViewMatrix() {
    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, this.#position, this.#target, this.#up);
    return viewMatrix;
  }

  // Get the projection matrix
  getProjectionMatrix() {
    const ratio = this.#width / this.#height;
    const projectionMatrix = mat4.create();
    mat4.perspective(
      projectionMatrix,
      Camera.#kDefaultFOV,
      ratio,
      this.#near,
      this.#far
    );
    return projectionMatrix;
  }

  // Get the world position of the camera
  getWorldPosition() {
    return this.#position;
  }

  // Private: Update basis vectors
  #updateBasisVectors() {
    vec3.sub(this.#forward, this.#target, this.#position);
    vec3.normalize(this.#forward, this.#forward);

    vec3.cross(this.#right, this.#forward, this.#baseUp);
    vec3.normalize(this.#right, this.#right);

    vec3.cross(this.#up, this.#right, this.#forward);
    vec3.normalize(this.#up, this.#up);
  }
}
