const { mat4, vec3 } = glMatrix;

export default class Camera {
  // Constants
  static #kTumbleSpeed = 0.004;
  static #kTiltClamp = 0.98; // Prevent gimbal lock.
  static #kPanSpeed = 0.01;
  static #kZoomSpeed = 0.01;
  static #kDefaultFOV = Math.PI / 4; // 45 degrees

  constructor(width = 800, height = 600) {
    this.m_width = width;
    this.m_height = height;

    this.m_near = 0.1;
    this.m_far = 100.0;

    this.m_position = vec3.fromValues(0.0, 0.0, -3.0);
    this.m_target = vec3.fromValues(0.0, 0.0, 0.0);

    this.m_forward = vec3.fromValues(0.0, 0.0, 1.0);
    this.m_right = vec3.fromValues(1.0, 0.0, 0.0);
    this.m_up = vec3.fromValues(0.0, 1.0, 0.0);
    this.m_baseUp = vec3.fromValues(0.0, 1.0, 0.0);

    this.tmpMat4 = mat4.create(); // Preallocated temporary matrix
  }

  // Resize the viewport
  resizeViewport(width, height) {
    if (width > 0 && height > 0) {
      this.m_width = width;
      this.m_height = height;
    }
  }

  // Tumble the camera (rotate around target)
  tumble(dx, dy) {
    const originalPosition = vec3.clone(this.m_position);
    const originalForward = vec3.clone(this.m_forward);

    // Step 1: Rotate around the world Y-axis (up-axis)
    const tmp = vec3.create();
    vec3.sub(tmp, this.m_position, this.m_target); // tmp = m_position - m_target

    const degreesX = dx * Camera.#kTumbleSpeed; // Horizontal rotation
    mat4.rotateY(this.tmpMat4, mat4.create(), degreesX);
    vec3.transformMat4(tmp, tmp, this.tmpMat4);

    vec3.add(this.m_position, this.m_target, tmp); // Update m_position
    this.#updateBasisVectors();

    // Step 2: Rotate around the camera's local X-axis (right-axis)
    const degreesY = dy * Camera.#kTumbleSpeed; // Vertical rotation
    const upRotationAxis = vec3.clone(this.m_right);
    mat4.rotate(this.tmpMat4, mat4.create(), degreesY, upRotationAxis);
    vec3.transformMat4(tmp, tmp, this.tmpMat4);

    vec3.add(this.m_position, this.m_target, tmp); // Update m_position
    this.#updateBasisVectors();

    // Step 3: Clamp forward vector to prevent flipping
    if (Math.abs(this.m_forward[1]) > Camera.#kTiltClamp) {
      vec3.copy(this.m_position, originalPosition);
      vec3.copy(this.m_forward, originalForward);
    }
  }

  // Zoom the camera (move along forward vector)
  zoom(dx, dy) {
    const delta = (-dx + dy) * Camera.#kZoomSpeed;

    const forwardDelta = vec3.create();
    vec3.scale(forwardDelta, this.m_forward, delta);

    const newPosition = vec3.create();
    vec3.add(newPosition, this.m_position, forwardDelta);

    const distanceToTarget = vec3.distance(newPosition, this.m_target);
    if (distanceToTarget > this.m_near && distanceToTarget < this.m_far) {
      vec3.copy(this.m_position, newPosition);
    }
  }

  // Pan the camera (move along right and up vectors)
  pan(dx, dy) {
    const deltaX = -dx * Camera.#kPanSpeed;
    const deltaY = dy * Camera.#kPanSpeed;

    const rightDelta = vec3.create();
    vec3.scale(rightDelta, this.m_right, deltaX);

    const upDelta = vec3.create();
    vec3.scale(upDelta, this.m_up, deltaY);

    vec3.add(this.m_position, this.m_position, rightDelta);
    vec3.add(this.m_position, this.m_position, upDelta);
    vec3.add(this.m_target, this.m_target, rightDelta);
    vec3.add(this.m_target, this.m_target, upDelta);
  }

  // Get the view matrix
  getViewMatrix() {
    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, this.m_position, this.m_target, this.m_up);
    return viewMatrix;
  }

  // Get the projection matrix
  getProjectionMatrix() {
    const ratio = this.m_width / this.m_height;
    const projectionMatrix = mat4.create();
    mat4.perspective(
      projectionMatrix,
      Camera.#kDefaultFOV,
      ratio,
      this.m_near,
      this.m_far
    );
    return projectionMatrix;
  }

  // Get the world position of the camera
  getWorldPosition() {
    return this.m_position;
  }

  // Private: Update basis vectors
  #updateBasisVectors() {
    vec3.sub(this.m_forward, this.m_target, this.m_position);
    vec3.normalize(this.m_forward, this.m_forward);

    vec3.cross(this.m_right, this.m_forward, this.m_baseUp);
    vec3.normalize(this.m_right, this.m_right);

    vec3.cross(this.m_up, this.m_right, this.m_forward);
    vec3.normalize(this.m_up, this.m_up);
  }
}
