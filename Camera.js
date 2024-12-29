const { mat4, vec3 } = glMatrix;

class Camera {
  constructor() {
    this.m_width = 0;
    this.m_height = 0;

    this.m_near = 0.1;
    this.m_far = 100.0;

    this.m_position = vec3.fromValues(0.0, 0.0, -3.0);
    this.m_target = vec3.fromValues(0.0, 0.0, 0.0);

    this.m_forward = vec3.fromValues(0.0, 0.0, 1.0);
    this.m_right = vec3.fromValues(1.0, 0.0, 0.0);
    this.m_up = vec3.fromValues(0.0, 1.0, 0.0);
    this.m_baseUp = vec3.fromValues(0.0, 1.0, 0.0);
  }

  init(width, height) {
    this.m_width = width;
    this.m_height = height;
  }

  setWidthAndHeight(width, height) {
    this.m_width = width;
    this.m_height = height;
  }

  tumble(dx, dy) {
    // Step 1: Rotate around the world Y-axis (up-axis)
    const tmp = vec3.create();
    vec3.sub(tmp, this.m_position, this.m_target); // tmp = m_position - m_target
  
    const degreesX = dx * 0.004; // Horizontal rotation (around Y-axis)
    const rotationY = mat4.create();
    mat4.rotateY(rotationY, mat4.create(), degreesX);
    vec3.transformMat4(tmp, tmp, rotationY); // Rotate the position vector
  
    vec3.add(this.m_position, this.m_target, tmp); // Update m_position
    vec3.sub(this.m_forward, this.m_target, this.m_position); // Update forward vector
    vec3.normalize(this.m_forward, this.m_forward);
  
    // Step 2: Rotate around the camera's local X-axis (right-axis)
    const degreesY = dy * 0.004; // Vertical rotation (around X-axis)
    const upRotationAxis = vec3.clone(this.m_right);
    const tiltMatrix = mat4.create();
    mat4.rotate(tiltMatrix, mat4.create(), degreesY, upRotationAxis);
    vec3.transformMat4(tmp, tmp, tiltMatrix);
  
    vec3.add(this.m_position, this.m_target, tmp); // Update m_position
    vec3.sub(this.m_forward, this.m_target, this.m_position); // Update forward vector
    vec3.normalize(this.m_forward, this.m_forward);
  
    // Step 3: Clamp forward vector to prevent flipping
    const maxVerticalComponent = 0.9995;
    if (this.m_forward[1] > maxVerticalComponent || this.m_forward[1] < -maxVerticalComponent) {
      return; // Abort if vertical component exceeds the limit
    }
  
    // Step 4: Update right and up vectors
    vec3.cross(this.m_right, this.m_forward, this.m_baseUp);
    vec3.normalize(this.m_right, this.m_right);
  
    vec3.cross(this.m_up, this.m_right, this.m_forward);
    vec3.normalize(this.m_up, this.m_up);
  }
  

  zoom(dx, dy) {
    const speed = 0.01;
    const delta = (-dx + dy) * speed;

    const forwardDelta = vec3.create();
    vec3.scale(forwardDelta, this.m_forward, delta);
    vec3.add(this.m_position, this.m_position, forwardDelta);
  }

  pan(dx, dy) {
    const speed = 0.01;

    // Move along the up vector
    const upDelta = vec3.create();
    vec3.scale(upDelta, this.m_up, dy * speed);
    vec3.add(this.m_position, this.m_position, upDelta);
    vec3.add(this.m_target, this.m_target, upDelta);

    // Move along the right vector
    const rightDelta = vec3.create();
    vec3.scale(rightDelta, this.m_right, -dx * speed);
    vec3.add(this.m_position, this.m_position, rightDelta);
    vec3.add(this.m_target, this.m_target, rightDelta);
  }

  getViewMatrix() {
    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, this.m_position, this.m_target, this.m_up);
    return viewMatrix;
  }

  getProjectionMatrix() {
    const ratio = this.m_width / this.m_height;
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, Math.PI / 4, ratio, this.m_near, this.m_far); // 45-degree FOV
    return projectionMatrix;
  }

  getWorldPosition() {
    return this.m_position;
  }
}

export default Camera;
