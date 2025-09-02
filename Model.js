import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
const { mat4, vec3 } = glMatrix;

export default class Model {
  constructor() {
    // Transformation properties
    this.m_transform = mat4.create();
    this.m_rotationAngle = 0.0;
    
    // Bounds
    this.m_minBounds = vec3.create();
    this.m_maxBounds = vec3.create();
    
    // Geometry data
    this.m_vertices = [];
    this.m_indices = [];
    this.m_materials = [];
    this.m_textures = [];
    this.m_subMeshes = [];
    
    // Loading state
    this.m_isLoaded = false;
  }

  // Load a glTF model from URL
  async load(url) {
    return new Promise((resolve, reject) => {
      const loader = new GLTFLoader();
      loader.load(
        url,
        (gltf) => {
          this.#processGLTF(gltf);
          this.m_isLoaded = true;
          resolve();
        },
        undefined,
        (error) => reject(error)
      );
    });
  }

  // Update the model (for animation, rotation, etc.)
  update(deltaTime, animate = false) {
    if (animate) {
      this.m_rotationAngle += deltaTime * 0.001; // Slow rotation
      if (this.m_rotationAngle > 2 * Math.PI) {
        this.m_rotationAngle -= 2 * Math.PI;
      }
    }
    
    // Update transformation matrix
    mat4.identity(this.m_transform);
    mat4.rotateX(this.m_transform, this.m_transform, Math.PI / 2);
    mat4.rotateZ(this.m_transform, this.m_transform, this.m_rotationAngle);
  }

  // Reset model orientation
  resetOrientation() {
    this.m_rotationAngle = 0.0;
    this.update(0, false);
  }

  // Accessors
  getTransform() {
    return this.m_transform;
  }

  getBounds() {
    return {
      min: this.m_minBounds,
      max: this.m_maxBounds
    };
  }

  getVertices() {
    return this.m_vertices;
  }

  getIndices() {
    return this.m_indices;
  }

  getMaterials() {
    return this.m_materials;
  }

  getTextures() {
    return this.m_textures;
  }

  getSubMeshes() {
    return this.m_subMeshes;
  }

  isLoaded() {
    return this.m_isLoaded;
  }

  // Private method to process the loaded glTF data
  #processGLTF(gltf) {
    // Clear existing data
    this.#clearData();

    // Assuming a single mesh for now
    const geometry = gltf.scene.children[0].geometry;
    const attributes = geometry.attributes;

    // Extract positions and normals
    const positions = attributes.position.array;
    const normals = attributes.normal.array;
    const indices = geometry.index ? geometry.index.array : null;

    // Convert to our vertex format (interleaved position + normal)
    this.#createVertexData(positions, normals);
    
    // Store indices
    if (indices) {
      this.m_indices = indices;
    }

    // Compute bounds
    this.#recomputeBounds(positions);

    // Create a basic submesh (for now, single submesh)
    this.m_subMeshes.push({
      firstIndex: 0,
      indexCount: this.m_indices.length,
      materialIndex: 0,
      minBounds: vec3.clone(this.m_minBounds),
      maxBounds: vec3.clone(this.m_maxBounds)
    });

    // Create a basic material (for now)
    this.m_materials.push({
      baseColorFactor: [1.0, 1.0, 1.0, 1.0],
      metallicFactor: 1.0,
      roughnessFactor: 1.0,
      normalScale: 1.0,
      occlusionStrength: 1.0,
      emissiveFactor: [0.0, 0.0, 0.0],
      alphaMode: 0, // Opaque
      alphaCutoff: 0.5,
      doubleSided: false,
      baseColorTexture: -1,
      metallicRoughnessTexture: -1,
      normalTexture: -1,
      emissiveTexture: -1,
      occlusionTexture: -1
    });
  }

  // Private method to create interleaved vertex data
  #createVertexData(positions, normals) {
    const vertexCount = positions.length / 3;
    this.m_vertices = new Float32Array(vertexCount * 6); // 3 position + 3 normal

    for (let i = 0; i < vertexCount; i++) {
      const posIndex = i * 3;
      const vertIndex = i * 6;

      // Position
      this.m_vertices[vertIndex] = positions[posIndex];
      this.m_vertices[vertIndex + 1] = positions[posIndex + 1];
      this.m_vertices[vertIndex + 2] = positions[posIndex + 2];

      // Normal
      this.m_vertices[vertIndex + 3] = normals[posIndex];
      this.m_vertices[vertIndex + 4] = normals[posIndex + 1];
      this.m_vertices[vertIndex + 5] = normals[posIndex + 2];
    }
  }

  // Private method to compute model bounds
  #recomputeBounds(positions) {
    if (positions.length === 0) return;

    // Initialize bounds
    vec3.set(this.m_minBounds, positions[0], positions[1], positions[2]);
    vec3.set(this.m_maxBounds, positions[0], positions[1], positions[2]);

    // Find min/max for each axis
    for (let i = 3; i < positions.length; i += 3) {
      const x = positions[i];
      const y = positions[i + 1];
      const z = positions[i + 2];

      if (x < this.m_minBounds[0]) {
        this.m_minBounds[0] = x;
      }
      if (x > this.m_maxBounds[0]) {
        this.m_maxBounds[0] = x;
      }
      if (y < this.m_minBounds[1]) {
        this.m_minBounds[1] = y;
      }
      if (y > this.m_maxBounds[1]) {
        this.m_maxBounds[1] = y;
      }
      if (z < this.m_minBounds[2]) {
        this.m_minBounds[2] = z;
      }
      if (z > this.m_maxBounds[2]) {
        this.m_maxBounds[2] = z;
      }
    }
  }

  // Private method to clear all data
  #clearData() {
    this.m_vertices = [];
    this.m_indices = [];
    this.m_materials = [];
    this.m_textures = [];
    this.m_subMeshes = [];
    this.m_isLoaded = false;
  }
}