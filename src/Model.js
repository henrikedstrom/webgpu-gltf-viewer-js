import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
const { mat4, vec3, mat3 } = glMatrix;

// Alpha modes
export const AlphaMode = {
  Opaque: 0,
  Mask: 1,
  Blend: 2,
};

export default class Model {
  constructor() {
    // Transformation properties
    this.m_transform = mat4.create();
    this.m_rotationAngle = 0.0;

    // Bounds
    this.m_minBounds = vec3.create();
    this.m_maxBounds = vec3.create();

    // Geometry data
    this.m_vertices = []; // Array of vertex objects
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
    mat4.rotateY(this.m_transform, this.m_transform, -this.m_rotationAngle);
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
      max: this.m_maxBounds,
    };
  }

  getVertices() {
    // Convert vertex objects to Float32Array for WebGPU (full vertex struct)
    const floatCount = this.m_vertices.length * 18; // 18 floats per vertex
    const vertexData = new Float32Array(floatCount);

    for (let i = 0; i < this.m_vertices.length; i++) {
      const vertex = this.m_vertices[i];
      const baseOffset = i * 18;

      // Position (3 floats): offset 0-2
      vertexData[baseOffset + 0] = vertex.position[0];
      vertexData[baseOffset + 1] = vertex.position[1];
      vertexData[baseOffset + 2] = vertex.position[2];

      // Normal (3 floats): offset 3-5
      vertexData[baseOffset + 3] = vertex.normal[0];
      vertexData[baseOffset + 4] = vertex.normal[1];
      vertexData[baseOffset + 5] = vertex.normal[2];

      // Tangent (4 floats): offset 6-9
      vertexData[baseOffset + 6] = vertex.tangent[0];
      vertexData[baseOffset + 7] = vertex.tangent[1];
      vertexData[baseOffset + 8] = vertex.tangent[2];
      vertexData[baseOffset + 9] = vertex.tangent[3];

      // TexCoord0 (2 floats): offset 10-11
      vertexData[baseOffset + 10] = vertex.texCoord0[0];
      vertexData[baseOffset + 11] = vertex.texCoord0[1];

      // TexCoord1 (2 floats): offset 12-13
      vertexData[baseOffset + 12] = vertex.texCoord1[0];
      vertexData[baseOffset + 13] = vertex.texCoord1[1];

      // Color (4 floats): offset 14-17
      vertexData[baseOffset + 14] = vertex.color[0];
      vertexData[baseOffset + 15] = vertex.color[1];
      vertexData[baseOffset + 16] = vertex.color[2];
      vertexData[baseOffset + 17] = vertex.color[3];
    }

    return vertexData;
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

  getTexture(index) {
    if (index >= 0 && index < this.m_textures.length) {
      return this.m_textures[index];
    }
    return null;
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

    // Process the scene with proper transform accumulation
    this.#processScene(gltf);

    // Process materials from the glTF data
    this.#processMaterials(gltf);

    // Process textures from the glTF data
    this.#processTextures(gltf);

    // Recompute bounds
    this.#recomputeBounds();
  }

  // Process the glTF scene recursively
  #processScene(gltf) {
    const scene = gltf.scene;
    const identityMatrix = mat4.create();

    // Process each root node
    scene.children.forEach((rootNode) => {
      this.#processNode(rootNode, identityMatrix);
    });
  }

  // Process a node recursively, accumulating transforms
  #processNode(node, parentTransform) {
    // Compute local transformation matrix
    const localTransform = mat4.create();
    mat4.identity(localTransform);

    // Build transform from Three.js node properties
    if (node.position) {
      mat4.translate(localTransform, localTransform, [
        node.position.x,
        node.position.y,
        node.position.z,
      ]);
    }

    if (node.quaternion) {
      const rotMat = mat4.create();
      mat4.fromQuat(rotMat, [
        node.quaternion.x,
        node.quaternion.y,
        node.quaternion.z,
        node.quaternion.w,
      ]);
      mat4.multiply(localTransform, localTransform, rotMat);
    }

    if (node.scale) {
      mat4.scale(localTransform, localTransform, [
        node.scale.x,
        node.scale.y,
        node.scale.z,
      ]);
    }

    // Combine with parent transform: globalTransform = parentTransform * localTransform
    const globalTransform = mat4.create();
    mat4.multiply(globalTransform, parentTransform, localTransform);

    // If this node has a mesh, process it with the accumulated transform
    if (node.isMesh) {
      this.#processMesh(node, globalTransform);
    }

    // Recursively process children with the accumulated transform
    if (node.children) {
      node.children.forEach((child) => {
        this.#processNode(child, globalTransform);
      });
    }
  }

  // Process a single mesh with accumulated transform
  #processMesh(meshObject, transform) {
    const geometry = meshObject.geometry;

    if (!geometry || !geometry.attributes.position) {
      console.warn("Mesh has no position data, skipping");
      return;
    }

    const attributes = geometry.attributes;
    const indices = geometry.index ? geometry.index.array : null;

    // Extract vertex attributes
    const positions = attributes.position.array;
    const normals = attributes.normal ? attributes.normal.array : null;
    const tangents = attributes.tangent ? attributes.tangent.array : null;
    const texCoord0 = attributes.uv ? attributes.uv.array : null;
    const texCoord1 = attributes.uv1 ? attributes.uv1.array : null;
    const colors = attributes.color ? attributes.color.array : null;

    const vertexCount = positions.length / 3;
    const vertexOffset = this.m_vertices.length;

    // Create submesh
    const subMesh = {
      firstIndex: this.m_indices.length,
      indexCount: 0,
      materialIndex: 0, // Will be updated when materials are processed
      _threeMaterial: meshObject.material || null,
      minBounds: vec3.fromValues(Infinity, Infinity, Infinity),
      maxBounds: vec3.fromValues(-Infinity, -Infinity, -Infinity),
    };

    // Precompute matrices for normal / tangent transform
    const linear3x3 = mat3.create();
    mat3.fromMat4(linear3x3, transform);
    const normalMatrix = mat3.create();
    if (!mat3.invert(normalMatrix, linear3x3)) {
      mat3.identity(normalMatrix); // Fallback if non-invertible (degenerate scale)
    }
    mat3.transpose(normalMatrix, normalMatrix);

    // Process each vertex with proper normal/tangent transforms
    for (let i = 0; i < vertexCount; i++) {
      const posIndex = i * 3;
      const uvIndex = i * 2;
      const tangentIndex = i * 4;
      const colorIndex = i * 4;

      // Position
      const localPos = vec3.fromValues(
        positions[posIndex],
        positions[posIndex + 1],
        positions[posIndex + 2]
      );
      const worldPos = vec3.create();
      vec3.transformMat4(worldPos, localPos, transform);

      // Bounds
      if (worldPos[0] < subMesh.minBounds[0])
        subMesh.minBounds[0] = worldPos[0];
      if (worldPos[0] > subMesh.maxBounds[0])
        subMesh.maxBounds[0] = worldPos[0];
      if (worldPos[1] < subMesh.minBounds[1])
        subMesh.minBounds[1] = worldPos[1];
      if (worldPos[1] > subMesh.maxBounds[1])
        subMesh.maxBounds[1] = worldPos[1];
      if (worldPos[2] < subMesh.minBounds[2])
        subMesh.minBounds[2] = worldPos[2];
      if (worldPos[2] > subMesh.maxBounds[2])
        subMesh.maxBounds[2] = worldPos[2];

      // Normal
      let normalArr;
      if (normals) {
        const n = vec3.fromValues(
          normals[posIndex],
          normals[posIndex + 1],
          normals[posIndex + 2]
        );
        vec3.transformMat3(n, n, normalMatrix);
        vec3.normalize(n, n);
        normalArr = [n[0], n[1], n[2]];
      } else {
        normalArr = [0, 0, 1];
      }

      // Tangent
      let tangentOut;
      if (tangents) {
        const t = vec3.fromValues(
          tangents[tangentIndex],
          tangents[tangentIndex + 1],
          tangents[tangentIndex + 2]
        );
        vec3.transformMat3(t, t, linear3x3);
        vec3.normalize(t, t);
        tangentOut = [t[0], t[1], t[2], tangents[tangentIndex + 3]];
      } else {
        const gen = this.#generateTangent(normals, posIndex);
        const t = vec3.fromValues(gen[0], gen[1], gen[2]);
        vec3.transformMat3(t, t, linear3x3);
        vec3.normalize(t, t);
        tangentOut = [t[0], t[1], t[2], gen[3]];
      }

      const vertex = {
        position: [worldPos[0], worldPos[1], worldPos[2]],
        normal: normalArr,
        tangent: tangentOut,
        texCoord0: texCoord0
          ? [texCoord0[uvIndex], texCoord0[uvIndex + 1]]
          : [0, 0],
        texCoord1: texCoord1
          ? [texCoord1[uvIndex], texCoord1[uvIndex + 1]]
          : [0, 0],
        color: colors
          ? [
              colors[colorIndex] || 1,
              colors[colorIndex + 1] || 1,
              colors[colorIndex + 2] || 1,
              colors[colorIndex + 3] || 1,
            ]
          : [1, 1, 1, 1],
      };
      this.m_vertices.push(vertex);
    }

    // Process indices
    if (indices) {
      for (let i = 0; i < indices.length; i++) {
        this.m_indices.push(vertexOffset + indices[i]);
      }
      subMesh.indexCount = indices.length;
    } else {
      // Non-indexed mesh: generate sequential indices
      for (let i = 0; i < vertexCount; i++) {
        this.m_indices.push(vertexOffset + i);
      }
      subMesh.indexCount = vertexCount;
    }

    this.m_subMeshes.push(subMesh);
  }

  // Process materials from glTF
  #processMaterials(gltf) {
    this.m_materials = [];
    this._materialIndexMap = new Map();

    // Extract materials from Three.js scene
    const materialMap = this._materialIndexMap; // Track unique materials

    gltf.scene.traverse((object) => {
      if (object.isMesh && object.material) {
        const threeMaterial = object.material;

        // Skip if we've already processed this material
        if (materialMap.has(threeMaterial)) {
          return;
        }

        // Extract material properties from Three.js material
        const material = {
          // Base color factor (vec4)
          baseColorFactor: threeMaterial.color
            ? [
                threeMaterial.color.r,
                threeMaterial.color.g,
                threeMaterial.color.b,
                threeMaterial.opacity !== undefined
                  ? threeMaterial.opacity
                  : 1.0,
              ]
            : [1.0, 1.0, 1.0, 1.0],

          // Emissive factor (vec3)
          emissiveFactor: threeMaterial.emissive
            ? [
                threeMaterial.emissive.r,
                threeMaterial.emissive.g,
                threeMaterial.emissive.b,
              ]
            : [0.0, 0.0, 0.0],

          // PBR properties
          metallicFactor:
            threeMaterial.metalness !== undefined
              ? threeMaterial.metalness
              : 0.0,
          roughnessFactor:
            threeMaterial.roughness !== undefined
              ? threeMaterial.roughness
              : 0.5,
          normalScale: threeMaterial.normalScale
            ? threeMaterial.normalScale.x
            : 1.0,
          occlusionStrength: 1.0, // Three.js doesn't expose this directly

          // Alpha properties
          alphaMode: this.#getAlphaMode(threeMaterial),
          alphaCutoff:
            threeMaterial.alphaTest !== undefined
              ? threeMaterial.alphaTest
              : 0.5,
          doubleSided: threeMaterial.side === 2, // THREE.DoubleSide = 2

          // Texture indices (will be set when processing textures)
          baseColorTexture: -1,
          metallicRoughnessTexture: -1,
          normalTexture: -1,
          emissiveTexture: -1,
          occlusionTexture: -1,
        };

        this.m_materials.push(material);
        materialMap.set(threeMaterial, this.m_materials.length - 1);
      }
    });

    // If no materials found, create a default one
    if (this.m_materials.length === 0) {
      console.log("No materials found in model, using default");
      this.m_materials.push(this.#createDefaultMaterial());
    }

    // Assign material indices to submeshes now that materials are collected
    for (const sm of this.m_subMeshes) {
      if (sm._threeMaterial && materialMap.has(sm._threeMaterial)) {
        sm.materialIndex = materialMap.get(sm._threeMaterial);
      } else {
        sm.materialIndex = 0; // fallback
      }
      delete sm._threeMaterial; // cleanup
    }
  }

  // Process textures from glTF
  #processTextures(gltf) {
    this.m_textures = [];
    const textureMap = new Map(); // Track unique textures
    const materialIndexMap = this._materialIndexMap || new Map();

    // Process each material to extract all texture types
    gltf.scene.traverse((object) => {
      if (object.isMesh && object.material) {
        const material = object.material;
        const modelMaterialIndex = materialIndexMap.get(material);

        // Define texture types to extract
        const textureTypes = [
          { prop: "map", name: "baseColor", type: "baseColor" },
          { prop: "normalMap", name: "normal", type: "normal" },
          {
            prop: "roughnessMap",
            name: "metallicRoughness",
            type: "metallicRoughness",
          }, // Three.js uses separate maps
          {
            prop: "metalnessMap",
            name: "metallicRoughness",
            type: "metallicRoughness",
          },
          { prop: "emissiveMap", name: "emissive", type: "emissive" },
          { prop: "aoMap", name: "occlusion", type: "occlusion" },
        ];

        textureTypes.forEach(({ prop, name, type }) => {
          const threeTexture = material[prop];
          if (threeTexture && threeTexture.image) {
            // Add texture if not already tracked
            if (!textureMap.has(threeTexture)) {
              const texture = {
                name: threeTexture.name || name,
                type: type,
                width: threeTexture.image.width,
                height: threeTexture.image.height,
                components: 4,
                image: threeTexture.image,
                threeTexture: threeTexture,
              };
              this.m_textures.push(texture);
              textureMap.set(threeTexture, this.m_textures.length - 1);
            }

            // Assign texture index to the corresponding material struct
            if (modelMaterialIndex !== undefined) {
              const texIndex = textureMap.get(threeTexture);
              const modelMaterial = this.m_materials[modelMaterialIndex];
              if (modelMaterial) {
                switch (type) {
                  case "baseColor":
                    modelMaterial.baseColorTexture = texIndex;
                    break;
                  case "normal":
                    modelMaterial.normalTexture = texIndex;
                    break;
                  case "metallicRoughness":
                    // glTF packs both metallic & roughness in one texture; we map either source map here
                    modelMaterial.metallicRoughnessTexture = texIndex;
                    break;
                  case "emissive":
                    modelMaterial.emissiveTexture = texIndex;
                    break;
                  case "occlusion":
                    modelMaterial.occlusionTexture = texIndex;
                    break;
                }
              }
            }
          } else if (threeTexture) {
            console.log(`Texture ${prop} exists but has no image data`);
          }
        });
      }
    });
  }

  // Get alpha mode from material
  #getAlphaMode(material) {
    if (material.transparent) {
      return AlphaMode.Blend;
    } else if (material.alphaTest > 0) {
      return AlphaMode.Mask;
    } else {
      return AlphaMode.Opaque;
    }
  }

  // Create default material
  #createDefaultMaterial() {
    return {
      baseColorFactor: [1.0, 1.0, 1.0, 1.0],
      emissiveFactor: [0.0, 0.0, 0.0],
      metallicFactor: 1.0,
      roughnessFactor: 1.0,
      normalScale: 1.0,
      occlusionStrength: 1.0,
      alphaMode: AlphaMode.Opaque,
      alphaCutoff: 0.5,
      doubleSided: false,
      baseColorTexture: -1,
      metallicRoughnessTexture: -1,
      normalTexture: -1,
      emissiveTexture: -1,
      occlusionTexture: -1,
    };
  }

  // Recompute bounds from vertex data
  #recomputeBounds() {
    vec3.set(this.m_minBounds, Infinity, Infinity, Infinity);
    vec3.set(this.m_maxBounds, -Infinity, -Infinity, -Infinity);

    // Calculate bounds from transformed vertices
    for (const vertex of this.m_vertices) {
      const pos = vertex.position;

      if (pos[0] < this.m_minBounds[0]) this.m_minBounds[0] = pos[0];
      if (pos[0] > this.m_maxBounds[0]) this.m_maxBounds[0] = pos[0];
      if (pos[1] < this.m_minBounds[1]) this.m_minBounds[1] = pos[1];
      if (pos[1] > this.m_maxBounds[1]) this.m_maxBounds[1] = pos[1];
      if (pos[2] < this.m_minBounds[2]) this.m_minBounds[2] = pos[2];
      if (pos[2] > this.m_maxBounds[2]) this.m_maxBounds[2] = pos[2];
    }
  }

  // Generate a simple tangent when not provided
  #generateTangent(normals, posIndex) {
    if (!normals) {
      return [1, 0, 0, 1]; // Default tangent
    }

    // Get the normal for this vertex
    const normal = [
      normals[posIndex],
      normals[posIndex + 1],
      normals[posIndex + 2],
    ];

    // Generate a tangent perpendicular to the normal
    // Choose the axis that's most different from the normal
    let tangent;
    if (Math.abs(normal[0]) < 0.9) {
      tangent = [1, 0, 0]; // Use X axis
    } else {
      tangent = [0, 1, 0]; // Use Y axis
    }

    // Make tangent perpendicular to normal using Gram-Schmidt
    const dot =
      tangent[0] * normal[0] + tangent[1] * normal[1] + tangent[2] * normal[2];
    tangent[0] -= dot * normal[0];
    tangent[1] -= dot * normal[1];
    tangent[2] -= dot * normal[2];

    // Normalize
    const length = Math.sqrt(
      tangent[0] * tangent[0] +
        tangent[1] * tangent[1] +
        tangent[2] * tangent[2]
    );
    if (length > 0.001) {
      tangent[0] /= length;
      tangent[1] /= length;
      tangent[2] /= length;
    }

    return [tangent[0], tangent[1], tangent[2], 1.0]; // w=1 for handedness
  }

  // Private method to clear all data
  #clearData() {
    mat4.identity(this.m_transform);
    this.m_rotationAngle = 0.0;
    vec3.set(this.m_minBounds, Infinity, Infinity, Infinity);
    vec3.set(this.m_maxBounds, -Infinity, -Infinity, -Infinity);
    this.m_vertices = [];
    this.m_indices = [];
    this.m_materials = [];
    this.m_textures = [];
    this.m_subMeshes = [];
    this.m_isLoaded = false;
  }
}
