const { mat4 } = glMatrix;
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export default class Environment {
  constructor() {
    this.m_transform = mat4.create();
    this.m_texture = {
      m_name: "",
      m_width: 0,
      m_height: 0,
      m_components: 0,
      m_data: null // Float32Array for HDR data
    };
  }

  async load(filename) {
    try {
      const loader = new RGBELoader();
      
      const hdrData = await new Promise((resolve, reject) => {
        loader.load(filename, resolve, undefined, reject);
      });
      
      // Extract data from Three.js texture
      this.m_texture.m_name = filename;
      this.m_texture.m_width = hdrData.image.width;
      this.m_texture.m_height = hdrData.image.height;
      this.m_texture.m_components = 3;
      this.m_texture.m_data = hdrData.image.data;

      console.log("Loaded environment texture:", this.m_texture);
      
      return true;
    } catch (error) {
      console.error("Failed to load HDR environment:", error);
      return false;
    }
  }

  getTexture() { return this.m_texture; }
  getTransform() { return this.m_transform; }
  
  updateRotation(rotationAngle) {
    mat4.identity(this.m_transform);
    mat4.rotateY(this.m_transform, this.m_transform, rotationAngle);
  }
}