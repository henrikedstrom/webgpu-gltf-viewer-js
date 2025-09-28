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
      const origWidth = hdrData.image.width;
      const origHeight = hdrData.image.height;
      let textureData = hdrData.image.data; // Float32Array
      
      // Check aspect ratio (should be 2:1 for equirectangular)
      if (origWidth !== 2 * origHeight) {
        console.warn(`Warning: Texture aspect ratio is not 2:1. Received: ${origWidth}x${origHeight}`);
      }
      
      // Convert RGB to RGBA (add alpha channel)
      let rgbaData;
      if (textureData.length === origWidth * origHeight * 3) {
        // Convert RGB to RGBA
        rgbaData = new Float32Array(origWidth * origHeight * 4);
        for (let i = 0; i < origWidth * origHeight; i++) {
          rgbaData[i * 4 + 0] = textureData[i * 3 + 0]; // R
          rgbaData[i * 4 + 1] = textureData[i * 3 + 1]; // G
          rgbaData[i * 4 + 2] = textureData[i * 3 + 2]; // B
          rgbaData[i * 4 + 3] = 1.0; // A (full opacity)
        }
      } else {
        // Already RGBA
        rgbaData = textureData;
      }
      
      this.m_texture.m_name = filename;
      this.m_texture.m_width = origWidth;
      this.m_texture.m_height = origHeight;
      this.m_texture.m_components = 4; // RGBA
      this.m_texture.m_data = rgbaData;
      
      console.log(`Loaded environment texture (${origWidth}x${origHeight})`);
      
      // Downsample if texture is larger than 4096x2048
      if (origWidth > 4096) {
        this.#downsampleTexture(origWidth, origHeight);
      }
      
      return true;
    } catch (error) {
      console.error("Failed to load HDR environment:", error);
      return false;
    }
  }

  #downsampleTexture(origWidth, origHeight) {
    console.log(`Downsampling texture from ${origWidth}x${origHeight} to 4096x2048.`);
    const startTime = performance.now();
    
    // Define target resolution (fixed 4096x2048; maintains 2:1 aspect ratio)
    const newWidth = 4096;
    const newHeight = 2048;
    const downsampled = new Float32Array(newWidth * newHeight * 4);
    
    // Compute scale factors from destination to source
    // Subtracting 1 ensures the last pixel maps correctly
    const scaleX = (origWidth - 1) / (newWidth - 1);
    const scaleY = (origHeight - 1) / (newHeight - 1);
    
    // Bilinear downsampling loop
    for (let j = 0; j < newHeight; j++) {
      const origY = j * scaleY;
      const y0 = Math.floor(origY);
      const y1 = Math.min(y0 + 1, origHeight - 1);
      const dy = origY - y0;
      
      for (let i = 0; i < newWidth; i++) {
        const origX = i * scaleX;
        const x0 = Math.floor(origX);
        const x1 = Math.min(x0 + 1, origWidth - 1);
        const dx = origX - x0;
        
        // Process each channel (RGBA)
        for (let c = 0; c < 4; c++) {
          const c00 = this.m_texture.m_data[(y0 * origWidth + x0) * 4 + c];
          const c10 = this.m_texture.m_data[(y0 * origWidth + x1) * 4 + c];
          const c01 = this.m_texture.m_data[(y1 * origWidth + x0) * 4 + c];
          const c11 = this.m_texture.m_data[(y1 * origWidth + x1) * 4 + c];
          
          // Bilinear interpolation: horizontal then vertical
          const top = c00 + dx * (c10 - c00);
          const bottom = c01 + dx * (c11 - c01);
          const value = top + dy * (bottom - top);
          
          downsampled[(j * newWidth + i) * 4 + c] = value;
        }
      }
    }
    
    const endTime = performance.now();
    console.log(`Downsampling took ${((endTime - startTime) / 1000).toFixed(3)} seconds.`);
    
    // Update the texture with downsampled data
    this.m_texture.m_width = newWidth;
    this.m_texture.m_height = newHeight;
    this.m_texture.m_data = downsampled;
  }

  getTexture() { return this.m_texture; }
  getTransform() { return this.m_transform; }
  
  updateRotation(rotationAngle) {
    mat4.identity(this.m_transform);
    mat4.rotateY(this.m_transform, this.m_transform, rotationAngle);
  }
}