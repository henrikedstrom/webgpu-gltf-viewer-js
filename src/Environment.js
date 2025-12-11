const { mat4 } = glMatrix;
import { FloatType } from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export default class Environment {
  // Private field declarations
  #transform;
  #texture;

  constructor() {
    this.#transform = mat4.create();
    this.#texture = {
      name: "",
      width: 0,
      height: 0,
      components: 0,
      data: null // Float32Array for HDR data
    };
  }

  async load(filename) {
    const t0 = performance.now();

    try {
      const loader = new RGBELoader();
      loader.setDataType(FloatType); // Load as Float32Array (convert to float16 later on the GPU)
      
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
        // Already RGBA - but ensure it's a Float32Array
        if (textureData.constructor.name !== 'Float32Array') {
          rgbaData = new Float32Array(textureData);
        } else {
          rgbaData = textureData;
        }
      }
      
      this.#texture.name = filename;
      this.#texture.width = origWidth;
      this.#texture.height = origHeight;
      this.#texture.components = 4; // RGBA
      this.#texture.data = rgbaData;
      
      console.log(`Loaded environment texture (${origWidth}x${origHeight}) in ${(performance.now() - t0).toFixed(2)} ms`);
      
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
          const c00 = this.#texture.data[(y0 * origWidth + x0) * 4 + c];
          const c10 = this.#texture.data[(y0 * origWidth + x1) * 4 + c];
          const c01 = this.#texture.data[(y1 * origWidth + x0) * 4 + c];
          const c11 = this.#texture.data[(y1 * origWidth + x1) * 4 + c];
          
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
    this.#texture.width = newWidth;
    this.#texture.height = newHeight;
    this.#texture.data = downsampled;
  }

  getTexture() { return this.#texture; }
  getTransform() { return this.#transform; }
  
  updateRotation(rotationAngle) {
    mat4.identity(this.#transform);
    mat4.rotateY(this.#transform, this.#transform, rotationAngle);
  }
}