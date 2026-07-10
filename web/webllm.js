// Dynamic import wrapper for MLC's WebLLM.
let _mod = null;

export async function loadWebLLM(modelId, onProgress) {
  if (!_mod) {
    _mod = await import('https://esm.run/@mlc-ai/web-llm@0.2.79');
  }
  if (!navigator.gpu) {
    throw new Error('WebGPU not available in this browser. Use Chrome/Edge 113+ with WebGPU enabled.');
  }
  const engine = await _mod.CreateMLCEngine(modelId, {
    initProgressCallback: onProgress,
  });
  return engine;
}
