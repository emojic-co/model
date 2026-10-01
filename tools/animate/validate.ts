// Syntax check only: the composed Lottie must load and render every sampled frame in headless Chrome (lottie-web,
// canvas renderer) without errors. Nothing about how it looks is judged. Errors go back to the model.
import { dumpDom, lottieJs } from "./render.ts"

export async function validate(lottie: { op: number }): Promise<string[]> {
  const F = lottie.op
  const script = `
const F=${F}, out=document.getElementById('out');
const fail=e=>{out.textContent=JSON.stringify({error:String((e&&e.message)||e)})};
try{
  const c=document.createElement('canvas');c.width=c.height=128;const ctx=c.getContext('2d');
  const a=lottie.loadAnimation({renderer:'canvas',rendererSettings:{context:ctx,clearCanvas:true},loop:false,autoplay:false,animationData:${JSON.stringify(lottie)}});
  a.addEventListener('data_failed',()=>fail('lottie failed to load'));
  a.addEventListener('DOMLoaded',()=>{try{for(let k=0;k<=24;k++)a.goToAndStop(Math.min(F-1,Math.round(k*F/24)),true);out.textContent=JSON.stringify({})}catch(e){fail(e)}});
}catch(e){fail(e)}
setTimeout(()=>{if(!out.textContent)fail('timed out loading the animation')},6000);`
  const dom = await dumpDom(`<!doctype html><body><pre id="out"></pre><script>${lottieJs()}</script><script>${script}</script>`)
  const m = /<pre id="out">([\s\S]*?)<\/pre>/.exec(dom)
  if (!m || !m[1].trim()) return ["validation page produced no result"]
  const r: { error?: string } = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&"))
  return r.error ? [`render failed: ${r.error}`] : []
}
