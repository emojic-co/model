// Checks a composed Lottie in headless Chrome (canvas renderer): the first or the last frame must match the original
// emoji, nothing may spill past the 128 grid, and something must actually move. Returns readable errors, which
// are fed back to the model.
import { dumpDom, lottieJs } from "./render.ts"

const N = 256
const DIFF = 0.01 // fraction of pixels allowed to differ from the original at the matching end frame
const MOTION = 0.004 // fraction of pixels that must differ from frame 0 at some frame

type Report = { error?: string; first: number; last: number; motion: number; edge: { frame: number; px: number } | null }

export async function validate(svgBody: string, lottie: { op: number }): Promise<string[]> {
  const F = lottie.op
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="${N}" height="${N}">${svgBody}</svg>`
  const script = `
const N=${N}, F=${F}, out=document.getElementById('out');
const fail=e=>{out.textContent=JSON.stringify({error:String(e)})};
const img=new Image();
img.onerror=()=>fail('original svg did not load');
img.onload=()=>{try{
  const oc=document.createElement('canvas');oc.width=oc.height=N;const ox=oc.getContext('2d');ox.drawImage(img,0,0,N,N);
  const orig=ox.getImageData(0,0,N,N).data;
  const c=document.createElement('canvas');c.width=c.height=N;const ctx=c.getContext('2d');
  const a=lottie.loadAnimation({renderer:'canvas',rendererSettings:{context:ctx,clearCanvas:true},loop:false,autoplay:false,animationData:${JSON.stringify(lottie)}});
  a.addEventListener('data_failed',()=>fail('lottie failed to load'));
  a.addEventListener('DOMLoaded',()=>{try{
    const diff=(p,q)=>{let n=0;for(let i=0;i<p.length;i+=4){if(Math.max(Math.abs(p[i]-q[i]),Math.abs(p[i+1]-q[i+1]),Math.abs(p[i+2]-q[i+2]),Math.abs(p[i+3]-q[i+3]))>40)n++}return n/(N*N)};
    const edge=p=>{let n=0;for(let y=0;y<N;y++)for(let x=0;x<N;x++){if((x<2||y<2||x>=N-2||y>=N-2)&&p[(y*N+x)*4+3]>16)n++}return n};
    const at=f=>{a.goToAndStop(f,true);return ctx.getImageData(0,0,N,N).data};
    const f0=at(0), fl=at(F-1);
    const base=edge(orig);let worst=null,motion=0;
    for(let k=0;k<=24;k++){const f=Math.min(F-1,Math.round(k*F/24));const p=at(f);motion=Math.max(motion,diff(f0,p));const e=edge(p);if(e>base+8&&(!worst||e>worst.px))worst={frame:f,px:e}}
    out.textContent=JSON.stringify({first:diff(orig,f0),last:diff(orig,fl),motion,edge:worst});
  }catch(e){fail(e)}});
}catch(e){fail(e)}};
img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(${JSON.stringify(svg)});
setTimeout(()=>{if(!out.textContent)fail('timed out')},6000);`
  const dom = await dumpDom(`<!doctype html><body><pre id="out"></pre><script>${lottieJs()}</script><script>${script}</script>`)
  const m = /<pre id="out">([\s\S]*?)<\/pre>/.exec(dom)
  if (!m || !m[1].trim()) return ["validation page produced no result"]
  if (process.env.ANIMATE_DEBUG) console.log(m[1])
  const r: Report = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&"))
  if (r.error) return [`render failed: ${r.error}`]
  const errors: string[] = []
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`
  if (r.first > DIFF && r.last > DIFF)
    errors.push(`neither the first frame (${pct(r.first)} of pixels differ) nor the last frame (${pct(r.last)} differ) matches the original emoji; one of them must show it exactly`)
  if (r.edge) errors.push(`content spills past the 128 grid edge around frame ${r.edge.frame}`)
  if (r.motion < MOTION) errors.push("there is almost no visible motion; make the animation clearly noticeable")
  return errors
}
