// Trace loader with embedded fallback
async function fetchFirst(urls){
  for(const url of urls){
    try{const r=await fetch(url,{cache:"no-store"});if(r.ok)return{data:await r.json(),url};}catch{}
  }
  return{data:null,url:null};
}
export async function loadTrace(task,policy){
  const e=`${task}_${policy}_enriched.json`,raw=`${task}_${policy}_trace.json`;
  const urls=[`../artifacts/traces/${e}`,`/artifacts/traces/${e}`,`artifacts/traces/${e}`,`../artifacts/traces/${raw}`,`/artifacts/traces/${raw}`,`artifacts/traces/${raw}`];
  const{data,url}=await fetchFirst(urls);
  if(data?.frames?.length)return{success:true,mode:"Trace Replay",frames:data.frames,message:`Loaded ${task}/${policy} (${data.frames.length} frames) from ${url}`};
  return{success:false,mode:"Cinematic Simulation",frames:[],message:`Trace not available for ${task}/${policy}. Running Cinematic Simulation.`};
}
