import * as THREE from "three";

export const WORLD = 1100;
const buildingMat = new THREE.MeshStandardMaterial({color:0x2c3540,roughness:0.64,metalness:0.18});
const blueMat = new THREE.MeshStandardMaterial({color:0x2b4c57,roughness:0.52,metalness:0.25});
const runwayMat = new THREE.MeshStandardMaterial({color:0x232b34,roughness:0.68,metalness:0.08});
const groundMat = new THREE.MeshStandardMaterial({color:0x263322,roughness:0.95,metalness:0.02});

export const mats = {
  route: new THREE.LineBasicMaterial({color:0xb894ff,transparent:true,opacity:0.85}),
  safe: new THREE.LineBasicMaterial({color:0x7dffb2,transparent:true,opacity:0.85}),
  caution: new THREE.LineBasicMaterial({color:0xffd36e,transparent:true,opacity:0.8}),
  blocked: new THREE.LineBasicMaterial({color:0xff5c78,transparent:true,opacity:0.7}),
};

function makeLabel(text,x,y,z,color){
  const c=document.createElement("canvas");c.width=256;c.height=48;
  const ctx=c.getContext("2d");
  ctx.fillStyle="rgba(7,10,15,0.55)";ctx.fillRect(4,4,248,40);
  ctx.fillStyle=`#${color.toString(16).padStart(6,"0")}`;
  ctx.font="bold 18px Inter,sans-serif";ctx.textAlign="center";ctx.fillText(text,128,30);
  const tex=new THREE.CanvasTexture(c);tex.colorSpace=THREE.SRGBColorSpace;
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false}));
  s.position.set(x,y,z);s.scale.set(60,12,1);return s;
}

export const landingZones=[
  ["Hospital",-75,-255,0xffd36e],["Campus",150,-245,0x7dffb2],
  ["Suburb",265,60,0x69f7ff],["Market",-280,-90,0xb894ff],
  ["Emergency",255,245,0xff5c78],["Cold Chain",-420,-350,0xffd36e],
  ["Residential",420,-60,0x69f7ff],["Rural Clinic",-470,260,0x7dffb2],
  ["Campus Lab",55,-420,0xb894ff],["Port Dock",430,320,0x8be4e7],
  ["Inspect Tower",10,360,0xf2c879],["Locker Bank",-360,20,0x7cb4ff],
];

export function buildWorld(parent){
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(WORLD,WORLD),groundMat);
  ground.rotation.x=-Math.PI/2;ground.receiveShadow=true;parent.add(ground);
  parent.add(new THREE.GridHelper(WORLD,44,0x365560,0x26333b));
  for(let i=-5;i<=5;i++){
    const r1=new THREE.Mesh(new THREE.BoxGeometry(12,0.16,WORLD),runwayMat);
    r1.position.set(i*65,0.12,0);parent.add(r1);
    const r2=new THREE.Mesh(new THREE.BoxGeometry(WORLD,0.16,12),runwayMat);
    r2.position.set(0,0.12,i*65);parent.add(r2);
  }
  // Instanced buildings
  const boxGeo=new THREE.BoxGeometry(1,1,1);
  const bInst=new THREE.InstancedMesh(boxGeo,buildingMat,130);
  const dummy=new THREE.Object3D();let bIdx=0;
  for(let i=0;i<130;i++){
    const x=-40+(i*53)%380,z=-340+Math.floor(i/18)*48;
    if(Math.abs(x+210)<100&&Math.abs(z-170)<80)continue;
    const h=18+((i*19)%120),w=18+(i%4)*6,d=18+(i%5)*4;
    dummy.position.set(x,h/2,z);dummy.scale.set(w,h,d);dummy.updateMatrix();
    bInst.setMatrixAt(bIdx++,dummy.matrix);
  }
  bInst.count=bIdx;bInst.instanceMatrix.needsUpdate=true;parent.add(bInst);
  // Instanced industrial
  const cylGeo=new THREE.CylinderGeometry(0.5,0.75,1,8);
  const iInst=new THREE.InstancedMesh(cylGeo,buildingMat,32);
  for(let i=0;i<32;i++){
    const h=42+(i%5)*13;
    dummy.position.set(200+(i%8)*28,h/2,120+Math.floor(i/8)*38);
    dummy.scale.set(12,h,12);dummy.updateMatrix();iInst.setMatrixAt(i,dummy.matrix);
  }
  iInst.instanceMatrix.needsUpdate=true;parent.add(iInst);
  // Instanced trees
  const crownGeo=new THREE.ConeGeometry(7,21,6);
  const crownMat2=new THREE.MeshStandardMaterial({color:0x1f5a36,roughness:0.9});
  const cInst=new THREE.InstancedMesh(crownGeo,crownMat2,80);
  for(let i=0;i<80;i++){
    const x=-340+(i*41)%200,z=-340+Math.floor(i/10)*50;
    dummy.position.set(x,22,z);dummy.scale.set(1+(i%3)*0.15,1,1+(i%3)*0.15);
    dummy.updateMatrix();cInst.setMatrixAt(i,dummy.matrix);
  }
  cInst.instanceMatrix.needsUpdate=true;parent.add(cInst);
  // Outer blocks
  const oInst=new THREE.InstancedMesh(boxGeo,buildingMat,60);
  for(let i=0;i<60;i++){
    const h=10+(i%6)*6;
    dummy.position.set(-515+(i%11)*36,h/2,-510+Math.floor(i/11)*40);
    dummy.scale.set(18+(i%5)*4,h,16+(i%4)*4);dummy.updateMatrix();
    oInst.setMatrixAt(i,dummy.matrix);
  }
  oInst.count=60;oInst.instanceMatrix.needsUpdate=true;parent.add(oInst);
  // Comms towers
  const tInst=new THREE.InstancedMesh(cylGeo,blueMat,40);
  for(let i=0;i<40;i++){
    const h=55+(i%4)*12;
    dummy.position.set(360+(i%10)*26,h/2,-440+Math.floor(i/10)*44);
    dummy.scale.set(4,h,4);dummy.updateMatrix();tInst.setMatrixAt(i,dummy.matrix);
  }
  tInst.instanceMatrix.needsUpdate=true;parent.add(tInst);
  buildWarehouse(parent);
  // Weather/No-fly volumes
  const wMat=new THREE.MeshBasicMaterial({color:0x4da3ff,transparent:true,opacity:0.1,depthWrite:false});
  const nMat=new THREE.MeshBasicMaterial({color:0xff5c78,transparent:true,opacity:0.12,depthWrite:false});
  const storm=new THREE.Mesh(new THREE.CylinderGeometry(85,110,220,24,1,true),wMat);
  storm.position.set(170,110,-180);parent.add(storm);
  const noFly=new THREE.Mesh(new THREE.CylinderGeometry(60,72,170,6,1,true),nMat);
  noFly.position.set(-30,85,-210);noFly.rotation.y=Math.PI/6;parent.add(noFly);
  // Landing discs
  const discGeo=new THREE.CylinderGeometry(18,18,1,16);
  landingZones.forEach(([name,x,z,color])=>{
    parent.add(new THREE.Mesh(discGeo,new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.65})));
    parent.children[parent.children.length-1].position.set(x,0.8,z);
    parent.add(makeLabel(name.toUpperCase(),x,28,z+24,color));
  });
  // Corridors as lines
  const corMat=new THREE.LineBasicMaterial({color:0x8be4e7,transparent:true,opacity:0.2});
  [[[-470,72,260],[-210,88,170],[-75,92,-255],[55,88,-420]],
   [[-210,82,170],[20,120,120],[265,96,60],[430,96,320]],
   [[-420,82,-350],[-75,90,-255],[150,100,-245],[420,95,-60]]
  ].forEach(pts=>{
    const c=new THREE.CatmullRomCurve3(pts.map(([x,y,z])=>new THREE.Vector3(x,y,z)));
    parent.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(c.getPoints(30)),corMat));
  });
  parent.add(makeLabel("STORM/WIND SHEAR",170,230,-180,0x7cb4ff));
  parent.add(makeLabel("NO-FLY VOLUME",-30,180,-210,0xf36f7f));
  parent.add(makeLabel("INDUSTRIAL",280,92,160,0xf2c879));
  parent.add(makeLabel("DENSE CITY",135,140,-300,0x8be4e7));
}

const pads=[];
function buildWarehouse(parent){
  const h=new THREE.Group();h.position.set(-210,0,170);parent.add(h);
  const floor=new THREE.Mesh(new THREE.BoxGeometry(190,2,140),runwayMat);
  floor.position.y=1;floor.receiveShadow=true;h.add(floor);
  const roof=new THREE.Mesh(new THREE.BoxGeometry(200,12,148),
    new THREE.MeshStandardMaterial({color:0x182532,roughness:0.38,metalness:0.42,transparent:true,opacity:0.88}));
  roof.position.y=44;h.add(roof);
  const beamGeo=new THREE.BoxGeometry(4,46,4);
  for(let i=0;i<10;i++){
    const b1=new THREE.Mesh(beamGeo,blueMat);b1.position.set(-90+i*20,23,-70);h.add(b1);
    const b2=new THREE.Mesh(beamGeo,blueMat);b2.position.set(-90+i*20,23,70);h.add(b2);
  }
  const padGeo=new THREE.CylinderGeometry(8,8,0.5,16);
  const padMat=new THREE.MeshStandardMaterial({color:0x102733,metalness:0.4,roughness:0.4});
  for(let row=0;row<7;row++)for(let col=0;col<7;col++){
    const p=new THREE.Mesh(padGeo,padMat);p.position.set(-80+col*24,2.1,-52+row*17);h.add(p);pads.push(p);
  }
  h.add(makeLabel("DRONE PORT / WAREHOUSE",0,62,-80,0xf2c879));
}
export {pads};

// Drone mesh — simple for perf, with payload box
const lowGeo=new THREE.SphereGeometry(4,8,6);
export function createDrone(color,isLow=false){
  const g=new THREE.Group();
  if(isLow){g.add(new THREE.Mesh(lowGeo,new THREE.MeshBasicMaterial({color})));return{group:g,isLow:true};}
  const bodyMat=new THREE.MeshStandardMaterial({color,roughness:0.24,metalness:0.68,emissive:color,emissiveIntensity:0.04});
  const body=new THREE.Mesh(new THREE.SphereGeometry(5.2,16,10),bodyMat);
  body.scale.set(1.35,0.52,1);g.add(body);
  const payload=new THREE.Mesh(new THREE.BoxGeometry(5,4,5),new THREE.MeshBasicMaterial({color:0xe9f7ff}));
  payload.position.y=-4.2;payload.visible=false;payload.userData.isPayload=true;g.add(payload);
  const armGeo=new THREE.CylinderGeometry(0.6,0.6,24,6);
  const darkMat=new THREE.MeshStandardMaterial({color:0x111419,roughness:0.35,metalness:0.62});
  const rotorGeo=new THREE.CylinderGeometry(4.5,4.5,0.2,12);
  const rotorMat=new THREE.MeshBasicMaterial({color:0xf7f4eb,transparent:true,opacity:0.22});
  [[13,0,0],[-13,0,0],[0,0,13],[0,0,-13]].forEach(([x,,z])=>{
    const arm=new THREE.Mesh(armGeo,darkMat);
    arm.position.set(x/2,0,z/2);arm.rotation.z=x?Math.PI/2:0;
    if(z)arm.rotation.x=Math.PI/2;g.add(arm);
    const rotor=new THREE.Mesh(rotorGeo,rotorMat);
    rotor.position.set(x,1.2,z);rotor.userData.rotor=true;g.add(rotor);
  });
  // Winch line (hidden by default)
  const winchGeo=new THREE.CylinderGeometry(0.15,0.15,1,4);
  const winchMat=new THREE.MeshBasicMaterial({color:0xf2c879});
  const winch=new THREE.Mesh(winchGeo,winchMat);
  winch.visible=false;winch.userData.isWinch=true;winch.position.y=-4;g.add(winch);
  g.add(new THREE.PointLight(color,0.3,25));
  return{group:g,isLow:false};
}

export function buildSelectedMarker(parent){
  const m=new THREE.Group();
  const ring=new THREE.Mesh(new THREE.TorusGeometry(16,0.6,8,32),
    new THREE.MeshBasicMaterial({color:0xf2c879,transparent:true,opacity:0.85}));
  ring.rotation.x=Math.PI/2;m.add(ring);parent.add(m);return m;
}

export function setupLighting(scene){
  const lights={};
  const hemi=new THREE.HemisphereLight(0xd6e9ff,0x1a2015,1.3);lights.hemi=hemi;scene.add(hemi);
  const sun=new THREE.DirectionalLight(0xffd7a1,3.5);
  sun.position.set(-360,460,300);sun.castShadow=false;
  sun.shadow.mapSize.set(1024,1024);
  sun.shadow.camera.near=1;sun.shadow.camera.far=1400;
  sun.shadow.camera.left=-400;sun.shadow.camera.right=400;
  sun.shadow.camera.top=400;sun.shadow.camera.bottom=-400;
  lights.sun=sun;scene.add(sun);
  const ambient=new THREE.PointLight(0x8be4e7,0.8,600,1.5);
  ambient.position.set(0,80,0);lights.ambient=ambient;scene.add(ambient);
  return lights;
}
