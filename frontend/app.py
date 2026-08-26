"""
Streamlit interface for the three-body simulator.

Run it with:   streamlit run app.py

Views:
  - Animation      : the RK4 solution in motion (what the system is doing)
  - Trajectory     : the whole path at once (what happened over the run)
  - Energy         : is the numerical method behaving? (conservation check)
  - Euler vs RK4   : same system, both integrators, side by side

Python computes every trajectory ONCE via simulate(); the browser just replays.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from three_body import simulate

st.set_page_config(page_title="Three-Body Simulator", layout="wide")
st.title("Three-Body Gravity Simulator")
st.caption("Set each body's mass, position and velocity, then run. "
           "Physics: Newtonian gravity. Integrators: Euler and RK4.")

# --- Preset scenarios --------------------------------------------------------
PRESETS = {
    "Two orbiting a heavy body": {
        "masses": [2.0, 1.0, 1.0],
        "positions": [[0.0, 0.0], [1.5, 0.0], [-1.5, 0.0]],
        "velocities": [[0.0, 0.0], [0.0, 0.8], [0.0, -0.8]],
        "dt": 0.001, "steps": 20000, "eps": 0.001,
    },
    "Figure-8 (famous stable orbit)": {
        "masses": [1.0, 1.0, 1.0],
        "positions": [[-0.97000436, 0.24308753],
                      [0.97000436, -0.24308753],
                      [0.0, 0.0]],
        "velocities": [[0.46620369, 0.43236573],
                       [0.46620369, 0.43236573],
                       [-0.93240737, -0.86473146]],
        "dt": 0.0005, "steps": 13000, "eps": 0.0,
    },
}

STATIC_COLORS = ["#185FA5", "#BA7517", "#D85A30"]
GLOW_COLORS = ["#5AD1FF", "#FFC24B", "#FF6B8B"]

# --- Sidebar: preset + simulation settings -----------------------------------
preset_name = st.sidebar.selectbox("Preset scenario", list(PRESETS.keys()))
p = PRESETS[preset_name]

st.sidebar.subheader("Simulation settings")
dt = st.sidebar.number_input("Time step (dt)", value=float(p["dt"]),
                             min_value=1e-5, max_value=0.05, step=1e-4,
                             format="%.5f", key=f"dt_{preset_name}")
steps = st.sidebar.number_input("Steps", value=int(p["steps"]),
                                min_value=1000, max_value=200000, step=1000,
                                key=f"steps_{preset_name}")
eps = st.sidebar.number_input("Softening (eps)", value=float(p["eps"]),
                              min_value=0.0, max_value=0.1, step=1e-3,
                              format="%.4f", key=f"eps_{preset_name}",
                              help="Prevents blow-ups during close encounters. "
                                   "0 = pure Newtonian gravity.")

# --- Body inputs -------------------------------------------------------------
st.subheader("Bodies")
cols = st.columns(3)
masses, positions, velocities = [], [], []
for i in range(3):
    with cols[i]:
        st.markdown(f"**Body {i + 1}**")
        mass = st.number_input("mass", value=float(p["masses"][i]),
                               key=f"m_{i}_{preset_name}")
        x = st.number_input("x", value=float(p["positions"][i][0]),
                            format="%.4f", key=f"x_{i}_{preset_name}")
        y = st.number_input("y", value=float(p["positions"][i][1]),
                            format="%.4f", key=f"y_{i}_{preset_name}")
        vx = st.number_input("vx", value=float(p["velocities"][i][0]),
                             format="%.4f", key=f"vx_{i}_{preset_name}")
        vy = st.number_input("vy", value=float(p["velocities"][i][1]),
                             format="%.4f", key=f"vy_{i}_{preset_name}")
        masses.append(mass)
        positions.append([x, y])
        velocities.append([vx, vy])


# --- Shared drawing routine injected into both animation components ----------
DRAW_JS = r"""
function makeStars(w,h,n){const s=[];for(let i=0;i<n;i++)s.push(
  {x:Math.random()*w,y:Math.random()*h,r:Math.random()*1.1+0.2,a:Math.random()*0.5+0.15});return s;}
function buildBG(D,w,h,dpr,tx,ty,stars){
  const bg=document.createElement('canvas');bg.width=w*dpr;bg.height=h*dpr;
  const bx=bg.getContext('2d');bx.scale(dpr,dpr);
  bx.fillStyle='#05070f';bx.fillRect(0,0,w,h);bx.fillStyle='#fff';
  for(const s of stars){bx.globalAlpha=s.a;bx.beginPath();bx.arc(s.x,s.y,s.r,0,6.29);bx.fill();}
  bx.globalAlpha=0.10;
  for(let b=0;b<N;b++){bx.strokeStyle=COLORS[b];bx.lineWidth=1;bx.beginPath();
    for(let k=0;k<D.length;k+=2){const X=tx(D[k][b][0]),Y=ty(D[k][b][1]);
      if(!isFinite(X)||!isFinite(Y))continue;k===0?bx.moveTo(X,Y):bx.lineTo(X,Y);}bx.stroke();}
  bx.globalAlpha=1;return bg;}
const TRAIL=100;
function draw(ctx,D,bg,f,w,h,tx,ty){
  ctx.drawImage(bg,0,0,w,h);ctx.lineCap='round';
  const ff=Math.min(f,D.length-1);
  for(let b=0;b<N;b++){
    const start=Math.max(0,ff-TRAIL),denom=(ff-start)||1;
    ctx.shadowColor=COLORS[b];ctx.shadowBlur=6;ctx.strokeStyle=COLORS[b];
    for(let k=start;k<ff;k++){const t=(k-start)/denom;ctx.globalAlpha=t*t*0.95;ctx.lineWidth=0.5+t*3.2;
      ctx.beginPath();ctx.moveTo(tx(D[k][b][0]),ty(D[k][b][1]));ctx.lineTo(tx(D[k+1][b][0]),ty(D[k+1][b][1]));ctx.stroke();}
    const hx=tx(D[ff][b][0]),hy=ty(D[ff][b][1]),rad=4+3*Math.cbrt(MASSES[b]);
    ctx.globalAlpha=1;ctx.shadowBlur=24;ctx.fillStyle=COLORS[b];ctx.beginPath();ctx.arc(hx,hy,rad,0,6.29);ctx.fill();
    ctx.shadowBlur=0;ctx.globalAlpha=0.92;ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(hx,hy,rad*0.4,0,6.29);ctx.fill();}
  ctx.shadowBlur=0;ctx.globalAlpha=1;}
function boundsOf(D){let a=Infinity,b=-Infinity,c=Infinity,d=-Infinity;
  for(let f=0;f<D.length;f++)for(let k=0;k<N;k++){const x=D[f][k][0],y=D[f][k][1];
    if(x<a)a=x;if(x>b)b=x;if(y<c)c=y;if(y>d)d=y;}return [a,b,c,d];}
function makeTransform(bounds,w,h,pad){
  const[a,b,c,d]=bounds,sx=(b-a)||1,sy=(d-c)||1;
  const scale=Math.min(w*(1-2*pad)/sx,h*(1-2*pad)/sy),mx=(a+b)/2,my=(c+d)/2;
  return [x=>w/2+(x-mx)*scale, y=>h/2-(y-my)*scale];}
"""

# --- Single (RK4) animation --------------------------------------------------
ANIM_TEMPLATE = r"""
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="controls">
    <button id="playBtn" class="btn">Pause</button>
    <button id="restartBtn" class="btn">Restart</button>
    <div class="speedbox">Speed <input id="speed" type="range" min="1" max="8" value="2"></div>
    <span id="frameLabel"></span>
  </div>
</div>
<style>
  #wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
  #c{display:block;border-radius:14px;box-shadow:0 0 45px rgba(90,209,255,0.14);}
  #controls{display:flex;align-items:center;gap:14px;margin-top:12px;color:#c7d0e0;font-size:14px;}
  .btn{background:#141a2b;color:#dbe6ff;border:1px solid #2a3350;padding:7px 16px;border-radius:9px;cursor:pointer;font-size:14px;}
  .btn:hover{background:#1e2740;border-color:#3d68b0;}
  .speedbox{display:flex;align-items:center;gap:8px;}
  #speed{accent-color:#5AD1FF;}
  #frameLabel{margin-left:auto;color:#5b6785;font-variant-numeric:tabular-nums;}
</style>
<script>
const DATA=__DATA__,MASSES=__MASSES__,COLORS=__COLORS__,N=__NBODY__;
const nFrames=DATA.length,dpr=window.devicePixelRatio||1;
const W=Math.min((window.innerWidth||880)-24,880),H=520;
const canvas=document.getElementById('c');canvas.style.width=W+'px';canvas.style.height=H+'px';
canvas.width=W*dpr;canvas.height=H*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
__DRAW__
const [tx,ty]=makeTransform(boundsOf(DATA),W,H,0.14);
const bg=buildBG(DATA,W,H,dpr,tx,ty,makeStars(W,H,150));
let frame=0,playing=true,speed=2;
const pB=document.getElementById('playBtn'),rB=document.getElementById('restartBtn'),
      sE=document.getElementById('speed'),fL=document.getElementById('frameLabel');
pB.onclick=()=>{playing=!playing;pB.textContent=playing?'Pause':'Play';};
rB.onclick=()=>{frame=0;};sE.oninput=()=>{speed=parseInt(sE.value);};
function loop(){if(playing){frame+=speed;if(frame>=nFrames)frame=0;}const f=Math.floor(frame);
  draw(ctx,DATA,bg,f,W,H,tx,ty);fL.textContent='frame '+f+' / '+nFrames;requestAnimationFrame(loop);}
loop();
</script>
"""

# --- Twin (Euler vs RK4) animation ------------------------------------------
COMPARE_TEMPLATE = r"""
<div id="cwrap">
  <div class="panel"><div class="plabel">__LABEL_A__ <span class="tag">drifts</span></div><canvas id="ca"></canvas></div>
  <div class="panel"><div class="plabel">__LABEL_B__ <span class="tag good">holds</span></div><canvas id="cb"></canvas></div>
</div>
<div id="controls2">
  <button id="playBtn2" class="btn">Pause</button>
  <button id="restartBtn2" class="btn">Restart</button>
  <div class="speedbox">Speed <input id="speed2" type="range" min="1" max="8" value="2"></div>
  <span id="frameLabel2"></span>
</div>
<style>
  #cwrap{display:flex;gap:16px;flex-wrap:wrap;font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
  .panel{display:flex;flex-direction:column;gap:6px;}
  .plabel{color:#9fb2d8;font-size:14px;font-weight:600;}
  .tag{font-weight:500;font-size:12px;color:#ff8f8f;border:1px solid #5c2f3a;border-radius:20px;padding:1px 9px;margin-left:4px;}
  .tag.good{color:#7ff0c0;border-color:#255049;}
  #ca,#cb{display:block;border-radius:12px;box-shadow:0 0 30px rgba(90,209,255,0.10);}
  #controls2{display:flex;align-items:center;gap:14px;margin-top:12px;color:#c7d0e0;font-size:14px;}
  .btn{background:#141a2b;color:#dbe6ff;border:1px solid #2a3350;padding:7px 16px;border-radius:9px;cursor:pointer;font-size:14px;}
  .btn:hover{background:#1e2740;border-color:#3d68b0;}
  .speedbox{display:flex;align-items:center;gap:8px;}
  #speed2{accent-color:#5AD1FF;}
  #frameLabel2{margin-left:auto;color:#5b6785;font-variant-numeric:tabular-nums;}
</style>
<script>
const A=__DATA_A__,B=__DATA_B__,MASSES=__MASSES__,COLORS=__COLORS__,N=__NBODY__;
const nFrames=Math.min(A.length,B.length),dpr=window.devicePixelRatio||1;
const W=Math.min(((window.innerWidth||900)-56)/2,430),H=410;
function setup(id){const cv=document.getElementById(id);cv.style.width=W+'px';cv.style.height=H+'px';
  cv.width=W*dpr;cv.height=H*dpr;const c=cv.getContext('2d');c.scale(dpr,dpr);return c;}
const ctxA=setup('ca'),ctxB=setup('cb');
__DRAW__
// frame the view to the RK4 (true) solution so Euler's drift is visible against it
const [tx,ty]=makeTransform(boundsOf(B),W,H,0.18);
const stars=makeStars(W,H,100);
const bgA=buildBG(A,W,H,dpr,tx,ty,stars),bgB=buildBG(B,W,H,dpr,tx,ty,stars);
let frame=0,playing=true,speed=2;
const pB=document.getElementById('playBtn2'),rB=document.getElementById('restartBtn2'),
      sE=document.getElementById('speed2'),fL=document.getElementById('frameLabel2');
pB.onclick=()=>{playing=!playing;pB.textContent=playing?'Pause':'Play';};
rB.onclick=()=>{frame=0;};sE.oninput=()=>{speed=parseInt(sE.value);};
function loop(){if(playing){frame+=speed;if(frame>=nFrames)frame=0;}const f=Math.floor(frame);
  draw(ctxA,A,bgA,f,W,H,tx,ty);draw(ctxB,B,bgB,f,W,H,tx,ty);
  fL.textContent='frame '+f+' / '+nFrames;requestAnimationFrame(loop);}
loop();
</script>
"""


def _subsample(traj, target):
    n = traj.shape[0]
    idx = np.arange(n) if n <= target else np.linspace(0, n - 1, target).astype(int)
    return idx


def build_anim_html(traj, masses):
    idx = _subsample(traj, 1800)
    return (ANIM_TEMPLATE
            .replace("__DRAW__", DRAW_JS)
            .replace("__DATA__", json.dumps(np.round(traj[idx], 4).tolist()))
            .replace("__MASSES__", json.dumps([float(x) for x in masses]))
            .replace("__COLORS__", json.dumps(GLOW_COLORS))
            .replace("__NBODY__", str(traj.shape[1])))


def build_compare_html(traj_a, traj_b, masses):
    idx = _subsample(traj_b, 1500)
    return (COMPARE_TEMPLATE
            .replace("__DRAW__", DRAW_JS)
            .replace("__DATA_A__", json.dumps(np.round(traj_a[idx], 4).tolist()))
            .replace("__DATA_B__", json.dumps(np.round(traj_b[idx], 4).tolist()))
            .replace("__MASSES__", json.dumps([float(x) for x in masses]))
            .replace("__COLORS__", json.dumps(GLOW_COLORS))
            .replace("__NBODY__", str(traj_b.shape[1]))
            .replace("__LABEL_A__", "Euler")
            .replace("__LABEL_B__", "RK4"))


def drift_pct(e):
    return abs((e[-1] - e[0]) / e[0]) * 100 if e[0] else 0.0


# --- Run ---------------------------------------------------------------------
if st.button("Run simulation", type="primary"):
    with st.spinner("Integrating (RK4 and Euler)..."):
        traj_rk4, e_rk4 = simulate(masses, positions, velocities, dt, int(steps), eps, method="rk4")
        traj_eul, e_eul = simulate(masses, positions, velocities, dt, int(steps), eps, method="euler")

    d_rk4, d_eul = drift_pct(e_rk4), drift_pct(e_eul)

    tab_anim, tab_traj, tab_energy, tab_cmp = st.tabs(
        ["Animation", "Trajectory", "Energy", "Euler vs RK4"])

    # 1) what the system is doing (RK4, the accurate one)
    with tab_anim:
        components.html(build_anim_html(traj_rk4, masses), height=620)
        st.caption("The RK4 solution in motion. Bright dot = a body now; "
                   "the fading comet is where it just came from.")

    # 2) what happened over the whole run
    with tab_traj:
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(3):
            ax.plot(traj_rk4[:, i, 0], traj_rk4[:, i, 1], color=STATIC_COLORS[i],
                    lw=0.8, label=f"Body {i + 1}")
            ax.plot(traj_rk4[0, i, 0], traj_rk4[0, i, 1], "o", color=STATIC_COLORS[i])
        ax.set_aspect("equal")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Full RK4 trajectories")
        st.pyplot(fig)

    # 3) is the numerical method behaving?
    with tab_energy:
        st.metric("RK4 energy drift", f"{d_rk4:.4f}%",
                  help="Change in total energy over the run. Real physics keeps it "
                       "at zero, so a tiny number means an accurate integration.")
        st.markdown("**Total energy over time (RK4)**")
        st.line_chart(e_rk4)
        st.caption("A flat line is proof the simulation is physically honest.")

    # 4) same system, both integrators, side by side
    with tab_cmp:
        st.markdown("**Same initial conditions, same system, two integrators.** "
                    "Watch Euler slowly wander off the true path while RK4 holds it.")
        components.html(build_compare_html(traj_eul, traj_rk4, masses), height=520)

        c1, c2 = st.columns(2)
        c1.metric("Euler energy drift", f"{d_eul:.3f}%")
        c2.metric("RK4 energy drift", f"{d_rk4:.4f}%")

        st.markdown("**Total energy over time — both methods on one axis**")
        st.line_chart(pd.DataFrame({"Euler": e_eul, "RK4": e_rk4}))
        st.caption("Euler bleeds energy step after step; RK4 stays essentially flat. "
                   "Same physics, same inputs — the only difference is the integrator.")
else:
    st.info("Set your values on the left and above, then click **Run simulation**.")