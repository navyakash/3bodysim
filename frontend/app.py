"""
Streamlit interface for the three-body simulator.

Run it with:   streamlit run app.py

Views:
  - Animation      : the RK4 solution in motion
  - Trajectory     : the whole path at once
  - Energy         : is the numerical method behaving? (conservation check)
  - Euler vs RK4   : same system, both integrators, side by side
  - Chaos          : same system, RK4 both times, but one started 1e-6 different

Python computes every trajectory ONCE via simulate(); the browser just replays.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from three_body import (simulate, separation, estimate_lyapunov,
                        energy_series, momentum_series, angular_momentum_series,
                        detect_events, potential_grid)
from survey import run_stability_map, DEFAULT_SCENARIO
from classifier import (build_dataset, train_fate_model, predict_grid,
                        predict_one, CLASSES as FATE_CLASSES)

st.set_page_config(page_title="Three-Body Simulator", layout="wide")
st.title("Three-Body Gravity Simulator")
st.caption("Newtonian gravity, integrated with Euler / RK4. "
           "Set the bodies, run, and explore the tabs.")

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
    "Pythagorean (chaotic)": {
        "masses": [3.0, 4.0, 5.0],
        "positions": [[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]],
        "velocities": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "dt": 0.001, "steps": 45000, "eps": 0.05,
    },
}

STATIC_COLORS = ["#185FA5", "#BA7517", "#D85A30"]
GLOW_COLORS = ["#5AD1FF", "#FFC24B", "#FF6B8B"]

# --- Sidebar -----------------------------------------------------------------
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
                              min_value=0.0, max_value=0.2, step=1e-3,
                              format="%.4f", key=f"eps_{preset_name}",
                              help="Prevents blow-ups during close encounters. "
                                   "0 = pure Newtonian gravity.")
pert = st.sidebar.number_input("Chaos: perturbation size", value=1e-6,
                               min_value=1e-9, max_value=1e-1, step=1e-6,
                               format="%.1e", key="pert",
                               help="How far the second ('perturbed') run's "
                                    "starting position is nudged, for the Chaos tab.")

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


# --- Shared canvas-drawing JS (injected into both animation components) -------
DRAW_JS = r"""
function makeStars(w,h,n){const s=[];for(let i=0;i<n;i++)s.push(
  {x:Math.random()*w,y:Math.random()*h,r:Math.random()*1.1+0.2,a:Math.random()*0.5+0.15});return s;}
function makeTransform(bounds,w,h,pad){const[a,b,c,d]=bounds,sx=(b-a)||1,sy=(d-c)||1;
  const scale=Math.min(w*(1-2*pad)/sx,h*(1-2*pad)/sy),mx=(a+b)/2,my=(c+d)/2;
  return [x=>w/2+(x-mx)*scale, y=>h/2-(y-my)*scale];}
function boundsOf(D){let a=Infinity,b=-Infinity,c=Infinity,d=-Infinity;
  for(let f=0;f<D.length;f++)for(let k=0;k<N;k++){const x=D[f][k][0],y=D[f][k][1];
    if(isFinite(x)){if(x<a)a=x;if(x>b)b=x;}if(isFinite(y)){if(y<c)c=y;if(y>d)d=y;}}return [a,b,c,d];}
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
    if(!isFinite(hx)||!isFinite(hy))continue;
    ctx.globalAlpha=1;ctx.shadowBlur=24;ctx.fillStyle=COLORS[b];ctx.beginPath();ctx.arc(hx,hy,rad,0,6.29);ctx.fill();
    ctx.shadowBlur=0;ctx.globalAlpha=0.92;ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(hx,hy,rad*0.4,0,6.29);ctx.fill();}
  ctx.shadowBlur=0;ctx.globalAlpha=1;}
"""

# --- Single animation --------------------------------------------------------
ANIM_TEMPLATE = r"""
<div id="wrap"><canvas id="c"></canvas>
  <div id="controls">
    <button id="playBtn" class="btn">Pause</button>
    <button id="restartBtn" class="btn">Restart</button>
    <div class="speedbox">Speed <input id="speed" type="range" min="1" max="8" value="2"></div>
    <span id="frameLabel"></span>
  </div></div>
<style>
  #wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
  #c{display:block;border-radius:14px;box-shadow:0 0 45px rgba(90,209,255,0.14);}
  #controls{display:flex;align-items:center;gap:14px;margin-top:12px;color:#c7d0e0;font-size:14px;}
  .btn{background:#141a2b;color:#dbe6ff;border:1px solid #2a3350;padding:7px 16px;border-radius:9px;cursor:pointer;font-size:14px;}
  .btn:hover{background:#1e2740;border-color:#3d68b0;}
  .speedbox{display:flex;align-items:center;gap:8px;} #speed{accent-color:#5AD1FF;}
  #frameLabel{margin-left:auto;color:#5b6785;font-variant-numeric:tabular-nums;}
</style>
<script>
const DATA=__DATA__,MASSES=__MASSES__,COLORS=__COLORS__,N=__NBODY__,BOUNDS=__BOUNDS__;
const nFrames=DATA.length,dpr=window.devicePixelRatio||1;
const W=Math.min((window.innerWidth||880)-24,880),H=520;
const canvas=document.getElementById('c');canvas.style.width=W+'px';canvas.style.height=H+'px';
canvas.width=W*dpr;canvas.height=H*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
__DRAW__
const [tx,ty]=makeTransform(BOUNDS,W,H,0.14);
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

# --- Twin animation (used by Euler-vs-RK4 AND Chaos) -------------------------
TWIN_TEMPLATE = r"""
<div id="cwrap">
  <div class="panel"><div class="plabel">__LABEL_A__ <span class="__CLASS_A__">__TAG_A__</span></div><canvas id="ca"></canvas></div>
  <div class="panel"><div class="plabel">__LABEL_B__ <span class="__CLASS_B__">__TAG_B__</span></div><canvas id="cb"></canvas></div>
</div>
<div id="controls2">
  <button id="playBtn2" class="btn">Pause</button>
  <button id="restartBtn2" class="btn">Restart</button>
  <div class="speedbox">Speed <input id="speed2" type="range" min="1" max="8" value="2"></div>
  <span id="frameLabel2"></span>
</div>
<style>
  #cwrap{display:flex;gap:16px;flex-wrap:wrap;font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
  .panel{display:flex;flex-direction:column;gap:6px;} .plabel{color:#9fb2d8;font-size:14px;font-weight:600;}
  .tag{font-weight:500;font-size:12px;color:#ff8f8f;border:1px solid #5c2f3a;border-radius:20px;padding:1px 9px;margin-left:4px;}
  .tag.good{color:#7ff0c0;border-color:#255049;} .tag.neutral{color:#8fc2ff;border-color:#26426b;}
  #ca,#cb{display:block;border-radius:12px;box-shadow:0 0 30px rgba(90,209,255,0.10);}
  #controls2{display:flex;align-items:center;gap:14px;margin-top:12px;color:#c7d0e0;font-size:14px;}
  .btn{background:#141a2b;color:#dbe6ff;border:1px solid #2a3350;padding:7px 16px;border-radius:9px;cursor:pointer;font-size:14px;}
  .btn:hover{background:#1e2740;border-color:#3d68b0;} .speedbox{display:flex;align-items:center;gap:8px;} #speed2{accent-color:#5AD1FF;}
  #frameLabel2{margin-left:auto;color:#5b6785;font-variant-numeric:tabular-nums;}
</style>
<script>
const A=__DATA_A__,B=__DATA_B__,MASSES=__MASSES__,COLORS=__COLORS__,N=__NBODY__,BOUNDS=__BOUNDS__;
const nFrames=Math.min(A.length,B.length),dpr=window.devicePixelRatio||1;
const W=Math.min(((window.innerWidth||900)-56)/2,430),H=410;
function setup(id){const cv=document.getElementById(id);cv.style.width=W+'px';cv.style.height=H+'px';
  cv.width=W*dpr;cv.height=H*dpr;const c=cv.getContext('2d');c.scale(dpr,dpr);return c;}
const ctxA=setup('ca'),ctxB=setup('cb');
__DRAW__
const [tx,ty]=makeTransform(BOUNDS,W,H,0.18);
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


def _idx(traj, target):
    n = traj.shape[0]
    return np.arange(n) if n <= target else np.linspace(0, n - 1, target).astype(int)


def framing_bounds(*trajs, q=82):
    """A robust SQUARE view box centered on the median position, with radius set
    by the q-th percentile of distance from center. Keeps the central action
    legible while letting rare outliers (an ejected body) fly off-frame."""
    pts = np.concatenate([np.asarray(t).reshape(-1, 2) for t in trajs], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    c = np.median(pts, axis=0)
    r = max(float(np.percentile(np.abs(pts - c).max(axis=1), q)), 1e-6)
    return [float(c[0] - r), float(c[0] + r), float(c[1] - r), float(c[1] + r)]


def build_anim_html(traj, masses):
    idx = _idx(traj, 1800)
    return (ANIM_TEMPLATE
            .replace("__DRAW__", DRAW_JS)
            .replace("__DATA__", json.dumps(np.round(traj[idx], 4).tolist()))
            .replace("__MASSES__", json.dumps([float(x) for x in masses]))
            .replace("__COLORS__", json.dumps(GLOW_COLORS))
            .replace("__NBODY__", str(traj.shape[1]))
            .replace("__BOUNDS__", json.dumps(framing_bounds(traj))))


def build_twin_html(traj_a, traj_b, masses, bounds,
                    label_a, label_b, tag_a, tag_b, class_a, class_b):
    idx = _idx(traj_b, 1500)
    return (TWIN_TEMPLATE
            .replace("__DRAW__", DRAW_JS)
            .replace("__DATA_A__", json.dumps(np.round(traj_a[idx], 4).tolist()))
            .replace("__DATA_B__", json.dumps(np.round(traj_b[idx], 4).tolist()))
            .replace("__MASSES__", json.dumps([float(x) for x in masses]))
            .replace("__COLORS__", json.dumps(GLOW_COLORS))
            .replace("__NBODY__", str(traj_b.shape[1]))
            .replace("__BOUNDS__", json.dumps(bounds))
            .replace("__LABEL_A__", label_a).replace("__LABEL_B__", label_b)
            .replace("__TAG_A__", tag_a).replace("__TAG_B__", tag_b)
            .replace("__CLASS_A__", class_a).replace("__CLASS_B__", class_b))


def drift_pct(e):
    return abs((e[-1] - e[0]) / e[0]) * 100 if e[0] else 0.0


# --- Run ---------------------------------------------------------------------
if st.button("Run simulation", type="primary"):
    with st.spinner("Integrating (RK4, Euler, and a perturbed twin)..."):
        t0 = time.perf_counter()
        pos_rk4, vel_rk4 = simulate(masses, positions, velocities, dt, int(steps), eps, "rk4")
        time_rk4 = time.perf_counter() - t0

        t0 = time.perf_counter()
        pos_eul, vel_eul = simulate(masses, positions, velocities, dt, int(steps), eps, "euler")
        time_eul = time.perf_counter() - t0

        pos_p = [list(row) for row in positions]
        pos_p[0][0] += pert                          # nudge body 1's x by `pert`
        pos_pert, _ = simulate(masses, pos_p, velocities, dt, int(steps), eps, "rk4")

    # conserved quantities for the primary (RK4) run
    E = energy_series(pos_rk4, vel_rk4, masses, eps)
    P = momentum_series(vel_rk4, masses)             # (steps, 2)
    L = angular_momentum_series(pos_rk4, vel_rk4, masses)
    # ...and for Euler, for the head-to-head
    E_e = energy_series(pos_eul, vel_eul, masses, eps)
    P_e = momentum_series(vel_eul, masses)

    # chaos
    sep = separation(pos_rk4, pos_pert)
    lam = estimate_lyapunov(sep, dt)
    horizon = np.argmax(sep > 1.0)
    t_horizon = horizon * dt if horizon > 0 else None

    ii = _idx(E, 2000)                               # shared chart downsample index

    def _relerr(series):
        a, b = series[0], series[-1]
        return abs((b - a) / a) if a != 0 else float("nan")

    tabs = st.tabs(["Animation", "Trajectory", "Conservation", "Euler vs RK4",
                    "Chaos", "Events", "Field"])

    # 1) Animation
    with tabs[0]:
        components.html(build_anim_html(pos_rk4, masses), height=620)
        st.caption("The RK4 solution in motion. Bright dot = a body now; "
                   "the fading comet is where it just came from.")

    # 2) Trajectory
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(3):
            ax.plot(pos_rk4[:, i, 0], pos_rk4[:, i, 1], color=STATIC_COLORS[i],
                    lw=0.8, label=f"Body {i + 1}")
            ax.plot(pos_rk4[0, i, 0], pos_rk4[0, i, 1], "o", color=STATIC_COLORS[i])
        ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Full RK4 trajectories")
        st.pyplot(fig)

    # 3) Conservation
    with tabs[2]:
        st.markdown("**Energy vs time**")
        st.line_chart(E[ii])
        st.markdown("**Linear momentum vs time (x and y components)**")
        st.line_chart(pd.DataFrame({"px": P[ii, 0], "py": P[ii, 1]}))
        st.markdown("**Angular momentum vs time**")
        st.line_chart(L[ii])

        P0, Pf = np.linalg.norm(P[0]), np.linalg.norm(P[-1])
        p_drift = np.linalg.norm(P[-1] - P[0])
        p_rel = f"{p_drift / P0:.2e}" if P0 > 1e-9 else f"{p_drift:.2e} (abs)"
        diag = pd.DataFrame(
            [["Energy",            f"{E[0]:.5g}", f"{E[-1]:.5g}", f"{_relerr(E):.2e}"],
             ["|Linear momentum|", f"{P0:.3g}",   f"{Pf:.3g}",    p_rel],
             ["Angular momentum",  f"{L[0]:.5g}", f"{L[-1]:.5g}", f"{_relerr(L):.2e}"]],
            columns=["Quantity", "Initial", "Final", "Relative error"]).set_index("Quantity")
        st.markdown("**Conservation diagnostics**")
        st.table(diag)
        st.caption("Real physics holds all three constant. Linear momentum is conserved to "
                   "machine precision by construction (the pairwise forces cancel); energy and "
                   "angular momentum reveal how good the integrator is. The Pythagorean preset "
                   "is a hard, close-encounter case and drifts more than the gentle ones.")

    # 4) Euler vs RK4
    with tabs[3]:
        st.markdown("**Same initial conditions, two integrators.** "
                    "Watch Euler wander off the true path while RK4 holds it.")
        components.html(build_twin_html(
            pos_eul, pos_rk4, masses, framing_bounds(pos_rk4),
            "Euler", "RK4", "drifts", "holds", "tag", "tag good"), height=520)

        traj_diff = separation(pos_eul, pos_rk4)[-1]
        pdrift_e = np.linalg.norm(P_e[-1] - P_e[0])
        pdrift_r = np.linalg.norm(P[-1] - P[0])
        table = pd.DataFrame({
            "Metric": ["Timestep (dt)", "Steps", "Compute time (s)",
                       "Energy error (%)", "Momentum drift",
                       "Trajectory diff vs RK4 (final)"],
            "Euler": [f"{dt:g}", f"{int(steps)}", f"{time_eul:.3f}",
                      f"{_relerr(E_e) * 100:.3f}", f"{pdrift_e:.1e}", f"{traj_diff:.3f}"],
            "RK4": [f"{dt:g}", f"{int(steps)}", f"{time_rk4:.3f}",
                    f"{_relerr(E) * 100:.4f}", f"{pdrift_r:.1e}", "0 (reference)"],
        }).set_index("Metric")
        st.markdown("**Euler vs RK4 - head to head**")
        st.table(table)
        st.markdown("**Total energy over time - both methods on one axis**")
        st.line_chart(pd.DataFrame({"Euler": E_e[ii], "RK4": E[ii]}))
        st.caption("Both methods conserve momentum exactly, so energy error and the trajectory "
                   "difference are what set them apart - RK4 for a little more compute per step.")

    # 5) Chaos
    with tabs[4]:
        st.markdown(
            f"**Two RK4 runs of the *same* system.** The only difference: the "
            f"'Perturbed' run started with body 1 nudged by **{pert:g}** in x - "
            f"about one part in {int(1 / pert):,} of the system size. They track "
            f"together, then split into completely different futures.")
        components.html(build_twin_html(
            pos_rk4, pos_pert, masses, framing_bounds(pos_rk4, pos_pert),
            "Original", "Perturbed", "start x\u2080", f"x\u2080 + {pert:g}",
            "tag neutral", "tag neutral"), height=520)

        m1, m2, m3 = st.columns(3)
        m1.metric("Perturbation", f"{pert:g}")
        m2.metric("Lyapunov estimate",
                  f"{lam:.3f}/time" if lam is not None else "-",
                  help="Slope of ln(separation) vs time in the growth region. "
                       "Positive = exponential divergence = chaos.")
        m3.metric("Predictability horizon",
                  f"t \u2248 {t_horizon:.1f}" if t_horizon else "-",
                  help="When the tiny difference has grown to order 1 - beyond "
                       "this, prediction is effectively impossible.")

        st.markdown("**Separation between the two systems over time (log scale)**")
        si = _idx(sep, 2000)
        tt = si * dt
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.semilogy(tt, np.maximum(sep[si], 1e-12), color="#3B7DD8", lw=1.3)
        ax2.axhline(pert, ls="--", color="#999", lw=1, label="initial difference")
        ax2.axhline(1.0, ls=":", color="#C0563A", lw=1, label="order-1 (macroscopic)")
        ax2.set_xlabel("time"); ax2.set_ylabel("separation (log scale)")
        ax2.set_title("Sensitive dependence on initial conditions")
        ax2.legend(loc="lower right", fontsize=8)
        st.pyplot(fig2)
        st.caption("A straight line on this log axis means the gap grows exponentially - "
                   "the signature of chaos. Try this on the Pythagorean preset for the full "
                   "effect; on the stable Figure-8 the line stays almost flat, because that "
                   "orbit resists perturbation. That contrast is itself a real result.")

    # 6) Events
    with tabs[5]:
        ev = detect_events(pos_rk4, vel_rk4, masses, dt, eps)

        def _yn(b):
            return "detected" if b else "-"

        def _pair(p):
            return f"bodies {p[0] + 1} & {p[1] + 1}" if p else "-"

        rows = [
            ["Collision / close encounter", _yn(ev["collision"]),
             (f"{_pair(ev['collision_pair'])} at t={ev['collision_time']:.1f}, "
              f"min gap {ev['min_pair_distance']:.3f}") if ev["collision"]
             else f"closest approach was {ev['min_pair_distance']:.3f}"],
            ["Escape / ejection", _yn(ev["escape"]),
             (f"body {ev['escape_body'] + 1} at t={ev['escape_time']:.1f}, reached "
              f"{ev['max_com_distance']:.1f} from centre") if ev["escape"]
             else f"max distance from centre was {ev['max_com_distance']:.1f}"],
            ["Gravitational slingshot", _yn(ev["slingshot"]),
             f"body {ev['slingshot_body'] + 1} sped up sharply after the encounter"
             if ev["slingshot"] else "no strong post-encounter speed-up"],
            ["Capture (binary forms)", _yn(ev["capture"]),
             f"bodies {ev['capture_pair'][0] + 1} & {ev['capture_pair'][1] + 1} left bound"
             if ev["capture"] else "no bound pair left at the end"],
        ]
        st.markdown("**Event detection**")
        st.table(pd.DataFrame(rows, columns=["Event", "Status", "Details"]).set_index("Event"))
        st.caption("Heuristic detectors with tunable thresholds (collision radius, escape "
                   "radius, slingshot gain). Try the Pythagorean preset: it ejects the "
                   "lightest body while the other two capture into a binary.")

    # 7) Field
    with tabs[6]:
        p0 = np.asarray(positions, float)
        cx, cy = p0[:, 0].mean(), p0[:, 1].mean()
        span = max(np.ptp(p0[:, 0]), np.ptp(p0[:, 1]), 1.0) * 0.5 + 2.0
        X, Y, Phi = potential_grid(masses, positions,
                                   (cx - span, cx + span), (cy - span, cy + span),
                                   240, max(eps, 0.05))
        Phi_c = np.clip(Phi, np.percentile(Phi, 3), np.percentile(Phi, 97))
        fig3, ax3 = plt.subplots(figsize=(6.5, 6))
        cf = ax3.contourf(X, Y, Phi_c, levels=30, cmap="magma")
        ax3.contour(X, Y, Phi_c, levels=14, colors="white", linewidths=0.3, alpha=0.35)
        for i in range(3):
            ax3.plot(pos_rk4[:, i, 0], pos_rk4[:, i, 1], color="white", lw=0.5, alpha=0.5)
        ax3.scatter(p0[:, 0], p0[:, 1], c="#7DF9FF",
                    s=[25 + 25 * mm for mm in masses], edgecolor="k", zorder=5)
        ax3.set_xlim(cx - span, cx + span)
        ax3.set_ylim(cy - span, cy + span)
        ax3.set_aspect("equal")
        ax3.set_title("Gravitational potential  \u03a6(x, y)")
        fig3.colorbar(cf, ax=ax3, label="potential (deeper = stronger pull)")
        st.pyplot(fig3)
        st.caption("Filled contours of the gravitational potential for the starting "
                   "configuration - the deep wells are the bodies. The faint white lines are "
                   "the actual trajectories, which flow through the shape of this landscape.")
else:
    st.info("Set your values on the left and above, then click **Run simulation**.")


# --- Monte Carlo stability map (independent of the single run above) ----------
st.divider()
st.header("Monte Carlo stability map")
st.markdown(
    "Scan a whole family of three-body systems at once. A heavy central body sits at "
    "the origin with a distant perturber; a test body is launched from a range of "
    "**positions** (Y axis) and **speeds** (X axis). Every point is one simulation "
    "(plus a perturbed twin to test for chaos), classified by its fate.")

cA, cB = st.columns(2)
res = cA.slider("Grid resolution (N x N simulations)", 20, 36, 30)
mc_steps = cB.slider("Steps per simulation", 1500, 4000, 2500, step=500)


@st.cache_data(show_spinner=False)
def _stability(N, steps):
    return run_stability_map(DEFAULT_SCENARIO, N, steps)


if st.button("Generate stability map", type="primary"):
    with st.spinner(f"Running {res * res} simulations (plus perturbed twins)..."):
        map_df, grid, ya, xa = _stability(res, int(mc_steps))

    CLASSES = ["periodic", "chaotic", "collision", "escape"]
    CLASS_COLORS = {"periodic": "#2E8B57", "chaotic": "#E8A317",
                    "collision": "#C0392B", "escape": "#2E6FB0"}
    code = {c: i for i, c in enumerate(CLASSES)}
    Z = np.vectorize(lambda c: code[c])(grid)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASSES])

    left, right = st.columns([3, 2])
    with left:
        figm, axm = plt.subplots(figsize=(6.5, 6))
        axm.imshow(Z, origin="lower", aspect="auto",
                   extent=[xa[0], xa[-1], ya[0], ya[-1]],
                   cmap=cmap, vmin=0, vmax=len(CLASSES) - 1)
        axm.set_xlabel("initial velocity  (vy of the test body)")
        axm.set_ylabel("initial position  (x of the test body)")
        axm.set_title("Stability map")
        axm.legend(handles=[Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES],
                   loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
        figm.tight_layout()
        st.pyplot(figm)
    with right:
        st.markdown("**Outcome counts**")
        counts = (map_df["class"].value_counts()
                  .rename_axis("class").reset_index(name="count").set_index("class"))
        st.table(counts)
        st.metric("Stable (bounded) fraction", f"{100 * map_df['stable'].mean():.1f}%")

    st.markdown("**Generated dataset** (one row per simulation)")
    st.dataframe(map_df, height=260)
    st.download_button("Download dataset (CSV)", map_df.to_csv(index=False),
                       "stability_dataset.csv", "text/csv")
    st.caption("A real labelled dataset: initial conditions in, outcome and diagnostics out. "
               "Exactly the kind of thing you could train a classifier on to predict a "
               "system's fate from its starting state - physics generating machine-learning data.")
else:
    st.info("Choose a resolution and click **Generate stability map**.")


# --- Fate predictor (machine learning surrogate) -----------------------------
st.divider()
st.header("Predict a system's fate (machine learning)")
st.markdown(
    "Train a model to predict the outcome (periodic / chaotic / collision / escape) "
    "from **only what's known at t=0** - initial position, velocity, energy, angular "
    "momentum, and closest starting approach. Outcome quantities are deliberately "
    "excluded (using them would be leakage). The result is a **surrogate model**: it "
    "predicts a system's fate in microseconds instead of running a full simulation.")

n_samples = st.slider("Training simulations to generate", 500, 3000, 1500, step=250)

FATE_COLORS = {"periodic": "#2E8B57", "chaotic": "#E8A317",
               "collision": "#C0392B", "escape": "#2E6FB0"}


@st.cache_data(show_spinner=False)
def _train(n, steps):
    df = build_dataset(n, steps, seed=0)
    clf, mx = train_fate_model(df, seed=0)
    return df, clf, mx


if st.button("Generate data & train model", type="primary"):
    st.session_state["fate_trained"] = True

if st.session_state.get("fate_trained"):
    with st.spinner(f"Simulating {n_samples} systems and training the model..."):
        ml_df, clf, mx = _train(n_samples, 2800)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{100 * mx['accuracy']:.1f}%")
    b.metric("Training set", f"{mx['n_train']}")
    c.metric("Test set", f"{mx['n_test']}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Per-class performance** (test set)")
        rep = mx["report"]
        rows = [[cl, f"{rep[cl]['precision']:.2f}", f"{rep[cl]['recall']:.2f}",
                 f"{rep[cl]['f1-score']:.2f}", int(rep[cl]['support'])]
                for cl in mx["labels"]]
        st.table(pd.DataFrame(rows, columns=["class", "precision", "recall",
                                             "f1", "support"]).set_index("class"))
        st.markdown("**Feature importance** (what the model relies on)")
        st.bar_chart(pd.Series(mx["importances"]).sort_values())
    with right:
        st.markdown("**Confusion matrix** (rows = true, cols = predicted)")
        cm = mx["confusion"]; labs = mx["labels"]
        figc, axc = plt.subplots(figsize=(4.6, 4.2))
        axc.imshow(cm, cmap="Blues")
        axc.set_xticks(range(len(labs))); axc.set_xticklabels(labs, rotation=45, ha="right")
        axc.set_yticks(range(len(labs))); axc.set_yticklabels(labs)
        for i in range(len(labs)):
            for j in range(len(labs)):
                axc.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        axc.set_xlabel("predicted"); axc.set_ylabel("true")
        figc.tight_layout()
        st.pyplot(figc)

    st.caption("Initial energy is typically the strongest predictor - the model "
               "rediscovers that a system's energy largely decides whether it stays bound "
               "or flies apart. Chaotic is the hard class: it's rare and lives on thin, "
               "fractal-like boundaries, so its recall is honestly the weakest.")

    # learned fate map vs the true stability map
    st.markdown("**The model's learned fate map** (predicted across the whole plane)")
    gridpred, pa, va = predict_grid(clf, 160)
    code = {cl: i for i, cl in enumerate(FATE_CLASSES)}
    Zc = np.vectorize(lambda cl: code[cl])(gridpred)
    cmap = ListedColormap([FATE_COLORS[cl] for cl in FATE_CLASSES])
    figp, axp = plt.subplots(figsize=(6.5, 5.5))
    axp.imshow(Zc, origin="lower", aspect="auto",
               extent=[va[0], va[-1], pa[0], pa[-1]],
               cmap=cmap, vmin=0, vmax=len(FATE_CLASSES) - 1)
    axp.set_xlabel("initial velocity (vy)"); axp.set_ylabel("initial position (x)")
    axp.set_title("Predicted fate (surrogate model)")
    axp.legend(handles=[Patch(color=FATE_COLORS[cl], label=cl) for cl in FATE_CLASSES],
               loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
    figp.tight_layout()
    st.pyplot(figp)
    st.caption("Compare this to the true stability map above - the model reconstructs the "
               "same regions from initial conditions alone, no simulation required.")

    # live surrogate
    st.markdown("**Try the surrogate** - pick a launch and get an instant prediction:")
    q1, q2 = st.columns(2)
    qpos = q1.slider("initial position (x)", float(DEFAULT_SCENARIO["pos_range"][0]),
                     float(DEFAULT_SCENARIO["pos_range"][1]), 1.5, step=0.01)
    qvel = q2.slider("initial velocity (vy)", float(DEFAULT_SCENARIO["vel_range"][0]),
                     float(DEFAULT_SCENARIO["vel_range"][1]), 1.0, step=0.01)
    lab, proba = predict_one(clf, qpos, qvel)
    st.metric("Predicted fate", lab)
    st.bar_chart(pd.Series(proba).sort_values(ascending=False))
    st.caption("This prediction is instant - no simulation runs. That's the point of a "
               "surrogate: once trained on simulated data, it replaces the simulation.")

    st.download_button("Download training dataset (CSV)", ml_df.to_csv(index=False),
                       "fate_dataset.csv", "text/csv")
else:
    st.info("Click **Generate data & train model** to build the dataset and train the surrogate.")