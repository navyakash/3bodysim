# Three-Body Gravity Simulator

An interactive, physically honest simulation of the gravitational **three-body problem** — the classic chaotic system that, unlike the two-body case, has *no* general closed-form solution and must be solved numerically. Built from first principles: Newtonian gravity, numerical integration (Euler and RK4), energy-conservation diagnostics, and a real-time animated web interface.

> _Add a short demo GIF here — the figure-8 preset in the Animation tab makes the best one._
> `![demo](docs/demo.gif)`

---

## What it does

- Simulates three gravitating bodies (the engine itself is n-body general) under Newtonian gravity.
- Two numerical integrators — **Euler** and **RK4** — that can be run on the same system and compared **side by side**.
- Four views of every run:
  - **Animation** — the motion in real time, with glowing bodies and fading comet trails.
  - **Trajectory** — the full path of each body over the whole run.
  - **Energy** — total energy vs time, the correctness check for the integrator.
  - **Euler vs RK4** — the same initial conditions integrated both ways, animated together, with both energy curves on one axis.
- Preset scenarios, including the famous **figure-8** stable orbit (Chenciner & Montgomery, 2000).
- Full control over each body's mass, position, and velocity, plus timestep and gravitational softening.

---

## The physics, briefly

The force between two masses is Newton's law of universal gravitation. In vector form, the force on body *i* from body *j* points along the line joining them and falls off with the square of the distance. The total force on a body is the **superposition** (sum) of the pulls from all the others.

Newton's second law makes this a system of **second-order** differential equations. Since numerical integrators solve **first-order** systems, the code promotes velocity to its own variable, turning the problem into:

```
d(position)/dt = velocity
d(velocity)/dt = acceleration(position)   # the gravity
```

That first-order system is marched forward in time by the integrator. **Total energy** (kinetic + gravitational potential) is computed at every step: real physics keeps it constant, so any drift is a direct measure of numerical error.

---

## Results

- On the same system over 20,000 steps, **Euler leaks ~22% of the system's total energy while RK4 stays flat to five decimal places** — a concrete, measured demonstration of why higher-order integrators matter.
- The figure-8 orbit **survives indefinitely under RK4 but disintegrates within a few loops under Euler**, from identical starting conditions.

---

## Tech stack

Python · NumPy (physics engine) · Streamlit (web UI) · HTML5 Canvas (60fps animation) · Matplotlib (static plots)

---

## Project structure

```
three_body.py     UI-agnostic physics engine: gravity, integrators, energy, simulate()
app.py            Streamlit interface: inputs, animation, plots, comparison
requirements.txt  dependencies
```

The engine and the interface are deliberately separated: `three_body.py` knows nothing about Streamlit or the web. `app.py` only imports `simulate()` and decides how to display the numbers. This makes the physics testable on its own and the UI easy to change.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens automatically. Pick a preset (or enter your own values), click **Run simulation**, and explore the four tabs.

---

## Deploy

The app deploys free on **Streamlit Community Cloud**: push this repo to GitHub, connect the repo at [share.streamlit.io](https://share.streamlit.io), and point it at `app.py`. Dependencies install automatically from `requirements.txt`.

---

## Roadmap

- [ ] **Chaos / sensitivity to initial conditions** — two runs differing by one part in a million, animated until they diverge.
- [ ] **Lyapunov exponent** estimation to quantify that divergence.
- [ ] Momentum and angular-momentum conservation checks.
- [ ] **Symplectic integrators** (leapfrog / velocity-Verlet) for long-term energy stability.
- [ ] Inverse problems: infer masses or initial conditions from an observed trajectory.

---

## Author

**Navya Kashyap** — B.Tech, AI & Data Science
[GitHub](https://github.com/navyakash) 

## License

Released under the MIT License. See [LICENSE](LICENSE).