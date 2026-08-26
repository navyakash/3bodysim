# Three-Body Gravity Simulator

An interactive, physically honest simulation of the gravitational **three-body problem** — the classic chaotic system that, unlike the two-body case, has *no* general closed-form solution and must be solved numerically. Built from first principles: Newtonian gravity, numerical integration (Euler and RK4), energy-conservation diagnostics, and a real-time animated web interface — plus a demonstration of deterministic **chaos**.

> _Add a short demo GIF here — the figure-8 preset or the Chaos tab makes the best one._
> `![demo](docs/demo.gif)`

---

## What it does

- Simulates three gravitating bodies (the engine itself is n-body general) under Newtonian gravity.
- Two numerical integrators — **Euler** and **RK4** — that can be run on the same system and compared **side by side**.
- Five views of every run:
  - **Animation** — the motion in real time, with glowing bodies and fading comet trails.
  - **Trajectory** — the full path of each body over the whole run.
  - **Energy** — total energy vs time, the correctness check for the integrator.
  - **Euler vs RK4** — the same initial conditions integrated both ways, animated together, with both energy curves on one axis.
  - **Chaos** — two RK4 runs whose starting positions differ by one part in a million, animated together until they diverge, with a log-scale separation plot and a Lyapunov-exponent estimate.
- Preset scenarios: two bodies orbiting a heavy one, the famous **figure-8** stable orbit (Chenciner & Montgomery, 2000), and the chaotic **Pythagorean** (Burrau) problem.
- Full control over each body's mass, position, and velocity, plus timestep, gravitational softening, and the chaos perturbation size.

---

## The physics, briefly

The force between two masses is Newton's law of universal gravitation. In vector form, the force on body *i* from body *j* points along the line joining them and falls off with the square of the distance. The total force on a body is the **superposition** (sum) of the pulls from all the others.

Newton's second law makes this a system of **second-order** differential equations. Since numerical integrators solve **first-order** systems, the code promotes velocity to its own variable:

```
d(position)/dt = velocity
d(velocity)/dt = acceleration(position)   # the gravity
```

That first-order system is marched forward by the integrator. **Total energy** (kinetic + gravitational potential) is computed each step as a correctness check: real physics keeps it constant, so any drift measures numerical error. **Chaos** is quantified by running two nearly identical initial conditions and tracking how their separation grows — exponential growth (a straight line on a log axis) is the signature, and its slope estimates the largest **Lyapunov exponent**.

---

## Results

- On the same system over 20,000 steps, **Euler leaks ~22% of the system's total energy while RK4 stays flat to five decimal places** — a concrete, measured demonstration of why higher-order integrators matter.
- The figure-8 orbit **survives indefinitely under RK4 but disintegrates within a few loops under Euler**, from identical starting conditions.
- On the chaotic Pythagorean problem, a **one-part-in-a-million** difference in one starting coordinate grows to a macroscopic separation of ~45 by t≈45, with an estimated Lyapunov exponent of ~0.6 per time unit and a predictability horizon of t≈18.

---

## Tech stack

Python · NumPy (vectorized physics engine) · Streamlit (web UI) · HTML5 Canvas (60fps animation) · Matplotlib (static plots)

---

## Project structure

```
three_body.py     UI-agnostic physics engine: gravity, integrators, energy, simulate(),
                  and the chaos analysis (separation, Lyapunov estimate)
app.py            Streamlit interface: inputs, animation, plots, comparisons
requirements.txt  dependencies
```

The engine and the interface are deliberately separated: `three_body.py` knows nothing about Streamlit or the web. `app.py` only imports `simulate()` (and the analysis helpers) and decides how to display the numbers. This keeps the physics testable on its own and the UI easy to change.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens automatically. Pick a preset (or enter your own values), click **Run simulation**, and explore the five tabs. For the Chaos tab, the **Pythagorean** preset gives the most dramatic divergence.

---

## Deploy

The app deploys free on **Streamlit Community Cloud**: push this repo to GitHub, connect the repo at [share.streamlit.io](https://share.streamlit.io), and point it at `app.py`. Dependencies install automatically from `requirements.txt`.

---

## Roadmap

- [x] **Chaos / sensitivity to initial conditions** — twin runs diverging, with a Lyapunov estimate.
- [ ] Momentum and angular-momentum conservation checks.
- [ ] **Symplectic integrators** (leapfrog / velocity-Verlet) for long-term energy stability.
- [ ] Inverse problems: infer masses or initial conditions from an observed trajectory.
- [ ] Variable body count (the engine is already n-body general).

---

## Author

**[Navya Kashyap]** — B.Tech, AI & Data Science
[GitHub](https://github.com/navyakash) 

## License

Released under the MIT License. See [LICENSE](LICENSE).