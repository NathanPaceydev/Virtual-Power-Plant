# 🌱 Virtual Power Plant Toolkit

A **Flask-based web app** for clean energy feasibility studies and project planning.

This toolkit helps users **design, price, and evaluate** clean energy generation projects—like solar, wind, and battery storage—based on location-specific data. It's built for **ease of use**, enabling anyone to quickly assess production potential and inform decisions on pricing and procurement.

---

## 🚀 Features

- 🌞 Solar, 💨 Wind, and 🔋 Battery modeling
- 📍 Location-specific forecasts (via APIs)
- 📊 Financial breakdowns & cost analysis
- 🧮 Intuitive interface for non-technical users
- 🔧 Developer-friendly codebase (Flask + Plotly)

---

## 🧰 Getting Started

Want to see it in action?  
**Watch this video tutorial** to learn how to download and start using the Toolkit:

[![Watch the video](https://img.youtube.com/vi/x4Zi4jsRHSM/0.jpg)](https://www.youtube.com/watch?v=x4Zi4jsRHSM)

## Live Demo Hosting

This repo is configured for a free Flask web service on Render so anyone can open a public demo URL without cloning the project.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/NathanPaceydev/Virtual-Power-Plant)

### Render deployment https://virtual-power-plant.onrender.com/

The included `render.yaml` creates a free Python web service using only the `flask app` folder:

- Root directory: `flask app`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 180 --max-requests 20 --max-requests-jitter 5`
- Health check: `/healthz`
- Python version: `3.11.11`

Manual setup is also straightforward:

1. Push this repository to GitHub.
2. In Render, create a new Web Service or Blueprint from the repository.
3. Choose the free instance type.
4. Set the root directory to `flask app` if you are not using the Blueprint.
5. Use the build and start commands above.
6. After the deploy finishes, share the generated `https://<service-name>.onrender.com` URL.

Notes checked on May 5, 2026: Render supports free Python web services, but free instances spin down after 15 minutes without traffic and may take about a minute to wake up. This is fine for demos, not production. PythonAnywhere free accounts currently have a 512 MiB disk limit and a 1-month-expiring web app, so the full repository plus dependencies is likely too large unless you deploy only the Flask folder.

The app is tuned for Render's small free instance: Plotly is loaded once from the browser CDN instead of being embedded in every chart, unused wind API variables are not requested, heavy unused scientific packages are excluded, and Gunicorn recycles workers periodically to avoid memory creep during demos.

### PythonAnywhere fallback

If you still want to try PythonAnywhere, use a manual Flask web app and upload only the `flask app` folder to save disk space. Install dependencies with:

```bash
pip install --user -r requirements.txt
```

Then configure the PythonAnywhere WSGI file like this, replacing `<username>` with your account name:

```python
import sys

project_home = "/home/<username>/flask app"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application
```

Add a static files mapping in PythonAnywhere from `/static/` to `/home/<username>/flask app/static`.

---

## 👩‍💻 Developer Guide

Planning to contribute or customize the app?  
**Watch this developer guide** for instructions on running the app locally and adding features:

[![Watch the video](https://img.youtube.com/vi/foTwMsRsJWI/0.jpg)](https://www.youtube.com/watch?v=foTwMsRsJWI&t=5s)

---

## 📦 Tech Stack

- **Backend**: Python, Flask
- **Frontend**: Plotly Dash, HTML/CSS
- **Data**: PVWatts (NREL), Open-Meteo APIs, Equipment list on Homepage (prices as of 2024)
- **Hosting**: Local server / Docker (optional)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.  
Let’s build better tools for the energy transition 🌍

---

## 📜 License

[MIT License](LICENSE)
