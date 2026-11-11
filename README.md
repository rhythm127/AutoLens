# AutoLens

AutoLens is a student capstone project exploring vehicle warning-light troubleshooting. The planned system will use a vehicle owner's manual, warning-light information, and safety rules to provide grounded guidance. The backend is currently at an early foundation stage; the full diagnostic system is not implemented yet.

## What works today

The Python backend is a small FastAPI application with two endpoints:

- `GET /health` returns a simple status response to show the API is responding.
- `POST /vehicles` checks a manufacturer's name, model, and year, then returns those details. It does not save them to a database yet.

Vehicle names must contain 1–50 characters after surrounding spaces are removed. The year must be between 1980 and 2027, inclusive.

The project does not yet include a frontend, PDF/manual extraction, search, image analysis, an AI model connection, diagnostic chat, or persistent vehicle records. PyMuPDF is listed as a runtime dependency for the planned PDF work, but the PDF extraction feature has not been built yet.

## Requirements

- Windows with Python 3.14 available
- The packages listed in `backend/requirements.txt`

## Set up the backend

Open PowerShell in the project folder, then run:

```powershell
cd backend
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A virtual environment is a separate Python workspace for this project. It keeps AutoLens packages separate from other Python projects on the computer.

## Start the API

From the `backend` folder, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Keep that PowerShell window open while using the API. Uvicorn is the small server that runs the FastAPI application. To stop it, press `Ctrl+C` in that window.

Open these addresses in a browser while the server is running:

- `http://127.0.0.1:8000/health` — health response
- `http://127.0.0.1:8000/docs` — interactive API documentation, where you can try the endpoints

## Run the backend tests

From the `backend` folder, install the development/test packages and run the tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -v
```

The tests check the health response and basic valid/invalid vehicle submissions. Ask the project owner to run these commands and review the output when code or dependencies change.

## Project status

The local project folder is the source of truth for the code. The synopsis describes the intended project; it does not mean those features are already implemented. GitHub publication is optional and will only happen if the owner chooses it.
