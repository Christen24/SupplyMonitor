"""FastAPI App wrapper exposing the SCRM Environment via HTTP and WebSocket."""
from openenv.core.env_server import create_fastapi_app
from server.environment import SupplyChainEnvironment
from models import SCRMAction, SCRMObservation

import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = create_fastapi_app(SupplyChainEnvironment, SCRMAction, SCRMObservation)

# Create frontend directory
if not os.path.exists("frontend"):
    os.makedirs("frontend")

# Mount static folder for assets
app.mount("/assets", StaticFiles(directory="frontend"), name="assets")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the dashboard UI at the root of the Hugging Face Space."""
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend UI not found.</h1><p>Please place index.html in the 'frontend/' directory.</p>")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
