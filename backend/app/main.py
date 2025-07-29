"""
FastAPI application for ingesting sensor events and classifying them using an
XGBoost model. Events are stored in a MongoDB collection and can be queried
through the provided API endpoints.

Environment variables used:

* ``MONGODB_URI`` – MongoDB connection string. This should include the
  username and password for the database user created in Atlas. Example::

      mongodb+srv://vape_user:<PASSWORD>@vape-alert.xnthaph3.mongodb.net/?retryWrites=true&w=majority

* ``DATABASE_NAME`` – Name of the database to use within the MongoDB cluster.
* ``MODEL_PATH`` – Path to the XGBoost model file. Defaults to ``model.json``.
* ``CORS_ORIGINS`` – Comma separated list of origins allowed to call the API.

To run locally, create a ``.env`` file with the above values.
"""

import os
from typing import List, Optional, Any

from datetime import datetime

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load environment variables from a local .env file if present. This is a no-op
# if the file does not exist.
load_dotenv()


# Read configuration from the environment. Fail fast if a required var is missing.
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI environment variable must be set.")
if not DATABASE_NAME:
    raise RuntimeError("DATABASE_NAME environment variable must be set.")


# Instantiate MongoDB client and get a handle to the events collection. The
# connection is created once and shared across the application.
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]
events_collection = db["events"]


# Load the XGBoost model. We expect a multi-class model that outputs
# probabilities for three classes: fire, vape and normal. If the model cannot
# be loaded, raise at startup to make the failure obvious.
model_path = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.json"))

# If the model file does not exist on disk, reconstruct it from the
# base64 encoded representation bundled with the code. This avoids
# committing large binary files to the repository. When MODEL_PATH
# exists (e.g. provided via a mounted volume), it will be used instead.
if not os.path.exists(model_path):
    from . import model_data  # imported lazily to avoid circular import
    import base64
    decoded = base64.b64decode(model_data.MODEL_B64)
    with open(model_path, "wb") as f:
        f.write(decoded)

try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(f"Failed to load XGBoost model from '{model_path}': {exc}") from exc


class EventIn(BaseModel):
    """Schema for incoming sensor events."""

    device_id: str = Field(..., description="Unique identifier of the device emitting the event")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of when the event was recorded")
    location: str = Field(..., description="Location of the device (e.g. classroom identifier)")
    humidity: float = Field(..., description="Humidity measurement from the sensor")
    pm25: float = Field(..., description="PM2.5 (particulate matter) reading from the sensor")
    particle_size: float = Field(..., description="Average particle size measurement")
    volume_spike: float = Field(..., description="Volume spike measurement from the acoustic sensor")


class EventOut(EventIn):
    """Schema for returned event documents, including prediction fields."""

    confidence: float = Field(..., description="Confidence score of the prediction (0–1)")
    predicted_type: str = Field(..., description="Predicted class: fire, vape or normal")
    _id: Optional[Any] = Field(None, alias="_id", description="MongoDB document identifier")


app = FastAPI(title="Vape Detection Backend")

# Configure CORS. Defaults to allowing all origins if CORS_ORIGINS is not set.
origins_env = os.getenv("CORS_ORIGINS", "*")
origins: List[str] = [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/events", response_model=EventOut)
async def create_event(event: EventIn) -> EventOut:
    """
    Receive a sensor event, classify it using the XGBoost model and insert
    the augmented event into the database.
    """
    # Prepare input for the model. Ensure the feature order matches the
    # training dataset used when fitting the model.
    features = np.array(
        [
            [
                event.humidity,
                event.pm25,
                event.particle_size,
                event.volume_spike,
            ]
        ]
    )

    # Predict probabilities; fallback gracefully if not available.
    try:
        probs = xgb_model.predict_proba(features)[0]
    except AttributeError:
        # If the model does not support predict_proba, use predict instead and
        # fabricate probabilities.
        pred_class = int(xgb_model.predict(features)[0])
        probs = np.zeros(3)
        probs[pred_class] = 1.0

    predicted_index = int(np.argmax(probs))
    classes = ["fire", "vape", "normal"]
    predicted_type = classes[predicted_index] if predicted_index < len(classes) else "normal"
    confidence = float(np.max(probs))

    # Build the document to store. Use event.dict() with by_alias to preserve
    # field names and convert datetime to ISO format for MongoDB.
    doc = event.model_dump(by_alias=True)
    doc.update({
        "confidence": confidence,
        "predicted_type": predicted_type,
    })

    result = await events_collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return EventOut(**doc)


@app.get("/api/events", response_model=List[EventOut])
async def get_events(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of events to return"),
    since: Optional[str] = Query(
        None,
        description="ISO 8601 timestamp; return events with timestamp greater than or equal to this",
    ),
) -> List[EventOut]:
    """
    Retrieve recent events from the database. Results are sorted by timestamp
    descending. Optionally filter by a minimum timestamp.
    """
    query = {}
    if since:
        try:
            # Validate ISO date and store in query; Mongo will use the string
            datetime.fromisoformat(since.replace("Z", "+00:00"))
            query["timestamp"] = {"$gte": since}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' timestamp format")

    cursor = (
        events_collection.find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )

    events: List[EventOut] = []
    async for doc in cursor:
        doc_out = EventOut(**doc, _id=doc.get("_id"))
        events.append(doc_out)
    return events
