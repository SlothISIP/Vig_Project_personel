"""
FastAPI Application - Digital Twin Factory System
Integrated API Gateway for all services
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import numpy as np
import cv2
import json
from datetime import datetime

from src.vision.inference.onnx_infer import ONNXInferenceEngine
from src.digital_twin.state.machine_state import MachineStateManager
from src.digital_twin.simulation.simulator import FactorySimulator, SimulationConfig
from src.predictive.predictor import PredictiveMaintenanceSystem
from src.scheduling.scheduler import ProductionScheduler
from src.scheduling.models import Job, Machine
from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger
from src.core.constants import ONNX_DIR
from src.core.exceptions import (
    ModelInferenceError,
    DataValidationError,
    DigitalTwinStateError,
    SchedulingError,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Digital Twin Factory - Integrated API",
    description="Complete AI-powered digital twin system",
    version="1.0.0",
)

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Exception Handlers (Enhanced Error Handling)
# ============================================================================

@app.exception_handler(DataValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors with 400 Bad Request."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(DigitalTwinStateError)
async def state_exception_handler(request, exc):
    """Handle state errors with 409 Conflict."""
    logger.error(f"State error: {exc}")
    return JSONResponse(
        status_code=409,
        content={
            "error": "State Error",
            "detail": str(exc),
            "type": "state_error"
        }
    )

@app.exception_handler(SchedulingError)
async def scheduling_exception_handler(request, exc):
    """Handle scheduling errors with 422 Unprocessable Entity."""
    logger.error(f"Scheduling error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Scheduling Error",
            "detail": str(exc),
            "type": "scheduling_error"
        }
    )

@app.exception_handler(ModelInferenceError)
async def model_exception_handler(request, exc):
    """Handle model inference errors with 503 Service Unavailable."""
    logger.error(f"Model inference error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Model Inference Error",
            "detail": str(exc),
            "type": "model_error"
        }
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle file not found errors with 404."""
    logger.error(f"File not found: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource Not Found",
            "detail": str(exc),
            "type": "not_found"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors with 400 Bad Request."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid Input",
            "detail": str(exc),
            "type": "value_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions with 500 Internal Server Error."""
    logger.exception(f"Unexpected error: {exc}")  # Logs full stack trace
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "type": "internal_error"
        }
    )

# Global services
inference_engine: Optional[ONNXInferenceEngine] = None
machine_state_manager: Optional[MachineStateManager] = None
factory_simulator: Optional[FactorySimulator] = None
predictive_system: Optional[PredictiveMaintenanceSystem] = None
production_scheduler: Optional[ProductionScheduler] = None
websocket_clients: List[WebSocket] = []

# Shutdown event for graceful termination
shutdown_event = asyncio.Event()

# Dashboard stats cache (TTL-based)
dashboard_cache: Dict[str, Any] = {}
DASHBOARD_CACHE_TTL = 5  # 5 seconds TTL


# ============================================================================
# Startup / Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    global inference_engine, machine_state_manager, factory_simulator
    global predictive_system, production_scheduler

    logger.info("🚀 Starting Digital Twin Factory System...")

    # 1. Vision AI (optional - only if model exists)
    model_path = ONNX_DIR / "swin_defect.onnx"
    if model_path.exists():
        try:
            inference_engine = ONNXInferenceEngine(
                model_path=model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                num_classes=2,
                image_size=224,
            )
            logger.info("✓ Vision AI loaded")
        except Exception as e:
            logger.warning(f"Vision AI not loaded: {e}")
    else:
        logger.warning("Vision AI model not found - skipping")

    # 2. Machine State Manager (core)
    try:
        machine_state_manager = MachineStateManager()
        # Initialize with 3 default machines
        for i in range(1, 4):
            machine_state_manager.add_machine(
                machine_id=f"M{i:03d}",
                machine_type=f"Type_{i}",
                initial_state="idle"
            )
        logger.info("✓ Machine State Manager initialized (3 machines)")
    except Exception as e:
        logger.error(f"Failed to initialize state manager: {e}")

    # 3. Factory Simulator
    try:
        config = SimulationConfig(
            num_machines=3,
            num_production_lines=1,
            simulation_duration=1000.0,
            product_arrival_rate=10.0,
        )
        factory_simulator = FactorySimulator(config)
        logger.info("✓ Factory Simulator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")

    # 4. Predictive Maintenance System
    try:
        predictive_system = PredictiveMaintenanceSystem()
        logger.info("✓ Predictive Maintenance System initialized")
    except Exception as e:
        logger.error(f"Failed to initialize predictive system: {e}")

    # 5. Production Scheduler
    try:
        scheduler_machines = [
            Machine(
                machine_id=f"M{i:03d}",
                machine_type=f"Type_{i}",
                capabilities=[f"op_{i}"],
                available=True
            )
            for i in range(1, 4)
        ]
        production_scheduler = ProductionScheduler(machines=scheduler_machines)
        logger.info("✓ Production Scheduler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {e}")

    logger.info("✅ All services initialized successfully")

    # 6. Start background tasks
    asyncio.create_task(simulate_factory_updates())
    logger.info("✓ Background tasks started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down Digital Twin Factory System...")

    # 1. Signal background tasks to stop
    shutdown_event.set()
    logger.info("✓ Shutdown signal sent to background tasks")

    # 2. Wait a bit for tasks to finish
    await asyncio.sleep(1)

    # 3. Close all WebSocket connections
    for ws in websocket_clients:
        try:
            await ws.close()
        except:
            pass
    websocket_clients.clear()
    logger.info("✓ WebSocket connections closed")

    logger.info("✅ Shutdown complete")


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Digital Twin Factory - Integrated System",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "vision_ai": inference_engine is not None,
            "digital_twin": machine_state_manager is not None,
            "simulator": factory_simulator is not None,
            "predictive": predictive_system is not None,
            "scheduler": production_scheduler is not None,
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "vision_ai": inference_engine is not None,
            "digital_twin": machine_state_manager is not None,
            "simulator": factory_simulator is not None,
            "predictive": predictive_system is not None,
            "scheduler": production_scheduler is not None,
        }
    }


# ============================================================================
# Vision AI Endpoints (existing)
# ============================================================================

def get_inference_engine() -> ONNXInferenceEngine:
    """Dependency to get inference engine."""
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Vision AI not loaded. Model file missing.",
        )
    return inference_engine


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image file."""
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}",
        )


@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
):
    """Predict defects in uploaded image."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}",
        )

    try:
        contents = await file.read()
        image = decode_image(contents)
        result = engine.predict(image, return_probabilities=True)
        result["image_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "shape": image.shape,
        }
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Digital Twin Endpoints (NEW)
# ============================================================================

@app.get("/api/v1/digital-twin/factory/{factory_id}")
async def get_factory_state(factory_id: str):
    """Get complete factory state."""
    if not machine_state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        machines_data = []
        all_machines = machine_state_manager.get_all_machines()

        for machine_id, state in all_machines.items():
            machines_data.append({
                "machine_id": machine_id,
                "machine_name": f"Machine {machine_id}",
                "machine_type": state.machine_type,
                "status": state.state,
                "health_score": state.health_score,
                "temperature": state.temperature,
                "vibration": state.vibration,
                "pressure": state.pressure,
                "speed": state.speed,
                "defect_rate": state.defect_rate,
                "cycle_count": state.cycle_count,
                "defect_count": state.defect_count,
                "last_maintenance": state.last_maintenance.isoformat() if state.last_maintenance else None,
            })

        # Aggregate statistics
        total_machines = len(machines_data)
        running_machines = sum(1 for m in machines_data if m["status"] == "running")

        return {
            "factory_id": factory_id,
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "machines": machines_data,
            "total_machines": total_machines,
            "running_machines": running_machines,
            "total_products_produced": sum(m["cycle_count"] for m in machines_data),
            "total_defects_detected": sum(m["defect_count"] for m in machines_data),
            "overall_oee": sum(m["health_score"] for m in machines_data) / total_machines if total_machines > 0 else 0,
            "uptime_percentage": running_machines / total_machines if total_machines > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error getting factory state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/digital-twin/machine/{machine_id}")
async def get_machine_state(machine_id: str):
    """Get specific machine state."""
    if not machine_state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        state = machine_state_manager.get_machine_state(machine_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

        return {
            "machine_id": machine_id,
            "machine_type": state.machine_type,
            "status": state.state,
            "health_score": state.health_score,
            "temperature": state.temperature,
            "vibration": state.vibration,
            "pressure": state.pressure,
            "speed": state.speed,
            "defect_rate": state.defect_rate,
            "cycle_count": state.cycle_count,
            "defect_count": state.defect_count,
            "last_maintenance": state.last_maintenance.isoformat() if state.last_maintenance else None,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting machine state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Predictive Maintenance Endpoints (NEW)
# ============================================================================

@app.get("/api/v1/predictive/maintenance/{machine_id}")
async def get_maintenance_recommendation(machine_id: str):
    """Get maintenance recommendation for specific machine."""
    if not machine_state_manager or not predictive_system:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Get machine state
        state = machine_state_manager.get_machine_state(machine_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

        # Get prediction (using mock features for now)
        features = np.array([[
            state.temperature or 70.0,
            state.vibration or 2.0,
            state.pressure or 90.0,
            state.speed or 1000.0,
            state.health_score,
            state.cycle_count / 10000.0,  # normalize
        ]])

        # Mock recommendation (in real system, use trained models)
        failure_prob = max(0, min(1, 1.0 - state.health_score + np.random.random() * 0.1))
        rul_hours = max(10, 500 * state.health_score + np.random.random() * 100)

        # Determine urgency
        if failure_prob > 0.7 or state.health_score < 0.5:
            urgency = "critical"
            action = "Immediate maintenance required"
            downtime = 4.0
        elif failure_prob > 0.5 or state.health_score < 0.7:
            urgency = "high"
            action = "Schedule maintenance within 24 hours"
            downtime = 2.0
        elif failure_prob > 0.3 or state.health_score < 0.85:
            urgency = "medium"
            action = "Schedule maintenance within 1 week"
            downtime = 1.0
        else:
            urgency = "low"
            action = "Routine maintenance during next scheduled downtime"
            downtime = 0.5

        return {
            "machine_id": machine_id,
            "timestamp": datetime.now().isoformat(),
            "failure_probability": round(failure_prob, 3),
            "failure_risk_level": "high" if failure_prob > 0.5 else "medium" if failure_prob > 0.3 else "low",
            "remaining_useful_life_hours": round(rul_hours, 1),
            "health_score": state.health_score,
            "urgency": urgency,
            "recommended_action": action,
            "estimated_downtime_hours": downtime,
            "confidence": 0.85,
            "contributing_factors": [
                f"Health score: {state.health_score:.2%}",
                f"Temperature: {state.temperature}°C" if state.temperature else "Temperature: Normal",
                f"Vibration: {state.vibration} mm/s" if state.vibration else "Vibration: Normal",
                f"Operating hours: {state.cycle_count} cycles",
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting maintenance recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predictive/maintenance/all")
async def get_all_maintenance_recommendations():
    """Get maintenance recommendations for all machines."""
    if not machine_state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        all_machines = machine_state_manager.get_all_machines()
        recommendations = []

        for machine_id in all_machines.keys():
            try:
                rec = await get_maintenance_recommendation(machine_id)
                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Failed to get recommendation for {machine_id}: {e}")
                continue

        return recommendations
    except Exception as e:
        logger.error(f"Error getting all recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Production Scheduling Endpoints (NEW)
# ============================================================================

@app.get("/api/v1/scheduling/current")
async def get_current_schedule():
    """Get current production schedule."""
    if not production_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        # Get the current schedule from the scheduler
        schedule = production_scheduler.get_current_schedule()

        if not schedule:
            return {
                "schedule_id": "default",
                "jobs": [],
                "assignments": [],
                "makespan": 0,
                "utilization": 0,
                "created_at": datetime.now().isoformat(),
            }

        return schedule
    except Exception as e:
        logger.error(f"Error getting current schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scheduling/schedules")
async def get_schedules():
    """Get all production schedules."""
    if not production_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        schedules = production_scheduler.get_all_schedules()
        return schedules if schedules else []
    except Exception as e:
        logger.error(f"Error getting schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/scheduling/schedule")
async def create_schedule(request: Dict[str, Any]):
    """Create a new production schedule."""
    if not production_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        jobs_data = request.get("jobs", [])

        # Convert dict to Job objects
        jobs = []
        for job_data in jobs_data:
            job = Job(
                job_id=job_data.get("job_id"),
                operations=job_data.get("operations", []),
                priority=job_data.get("priority", 1),
                deadline=job_data.get("deadline"),
                duration=job_data.get("duration", 0),
            )
            jobs.append(job)

        # Create schedule
        schedule = production_scheduler.schedule_jobs(jobs)

        # Broadcast update via WebSocket
        await broadcast_update({
            "type": "schedule_update",
            "data": schedule
        })

        return schedule
    except Exception as e:
        logger.error(f"Error creating schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scheduling/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if not production_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        job = production_scheduler.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard Stats Endpoint (NEW) - with TTL caching
# ============================================================================

def get_cache_key(prefix: str) -> str:
    """Generate cache key with TTL bucket."""
    current_bucket = int(datetime.now().timestamp() // DASHBOARD_CACHE_TTL)
    return f"{prefix}:{current_bucket}"

def cleanup_old_cache():
    """Remove expired cache entries."""
    current_time = datetime.now().timestamp()
    keys_to_delete = []

    for key in list(dashboard_cache.keys()):
        if ':' in key:
            try:
                bucket = int(key.split(':')[1])
                age = current_time - (bucket * DASHBOARD_CACHE_TTL)
                if age > 60:  # Remove entries older than 1 minute
                    keys_to_delete.append(key)
            except (ValueError, IndexError):
                pass

    for key in keys_to_delete:
        del dashboard_cache[key]

@app.get("/api/v1/dashboard/stats")
async def get_dashboard_stats():
    """Get aggregated dashboard statistics from all services (with 5s cache)."""
    global dashboard_cache

    # Check cache
    cache_key = get_cache_key("dashboard_stats")
    if cache_key in dashboard_cache:
        logger.debug("Dashboard stats: cache HIT")
        return dashboard_cache[cache_key]

    logger.debug("Dashboard stats: cache MISS, computing...")

    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cached": False,  # Indicate this is fresh data
            "services_status": {
                "vision_ai": inference_engine is not None,
                "digital_twin": machine_state_manager is not None,
                "simulator": factory_simulator is not None,
                "predictive": predictive_system is not None,
                "scheduler": production_scheduler is not None,
            },
        }

        # Machine statistics
        if machine_state_manager:
            all_machines = machine_state_manager.get_all_machines()
            machines_data = list(all_machines.values())

            stats["machines"] = {
                "total": len(machines_data),
                "running": sum(1 for m in machines_data if m.state == "running"),
                "idle": sum(1 for m in machines_data if m.state == "idle"),
                "maintenance": sum(1 for m in machines_data if m.state == "maintenance"),
                "error": sum(1 for m in machines_data if m.state == "error"),
                "avg_health": sum(m.health_score for m in machines_data) / len(machines_data) if machines_data else 0,
                "total_cycles": sum(m.cycle_count for m in machines_data),
                "total_defects": sum(m.defect_count for m in machines_data),
            }
        else:
            stats["machines"] = None

        # Predictive maintenance statistics
        if predictive_system and machine_state_manager:
            all_machines = machine_state_manager.get_all_machines()
            critical_machines = []
            high_risk_machines = []

            for machine_id in all_machines.keys():
                try:
                    rec = await get_maintenance_recommendation(machine_id)
                    if rec["urgency"] == "critical":
                        critical_machines.append(machine_id)
                    elif rec["urgency"] == "high":
                        high_risk_machines.append(machine_id)
                except:
                    continue

            stats["maintenance"] = {
                "critical_alerts": len(critical_machines),
                "high_risk": len(high_risk_machines),
                "critical_machines": critical_machines,
                "high_risk_machines": high_risk_machines,
            }
        else:
            stats["maintenance"] = None

        # Scheduling statistics
        if production_scheduler:
            try:
                schedules = production_scheduler.get_all_schedules()
                if schedules:
                    all_jobs = []
                    for schedule in schedules:
                        all_jobs.extend(schedule.get("jobs", []))

                    stats["scheduling"] = {
                        "total_schedules": len(schedules),
                        "total_jobs": len(all_jobs),
                        "completed_jobs": sum(1 for j in all_jobs if j.get("status") == "completed"),
                        "in_progress_jobs": sum(1 for j in all_jobs if j.get("status") == "in_progress"),
                        "pending_jobs": sum(1 for j in all_jobs if j.get("status") == "pending"),
                        "overdue_jobs": sum(1 for j in all_jobs if j.get("is_overdue", False)),
                    }
                else:
                    stats["scheduling"] = {
                        "total_schedules": 0,
                        "total_jobs": 0,
                        "completed_jobs": 0,
                        "in_progress_jobs": 0,
                        "pending_jobs": 0,
                        "overdue_jobs": 0,
                    }
            except:
                stats["scheduling"] = None
        else:
            stats["scheduling"] = None

        # Overall OEE calculation
        if machine_state_manager:
            machines_data = list(machine_state_manager.get_all_machines().values())
            if machines_data:
                stats["overall_oee"] = sum(m.health_score for m in machines_data) / len(machines_data)
            else:
                stats["overall_oee"] = 0
        else:
            stats["overall_oee"] = 0

        # Cache the result
        dashboard_cache[cache_key] = stats
        logger.debug(f"Dashboard stats cached with key: {cache_key}")

        # Cleanup old cache entries
        cleanup_old_cache()

        return stats
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint (NEW)
# ============================================================================

async def safe_send(ws: WebSocket, message: dict):
    """Safely send message to WebSocket client."""
    try:
        await ws.send_json(message)
        return True
    except Exception as e:
        logger.warning(f"WebSocket send failed: {e}")
        raise


async def broadcast_update(message: dict):
    """Broadcast update to all connected WebSocket clients (parallel)."""
    global websocket_clients

    if not websocket_clients:
        return

    # Parallel broadcast using asyncio.gather
    tasks = [safe_send(ws, message) for ws in websocket_clients]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out disconnected clients (O(N) operation)
    websocket_clients = [
        ws for ws, result in zip(websocket_clients, results)
        if not isinstance(result, Exception)
    ]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_clients)}")

    try:
        # Send initial state
        if machine_state_manager:
            all_machines = machine_state_manager.get_all_machines()
            await websocket.send_json({
                "type": "initial_state",
                "data": {
                    "machines": {
                        machine_id: {
                            "machine_id": machine_id,
                            "status": state.state,
                            "health_score": state.health_score,
                            "temperature": state.temperature,
                            "vibration": state.vibration,
                        }
                        for machine_id, state in all_machines.items()
                    }
                }
            })

        # Keep connection alive and listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    # Client subscribing to specific updates
                    await websocket.send_json({
                        "type": "subscribed",
                        "data": message.get("topics", [])
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(websocket_clients)}")


# ============================================================================
# Background Tasks for Real-time Updates (NEW)
# ============================================================================

async def simulate_factory_updates():
    """Simulate factory updates and broadcast to WebSocket clients with graceful shutdown."""
    await asyncio.sleep(5)  # Wait for startup to complete

    logger.info("Starting factory simulation updates...")

    error_count = 0
    MAX_ERRORS = 10

    while not shutdown_event.is_set():
        try:
            # Skip simulation if no clients connected (resource optimization)
            if not websocket_clients:
                await asyncio.sleep(2)
                continue

            if machine_state_manager and factory_simulator:
                # Run simulation step
                factory_simulator.run_step()

                # Update machine states from simulator
                all_machines = machine_state_manager.get_all_machines()

                for machine_id, state in all_machines.items():
                    # Simulate state changes
                    if np.random.random() < 0.1:  # 10% chance of state change
                        # Randomly update temperature, vibration
                        state.temperature = max(60, min(100, state.temperature + np.random.randn() * 2))
                        state.vibration = max(0, min(10, state.vibration + np.random.randn() * 0.5))
                        state.health_score = max(0.5, min(1.0, state.health_score + np.random.randn() * 0.02))

                        # Update cycle count
                        if state.state == "running":
                            state.cycle_count += 1

                # Broadcast update
                await broadcast_update({
                    "type": "factory_update",
                    "data": {
                        "timestamp": datetime.now().isoformat(),
                        "machines": {
                            machine_id: {
                                "machine_id": machine_id,
                                "status": state.state,
                                "health_score": state.health_score,
                                "temperature": state.temperature,
                                "vibration": state.vibration,
                                "cycle_count": state.cycle_count,
                            }
                            for machine_id, state in all_machines.items()
                        }
                    }
                })

            # Reset error count on success
            error_count = 0

            # Update every 2 seconds (cancellation point)
            await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.info("Factory simulation cancelled")
            break
        except Exception as e:
            error_count += 1
            logger.error(f"Error in factory simulation ({error_count}/{MAX_ERRORS}): {e}")

            # Stop if too many errors
            if error_count >= MAX_ERRORS:
                logger.error("Too many errors in simulation, stopping")
                break

            await asyncio.sleep(5)

    logger.info("Factory simulation stopped")


# ============================================================================
# Integration Logic - Service Coordination (NEW)
# ============================================================================

@app.post("/api/v1/integration/process-defect")
async def process_defect_detection(request: Dict[str, Any]):
    """
    Integration endpoint: Process defect detection and trigger downstream actions.
    Flow: Vision AI → Machine State Update → Predictive Analysis → Schedule Adjustment
    """
    try:
        machine_id = request.get("machine_id")
        defect_detected = request.get("defect_detected", False)
        image_path = request.get("image_path")

        if not machine_id:
            raise HTTPException(status_code=400, detail="machine_id required")

        logger.info(f"Processing defect detection for {machine_id}: defect={defect_detected}")

        # Step 1: Update machine state
        if machine_state_manager and defect_detected:
            state = machine_state_manager.get_machine_state(machine_id)
            if state:
                state.defect_count += 1
                state.defect_rate = state.defect_count / max(1, state.cycle_count)
                state.health_score = max(0.3, state.health_score - 0.05)
                logger.info(f"Updated {machine_id}: defect_count={state.defect_count}, health={state.health_score:.2f}")

        # Step 2: Get predictive maintenance recommendation
        maintenance_rec = None
        if predictive_system and machine_state_manager:
            try:
                maintenance_rec = await get_maintenance_recommendation(machine_id)
                logger.info(f"Maintenance recommendation for {machine_id}: urgency={maintenance_rec.get('urgency')}")
            except:
                pass

        # Step 3: If critical, trigger schedule adjustment
        schedule_adjusted = False
        if maintenance_rec and maintenance_rec.get("urgency") == "critical":
            if production_scheduler and machine_state_manager:
                # Mark machine as unavailable
                state = machine_state_manager.get_machine_state(machine_id)
                if state:
                    state.state = "maintenance"
                    logger.info(f"Marked {machine_id} for maintenance")
                    schedule_adjusted = True

        # Step 4: Broadcast integrated update
        await broadcast_update({
            "type": "defect_detected",
            "data": {
                "machine_id": machine_id,
                "defect_detected": defect_detected,
                "image_path": image_path,
                "maintenance_recommendation": maintenance_rec,
                "schedule_adjusted": schedule_adjusted,
                "timestamp": datetime.now().isoformat(),
            }
        })

        return {
            "success": True,
            "machine_id": machine_id,
            "actions_taken": {
                "machine_state_updated": machine_state_manager is not None,
                "maintenance_analyzed": maintenance_rec is not None,
                "schedule_adjusted": schedule_adjusted,
            },
            "maintenance_recommendation": maintenance_rec,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing defect detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
