from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as query_router

app=FastAPI(title="Medical Assistant API", description="API for AI Medical Assistant ChatBot")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Medical Chatbot API running"}

#CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#Middleware exception handlers
app.middleware("http")(catch_exception)

# Routers: 1. Upload PDF, 2. Asking query

# upload pdfs documents
app.include_router(upload_router)

# Asking query
app.include_router(query_router)